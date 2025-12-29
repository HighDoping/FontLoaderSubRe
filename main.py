import ctypes
import hashlib
import json
import logging
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from multiprocessing import Pool, cpu_count, freeze_support
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, ClassVar, Optional

from fontTools.ttLib import TTCollection, TTFont, TTLibFileIsCollectionError
from platformdirs import user_config_dir
from tkinterdnd2 import DND_FILES, TkinterDnD

logger = logging.getLogger(__name__)

class FontMetadataExtractor:
    """Extracts metadata from font files using fontTools."""

    LANG_ID_MAP: ClassVar[[int, str]] = {
        0x0409: "English (US)",
        0x0809: "English (UK)",
        0x0411: "Japanese",
        0x0412: "Korean",
        0x0804: "Chinese (Simplified)",
        0x0404: "Chinese (Traditional/Taiwan)",
        0x0C04: "Chinese (Traditional/Hong Kong)",
    }

    def __init__(self):
        pass

    def analyze(self, filepath: str | Path) -> dict[str, Any]:
        path_obj = Path(filepath)
        if not path_obj.exists():
            return {"error": f"File not found: {path_obj}", "filepath": str(path_obj)}

        try:
            fonts_data = []

            # Helper to safely open font
            def process_ttfont(pt):
                font = TTFont(pt)
                data = self._extract_font_details(font)
                font.close()
                return data

            try:
                # Attempt to open as single font first (faster common case)
                fonts_data.append(process_ttfont(path_obj))
            except TTLibFileIsCollectionError:
                # Fallback to collection
                ttc = TTCollection(path_obj)
                for i, font in enumerate(ttc):
                    fonts_data.append(self._extract_font_details(font, i))
            except Exception:
                # If TTFont fails, it might be a true collection or invalid
                # Try collection explicitly one last time
                try:
                    ttc = TTCollection(path_obj)
                    for i, font in enumerate(ttc):
                        fonts_data.append(self._extract_font_details(font, i))
                except Exception as e:
                    return {
                        "error": f"Font parsing error: {e}",
                        "filepath": str(path_obj),
                    }

            # Calculate MD5
            with path_obj.open("rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()  # noqa: S324

            return {"filepath": str(path_obj), "fonts": fonts_data, "hash": file_hash}

        except Exception as e:
            return {"error": str(e), "filepath": str(path_obj)}

    def _extract_font_details(
        self, font: TTFont, font_index: int = 0
    ) -> dict[str, Any]:
        info = {
            "index": font_index,
            "format": self._determine_format(font),
            "version": None,
            "ps_name": None,
            "unique_id": None,
            "localized": {},
        }
        name_table = font.get("name")
        if name_table:
            info["ps_name"] = self._get_specific_name(name_table, 6)
            info["unique_id"] = self._get_specific_name(name_table, 3)
            info["version"] = self._get_specific_name(name_table, 5)
            info["localized"] = self._extract_localized_names(name_table)
        return info

    def _determine_format(self, font: TTFont) -> str:
        if font.sfntVersion == "OTTO":
            return "OpenType (CFF)"
        elif font.sfntVersion == "\x00\x01\x00\x00":
            return "TrueType"
        elif font.sfntVersion == "wOFF":
            return "WOFF"
        return f"Unknown ({font.sfntVersion})"

    def _extract_localized_names(self, name_table):
        localized_data = {}
        for record in name_table.names:
            if record.nameID not in [1, 4]:
                continue
            if record.platformID == 3:  # Windows
                lang_id = record.langID
                if lang_id in self.LANG_ID_MAP:
                    lang_name = self.LANG_ID_MAP[lang_id]
                    text = self._get_decoded_string(record)
                    if not text:
                        continue
                    if lang_name not in localized_data:
                        localized_data[lang_name] = {"family": None, "full": None}
                    if record.nameID == 1:
                        localized_data[lang_name]["family"] = text
                    elif record.nameID == 4:
                        localized_data[lang_name]["full"] = text

        # English Fallback
        if "English (US)" not in localized_data:
            default_fam = self._get_specific_name(name_table, 1)
            default_full = self._get_specific_name(name_table, 4)
            if default_fam != "N/A" or default_full != "N/A":
                localized_data["English/Default"] = {
                    "family": default_fam,
                    "full": default_full,
                }
        return localized_data

    def _get_decoded_string(self, record) -> Optional[str]:
        try:
            text = record.toUnicode()
            if "\x00" in text:
                raw_bytes = record.string
                try:
                    text = raw_bytes.decode("utf-16-be")
                except UnicodeDecodeError:
                    text = text.replace("\x00", "")
            return text.strip()
        except UnicodeDecodeError:
            return None
        except Exception:
            return None

    def _get_specific_name(self, name_table, name_id) -> str:
        rec = name_table.getName(name_id, 3, 1, 0x0409)  # Win Eng
        if rec:
            return rec.toUnicode().strip()
        rec = name_table.getName(name_id, 1, 0, 0)  # Mac Roman
        if rec:
            return rec.toUnicode().strip()
        for record in name_table.names:  # Fallback
            if record.nameID == name_id:
                res = self._get_decoded_string(record)
                if res:
                    return res
        return "N/A"

    def db_decode_file(self, path: Path) -> dict:
        result = self.analyze(path)
        if "error" in result:
            return {"error": result["error"]}

        res_dict = {
            "filepath": result["filepath"],
            "hash": result.get("hash"),
            "ps_names": set(),
            "unique_ids": set(),
            "family_names": set(),
            "full_names": set(),
        }

        for font in result["fonts"]:
            if font.get("ps_name") and font["ps_name"] != "N/A":
                res_dict["ps_names"].add(font["ps_name"])
            if font.get("unique_id") and font["unique_id"] != "N/A":
                res_dict["unique_ids"].add(font["unique_id"])
            for names in font.get("localized", {}).values():
                if names.get("family"):
                    res_dict["family_names"].add(names["family"])
                if names.get("full"):
                    res_dict["full_names"].add(names["full"])

        res_dict["ps_names"].discard("")
        res_dict["unique_ids"].discard("")
        res_dict["family_names"].discard("")
        res_dict["full_names"].discard("")

        # Convert sets to lists for serialization
        return {k: list(v) if isinstance(v, set) else v for k, v in res_dict.items()}


class FontDatabase:
    """Manages the SQLite database for font metadata storage and retrieval."""

    def __init__(self, db_name="FontLoaderSubRe.db"):
        self.db_name = db_name
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_name)
        # PERFORMANCE PRAGMAS
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS files (
                    file_hash TEXT PRIMARY KEY,
                    path TEXT
                );
                CREATE TABLE IF NOT EXISTS file_fullname (
                    file_hash TEXT, font_name TEXT,
                    FOREIGN KEY(file_hash) REFERENCES files(file_hash) ON DELETE CASCADE,
                    UNIQUE(file_hash, font_name)
                );
                CREATE TABLE IF NOT EXISTS file_families (
                    file_hash TEXT, family_name TEXT,
                    FOREIGN KEY(file_hash) REFERENCES files(file_hash) ON DELETE CASCADE,
                    UNIQUE(file_hash, family_name)
                );
                CREATE TABLE IF NOT EXISTS file_psnames (
                    file_hash TEXT, ps_name TEXT,
                    FOREIGN KEY(file_hash) REFERENCES files(file_hash) ON DELETE CASCADE,
                    UNIQUE(file_hash, ps_name)
                );
                CREATE TABLE IF NOT EXISTS file_uniqueids (
                    file_hash TEXT, unique_id TEXT,
                    FOREIGN KEY(file_hash) REFERENCES files(file_hash) ON DELETE CASCADE,
                    UNIQUE(file_hash, unique_id)
                );
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_font_name ON file_fullname (font_name);
                CREATE INDEX IF NOT EXISTS idx_family_name ON file_families (family_name);
                CREATE INDEX IF NOT EXISTS idx_ps_name ON file_psnames (ps_name);
                CREATE INDEX IF NOT EXISTS idx_unique_id ON file_uniqueids (unique_id);
            """
            )

    def bulk_insert_batch(
        self,
        files_data: list[tuple],
        fonts_map: dict[str, list[tuple]],
    ):
        """Inserts a massive batch of data in a SINGLE transaction.

        files_data: [(hash, path), ...]
        fonts_map: {'full': [(hash, name), ...], 'ps': ...}
        """
        if not files_data:
            return

        with self._get_conn() as conn:
            cursor = conn.cursor()

            # 1. Update/Insert Files (Upsert)
            cursor.executemany(
                """
                INSERT INTO files (file_hash, path) VALUES (?, ?)
                ON CONFLICT(file_hash) DO UPDATE SET path=excluded.path
            """,
                files_data,
            )

            # 2. Insert Names (Ignore duplicates)
            cursor.executemany(
                "INSERT OR IGNORE INTO file_fullname (file_hash, font_name) VALUES (?, ?)",
                fonts_map["full"],
            )
            cursor.executemany(
                "INSERT OR IGNORE INTO file_families (file_hash, family_name) VALUES (?, ?)",
                fonts_map["family"],
            )
            cursor.executemany(
                "INSERT OR IGNORE INTO file_psnames (file_hash, ps_name) VALUES (?, ?)",
                fonts_map["ps"],
            )
            cursor.executemany(
                "INSERT OR IGNORE INTO file_uniqueids (file_hash, unique_id) VALUES (?, ?)",
                fonts_map["unique"],
            )

            conn.commit()

    def metadata_get(self, key: str) -> Optional[str]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else None

    def metadata_set(self, key: str, value: str):
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO metadata (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
                (key, value),
            )

    def metadata_delete(self, key: str):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM metadata WHERE key = ?", (key,))

    def clean_database(self):
        with self._get_conn() as conn:
            conn.executescript(
                """
                DELETE FROM files;
                DELETE FROM file_fullname;
                DELETE FROM file_families;
                DELETE FROM file_psnames;
                DELETE FROM file_uniqueids;
                VACUUM;
            """
            )
            self.metadata_delete("last_scan")
            self.metadata_delete("last_scan_time")
            self.metadata_delete("file_count")
            self.metadata_delete("ps_name_count")
            self.metadata_delete("family_name_count")
            self.metadata_delete("full_name_count")
            self.metadata_delete("unique_id_count")

    def search_by_font(self, font_name):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            wild = f"%{font_name}%"
            # Exact match for font_name, LIKE for other names
            query = """
                SELECT DISTINCT f.path, f.file_hash 
                FROM files f
                LEFT JOIN file_fullname ff ON f.file_hash = ff.file_hash
                LEFT JOIN file_families fam ON f.file_hash = fam.file_hash
                LEFT JOIN file_psnames fps ON f.file_hash = fps.file_hash
                LEFT JOIN file_uniqueids fuid ON f.file_hash = fuid.file_hash
                WHERE ff.font_name = ? 
                   OR fam.family_name LIKE ? 
                   OR fps.ps_name LIKE ? 
                   OR fuid.unique_id LIKE ?
            """
            cursor.execute(query, (font_name, wild, wild, wild))
            results = cursor.fetchall()

            # If no results and font_name starts with '@', try again without '@'
            if not results and font_name.startswith("@"):
                stripped_name = font_name[1:]
                wild = f"%{stripped_name}%"
                cursor.execute(query, (stripped_name, wild, wild, wild))
                results = cursor.fetchall()

            return results

    def table_length(self,table_name:str)->int:
        # Whitelist of allowed table names to prevent SQL injection
        allowed_tables = {
            "files",
            "file_fullname",
            "file_families",
            "file_psnames",
            "file_uniqueids",
            "metadata",
        }
        if table_name not in allowed_tables:
            raise ValueError(f"Invalid table name: {table_name}")

        with self._get_conn() as conn:
            cursor = conn.cursor()
            # Safe to use f-string here because table_name is validated against whitelist
            query = f"SELECT COUNT(*) FROM {table_name}"  # noqa: S608
            cursor.execute(query)
            row = cursor.fetchone()
            return row[0] if row else 0


class SessionFontManager:
    """Manages loading and unloading of fonts for the current session.

    Tracks detailed status of every font attempt using MD5 hashes.
    """

    def __init__(self):
        self.current_os = platform.system()
        self.status: dict[str, dict[str, Any]] = {}

    def _calculate_file_md5(self, path: Path) -> str:
        """Calculates the MD5 hash of a file by reading the whole file at once."""
        try:
            with path.open("rb") as f:
                return hashlib.md5(f.read()).hexdigest().lower()
        except OSError as e:
            logger.error("Failed to read file for hashing '%s': %s", path, e)
            return ""

    def _get_font_destination_path(
        self, font_path: Path
    ) -> tuple[Optional[Path], Optional[Path]]:
        """Determines the correct user-level font directory for macOS and Linux."""
        home_dir = Path.home()
        font_filename = font_path.name

        if self.current_os == "Darwin":  # macOS
            dest_dir = home_dir / "Library" / "Fonts"
            return dest_dir, dest_dir / font_filename
        elif self.current_os == "Linux":
            dest_dir = home_dir / ".local" / "share" / "fonts"
            return dest_dir, dest_dir / font_filename
        return None, None

    def load_font(self, font_path: Path, expected_hash: str) -> bool:
        """
        Verifies integrity and loads a font. Updates self.status with the result.
        """
        expected_hash = expected_hash.lower()

        # 1. Initialize status entry if new
        if expected_hash not in self.status:
            self.status[expected_hash] = {
                "hash": expected_hash,
                "file_path": font_path,  # The source path provided by user
                "sys_path": None,  # Where it is installed (populated on success)
                "loaded": False,
                "message": "Initializing",
            }
        # Update the source path to the current attempt
        # (useful if retrying a failed hash from a new location)
        elif not self.status[expected_hash]["loaded"]:
            self.status[expected_hash]["file_path"] = font_path

        # 2. Check if already loaded
        if self.status[expected_hash]["loaded"]:
            logger.debug("Font %s already active.", expected_hash)
            self.status[expected_hash]["message"] = "Already loaded"
            return True

        # 3. Validation
        if not font_path.is_absolute():
            msg = f"Path must be absolute: {font_path}"
            logger.error(msg)
            self.status[expected_hash]["message"] = msg
            return False

        if not font_path.exists():
            msg = f"File not found: {font_path}"
            logger.error(msg)
            self.status[expected_hash]["message"] = msg
            return False

        # 4. Integrity Check
        logger.debug("Verifying integrity of %s...", font_path.name)
        actual_hash = self._calculate_file_md5(font_path)

        if not actual_hash:
            self.status[expected_hash]["message"] = "Read error during hashing"
            return False

        if actual_hash != expected_hash:
            msg = f"Hash Mismatch. Expected: {expected_hash}, Found: {actual_hash}"
            logger.error(msg)
            self.status[expected_hash]["message"] = msg
            return False

        # 5. OS Loading
        success = False
        sys_path = None

        if self.current_os == "Windows":
            success = self._load_windows(font_path)
            if success:
                sys_path = font_path  # On Windows, source is the system path
        elif self.current_os in ["Darwin", "Linux"]:
            success, sys_path = self._load_unix(font_path)
        else:
            msg = f"Unsupported OS: {self.current_os}"
            logger.error(msg)
            self.status[expected_hash]["message"] = msg
            return False

        # 6. Final Status Update
        self.status[expected_hash]["loaded"] = success
        self.status[expected_hash]["sys_path"] = sys_path
        self.status[expected_hash]["message"] = (
            "Success" if success else "OS Load Failed"
        )

        return success

    def _load_windows(self, font_path: Path) -> bool:
        try:
            gdi32 = ctypes.WinDLL("gdi32")
            add_font_resource_w = gdi32.AddFontResourceW
            add_font_resource_w.argtypes = [ctypes.c_wchar_p]
            add_font_resource_w.restype = ctypes.c_int

            result = add_font_resource_w(str(font_path))
            if result > 0:
                logger.info("Windows: Loaded font %s", font_path.name)
                return True
            return False
        except Exception as e:
            logger.error("Windows loading error: %s", e)
            return False

    def _load_unix(self, font_path: Path) -> tuple[bool, Optional[Path]]:
        dest_dir, dest_path = self._get_font_destination_path(font_path)
        if not dest_dir or not dest_path:
            return False, None

        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            if dest_path.exists():
                logger.debug("Unix: Font exists, skipping copy.")
                return True, dest_path

            shutil.copy(font_path, dest_path)
            if self.current_os == "Linux":
                subprocess.run(["fc-cache", "-f"], check=True, capture_output=True)

            logger.info("Unix: Installed font to %s", dest_path)
            return True, dest_path
        except Exception as e:
            logger.error("Unix loading error: %s", e)
            return False, None

    def unload_font(self, file_hash: str) -> bool:
        """Unloads a specific font by its hash if it is currently loaded."""
        file_hash = file_hash.lower()

        if file_hash not in self.status:
            return False

        font_info = self.status[file_hash]
        if not font_info["loaded"]:
            return False

        # Use the system path for unloading (important for Unix where file moved)
        path_to_remove = font_info["sys_path"]
        if not path_to_remove:
            return False

        success = False
        if self.current_os == "Windows":
            success = self._unload_windows(path_to_remove)
        elif self.current_os in ["Darwin", "Linux"]:
            success = self._unload_unix(path_to_remove)

        if success:
            logger.info("Unloaded font: %s", path_to_remove.name)
            self.status[file_hash]["loaded"] = False
            self.status[file_hash]["message"] = "Unloaded"
        else:
            logger.error("Failed to unload font: %s", path_to_remove.name)
            self.status[file_hash]["message"] = "Unload Failed"

        return success

    def _unload_windows(self, path: Path) -> bool:
        gdi32 = ctypes.WinDLL("gdi32")
        remove_font_resource_w = gdi32.RemoveFontResourceW
        remove_font_resource_w.argtypes = [ctypes.c_wchar_p]
        remove_font_resource_w.restype = ctypes.c_int
        return remove_font_resource_w(str(path)) != 0

    def _unload_unix(self, path: Path) -> bool:
        try:
            if path.exists():
                path.unlink()
                if self.current_os == "Linux":
                    subprocess.run(["fc-cache", "-f"], check=True, capture_output=True)
            return True
        except Exception as e:
            logger.error("Error removing font %s: %s", path, e)
            return False

    def cleanup(self):
        """Iterates through status and unloads all loaded fonts."""
        logger.debug("Starting cleanup...")
        hashes = list(self.status.keys())
        for h in hashes:
            if self.status[h]["loaded"]:
                self.unload_font(h)
        logger.debug("Cleanup complete.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def extract_ass_fonts(path: Path):
    """Accepts a filepath to a single .ass file or a dirpath containing .ass files.

    Returns a list of unique font names found in the [V4+ Styles] / [V4 Styles] sections.
    """
    fonts = set()
    p = Path(path)

    logging.debug(f"Extract file: {p}")

    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = [
            f
            for f in p.rglob("*")
            if f.is_file() and f.suffix.lower() in {".ass", ".ssa"}
        ]
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    for fp in files:
        for enc in ["utf-8", "utf-16", "gb18030", "utf-8-sig"]:
            try:
                with fp.open("r", encoding=enc) as fh:
                    lines = fh.read().splitlines()
                # If successful, break and process
                break
            except UnicodeError:
                continue

        in_styles_section = False
        font_name_index = -1

        for line in lines:
            line = line.strip()

            # Enter Styles section
            if line in {"[V4+ Styles]", "[V4 Styles]"}:
                in_styles_section = True
                continue

            # Exit Styles section when a new non-V4 section starts
            if (
                in_styles_section
                and line.startswith("[")
                and not line.startswith("[V4")
            ):
                break

            if not in_styles_section:
                continue

            # Parse Format line to find "Fontname" index
            if line.startswith("Format:"):
                format_headers = [x.strip() for x in line[7:].split(",")]
                try:
                    font_name_index = format_headers.index("Fontname")
                except ValueError:
                    font_name_index = 1  # fallback
            # Parse Style lines
            elif line.startswith("Style:") and font_name_index != -1:
                parts = line[6:].split(",")
                if len(parts) > font_name_index:
                    fonts.add(parts[font_name_index].strip())

    return len(files), list(fonts)


def _worker_process_font(args):
    """Worker function. DOES NOT interact with DB. Returns: dict or None"""
    path, dir_path = args
    extractor = FontMetadataExtractor()
    try:
        data = extractor.db_decode_file(path)
        if "error" in data:
            return None
        # Make path relative string here to save main thread CPU
        rel_path = str(path.relative_to(dir_path))
        data["relative_path"] = rel_path
        return data
    except Exception:
        return None


def scan_fonts_in_directory(db: FontDatabase, dir_path: Path):
    font_extensions = {".ttf", ".otf", ".ttc"}
    font_files = [
        p
        for p in dir_path.rglob("*")
        if p.suffix.lower() in font_extensions and p.is_file()
    ]

    # Initialize stats to match requested return format
    final_stats = {
        "files_processed": 0,
        "unique_font_names": {"full": 0, "family": 0, "ps": 0, "unique": 0},
    }

    if not font_files:
        return final_stats

    # 1. Clean DB
    db.clean_database()

    # 2. Setup Batching
    BATCH_SIZE = 500
    batch_files = []
    batch_names = {"full": [], "family": [], "ps": [], "unique": []}

    # 3. Process
    with Pool(processes=min(16, cpu_count())) as pool:
        args_iter = [(p, dir_path) for p in font_files]

        for result in pool.imap_unordered(
            _worker_process_font, args_iter, chunksize=10
        ):
            if not result:
                continue

            f_hash = result["hash"]

            # Prepare data for DB
            batch_files.append((f_hash, result["relative_path"]))

            for n in result["full_names"]:
                batch_names["full"].append((f_hash, n))
            for n in result["family_names"]:
                batch_names["family"].append((f_hash, n))
            for n in result["ps_names"]:
                batch_names["ps"].append((f_hash, n))
            for n in result["unique_ids"]:
                batch_names["unique"].append((f_hash, n))

            # Update Statistics
            final_stats["files_processed"] += 1
            final_stats["unique_font_names"]["full"] += len(result["full_names"])
            final_stats["unique_font_names"]["family"] += len(result["family_names"])
            final_stats["unique_font_names"]["ps"] += len(result["ps_names"])
            final_stats["unique_font_names"]["unique"] += len(result["unique_ids"])

            # Flush Batch if full
            if len(batch_files) >= BATCH_SIZE:
                db.bulk_insert_batch(batch_files, batch_names)
                # Reset buffers
                batch_files = []
                batch_names = {"full": [], "family": [], "ps": [], "unique": []}
                logger.info(
                    "Processed %d / %d files...",
                    final_stats["files_processed"],
                    len(font_files),
                )

        # Flush remaining data
        if batch_files:
            db.bulk_insert_batch(batch_files, batch_names)
    # write final stats to metadata
    db.metadata_set("last_scan", dir_path.as_posix())
    db.metadata_set(
        "last_scan_time",
        str(int(sqlite3.datetime.datetime.now().timestamp())),
    )
    # db.metadata_set("file_count", str(final_stats["files_processed"]))

    db.metadata_set("file_count", str(db.table_length("files")))
    db.metadata_set("ps_name_count", str(db.table_length("file_psnames")))
    db.metadata_set(
        "family_name_count", str(db.table_length("file_families"))    )
    db.metadata_set("full_name_count", str(db.table_length("file_fullname")))
    db.metadata_set("unique_id_count", str(db.table_length("file_uniqueids")))

    logging.debug(
        "Font scan complete. Processed %d files.", final_stats["files_processed"]
    )
    return final_stats


class FontLoaderApp:

    def __init__(self, root, sub_path: Optional[Path] = None):
        self.root = root
        self.root.title("FontLoaderSubRe 0.1.1")
        self.root.resizable(True, True)
        self.root.minsize(350, 160)

        try:
            res_dir = Path(__file__).parent / "resources"
            icon_ico = res_dir / "icon.ico"
            icon_png = res_dir / "icon.png"

            if platform.system() == "Darwin":
                if "__compiled__" not in globals() and icon_png.exists():
                    self._icon_img = tk.PhotoImage(file=icon_png)
                    self.root.iconphoto(True, self._icon_img)
            elif platform.system() == "Windows":
                if icon_ico.exists():
                    self.root.iconbitmap(icon_ico)
            elif platform.system() == "Linux" and icon_png.exists():
                self._icon_img = tk.PhotoImage(file=icon_png)
                self.root.iconphoto(True, self._icon_img)
        except Exception:
            logger.debug("Failed to load window icon.", exc_info=True)

        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>", self.on_drop)

        # Center window on screen
        self.root.update_idletasks()
        window_width = self.root.winfo_reqwidth()
        window_height = self.root.winfo_reqheight()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"+{x}+{y}")

        self.project_link = "https://github.com/HighDoping/FontLoaderSubRe"

        self.app_config = AppConfig()
        self.db = FontDatabase(self.app_config.get_db_path())
        self.font_manager = SessionFontManager()

        self.sub_path = sub_path
        self.font_path = self.app_config.get_font_base_path()

        logger.debug("FontLoaderApp initialized.")
        logger.debug(f"Font base path: {self.font_path}")
        logger.debug(f"Subtitle path: {self.sub_path}")
        logger.debug(f"Database path: {self.db.db_name}")

        # --- Dynamic Data (The Numbers) ---
        self.stats = {
            "loaded": 0,
            "errors": 0,
            "no_match": 0,
            "index_fonts": 0,
            "index_names": 0,
            "subtitles": 0,
        }

        # --- State ---
        self.details_visible = False  # New state for foldable details

        # --- Localization Data ---
        self.locales = {
            "zh_cn": {
                "header": "完成",
                "status_1": "{loaded} 个字体加载成功，{errors} 个出错，{no_match} 个无匹配。",
                "status_2": "索引中有 {index_fonts} 个字体，{index_names} 种名称；当前共 {subtitles} 个字幕。",
                "btn_menu": "菜单",
                "btn_details": "详情",
                "btn_close": "关闭",
                "btn_change": "更改...",  # Added
                "menu_update": "更新索引",
                "menu_font_base": "设置字体库",  # Added
                "msg_update_complete": "字体索引更新完成。",
                "menu_export": "导出字体",
                "menu_help": "FontLoaderSubRe帮助",
                "menu_lang": "语言",
                "title_font_base": "字体库路径设置",  # Added
                "lbl_current_path": "当前字体库路径:",  # Added
                "msg_export": "导出字体完成。",
                "msg_help": "使用方法：\n1. 将本程序移动到字体文件夹（Windows）,或在目录中设置字体库路径；\n2. 把字幕或其文件夹拖动到程序、快捷方式（Windows），或窗口上；\n3. 字体库变更后请“更新索引”。",
                "footer": f"GPLv2: {self.project_link}",
            },
            "zh_tw": {
                "header": "完成",
                "status_1": "{loaded} 個字型載入成功，{errors} 個錯誤，{no_match} 個無匹配。",
                "status_2": "索引中有 {index_fonts} 個字型，{index_names} 種名稱；當前共 {subtitles} 個字幕。",
                "btn_menu": "菜單",
                "btn_details": "詳情",
                "btn_close": "關閉",
                "btn_change": "更改...",  # Added
                "menu_update": "更新索引",
                "menu_font_base": "設定字型庫",  # Added
                "msg_update_complete": "字型索引更新完成。",
                "menu_export": "匯出字型",
                "menu_help": "FontLoaderSubRe幫助",
                "menu_lang": "語言",
                "title_font_base": "字型庫路徑設定",  # Added
                "lbl_current_path": "當前字型庫路徑:",  # Added
                "msg_export": "匯出字型完成。",
                "msg_help": "使用方法：\n1. 將本程式移動到字型資料夾（Windows）,或在目錄中設定字型庫路徑；\n2. 把字幕或其資料夾拖動到程式、捷徑（Windows），或視窗上；\n3. 字型庫變更後請“更新索引”。",
                "footer": f"GPLv2: {self.project_link}",
            },
            "en_us": {
                "header": "Finished",
                "status_1": "{loaded} fonts loaded, {errors} errors, {no_match} no match.",
                "status_2": "Index: {index_fonts} fonts, {index_names} names; {subtitles} subtitles.",
                "btn_menu": "Menu",
                "btn_details": "Details",
                "btn_close": "Close",
                "btn_change": "Change...",  # Added
                "menu_update": "Update Index",
                "menu_font_base": "Set Font Base",  # Added
                "msg_update_complete": "Font index update complete.",
                "menu_export": "Export Fonts",
                "menu_help": "FontLoaderSubRe Help",
                "menu_lang": "Language",
                "title_font_base": "Font Base Settings",  # Added
                "lbl_current_path": "Current Base Path:",  # Added
                "msg_export": "Font export complete.",
                "msg_help": "Instructions:\n1. Move this program to your font folder (Windows), or set the font base path in the menu;\n2. Drag and drop subtitle files or folders onto the program, shortcut (Windows), or window;\n3. Please 'Update Index' after changing the font base.",
                "footer": f"GPLv2: {self.project_link}",
            },
        }

        self.create_widgets()
        self.refresh_ui()
        if self.sub_path is None:
            # self.action_help()
            pass
        else:
            self.load_fonts()

    def create_widgets(self):
        # CHANGED: Reduced outer padding (was "15 15 15 5" -> "10 10 10 2")
        main_frame = ttk.Frame(self.root, padding="10 10 10 2")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. Header
        self.lbl_header = ttk.Label(
            main_frame,
            text="",
            foreground="#0033CC",
            font=("", 16, "bold"),
        )
        self.lbl_header.pack(anchor="w", pady=(0, 5))

        # 2. Status Lines (Dynamic Text)
        self.lbl_status1 = ttk.Label(main_frame, text="")
        self.lbl_status1.pack(anchor="w", pady=(0, 0))

        self.lbl_status2 = ttk.Label(main_frame, text="")
        self.lbl_status2.pack(anchor="w", pady=(0, 10))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame, variable=self.progress_var, maximum=100, mode="determinate"
        )
        # Hide it initially
        self.lbl_progress_task = ttk.Label(main_frame, text="", font=("", 8))

        # 3. Details Frame (Hidden by default)
        self.frame_details = ttk.Frame(main_frame)
        self.txt_details = tk.Text(
            self.frame_details,
            height=10,
            width=50,
            state="disabled",
            font=("Consolas", 9),
        )
        scroll_details = ttk.Scrollbar(
            self.frame_details, orient="vertical", command=self.txt_details.yview
        )
        self.txt_details.configure(yscrollcommand=scroll_details.set)

        self.txt_details.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_details.pack(side=tk.RIGHT, fill=tk.Y)

        # 4. Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5, side=tk.BOTTOM, anchor="s")

        self.btn_menu = ttk.Button(btn_frame, command=self.show_menu)
        self.btn_menu.pack(side=tk.LEFT, padx=(0, 5))

        self.btn_details = ttk.Button(btn_frame, command=self.toggle_details)
        self.btn_details.pack(side=tk.LEFT, padx=(0, 5))

        self.btn_close = ttk.Button(btn_frame, command=self.action_close)
        self.btn_close.pack(side=tk.RIGHT, padx=(5, 0))

        # 5. Footer
        footer_frame = ttk.Frame(self.root)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)

        ttk.Separator(footer_frame, orient="horizontal").pack(fill=tk.X)

        link_inner_frame = ttk.Frame(footer_frame, padding="5 2")
        link_inner_frame.pack(fill=tk.X, anchor="w")

        self.lbl_footer = ttk.Label(link_inner_frame, foreground="blue", cursor="hand2")
        self.lbl_footer.pack(side=tk.LEFT)
        self.lbl_footer.bind("<Button-1>", lambda e: webbrowser.open(self.project_link))

    def toggle_details(self):
        """Toggles the visibility of the details log."""
        if self.details_visible:
            self.frame_details.pack_forget()
            self.details_visible = False
            # self.root.geometry("") # Reset to allow shrinking if desired
        else:
            self.frame_details.pack(
                fill=tk.BOTH,
                expand=True,
                before=self.lbl_header.master.pack_slaves()[-1],
            )
            self.details_visible = True

    def refresh_ui(self):
        """Updates all labels based on current stats and language"""
        txt = self.locales[self.app_config.get_language()]

        self.lbl_header.config(text=txt["header"])

        self.updata_stats()
        s1 = txt["status_1"].format(**self.stats)
        s2 = txt["status_2"].format(**self.stats)

        self.lbl_status1.config(text=s1)
        self.lbl_status2.config(text=s2)

        self.btn_menu.config(text=txt["btn_menu"])
        self.btn_details.config(text=txt["btn_details"])
        self.btn_close.config(text=txt["btn_close"])
        self.lbl_footer.config(text=txt["footer"])

    def create_popup_menu(self):
        """Recreates the popup menu to ensure language is correct"""
        self.popup_menu = tk.Menu(self.root, tearoff=0)
        txt = self.locales[self.app_config.get_language()]

        # Actions
        self.popup_menu.add_command(
            label=txt["menu_update"], command=self.action_update_index
        )
        self.popup_menu.add_command(
            label=txt["menu_font_base"], command=self.action_set_font_base
        )
        self.popup_menu.add_command(
            label=txt["menu_export"], command=self.action_export
        )
        self.popup_menu.add_separator()
        self.popup_menu.add_command(label=txt["menu_help"], command=self.action_help)
        self.popup_menu.add_separator()

        # Language Submenu
        lang_menu = tk.Menu(self.popup_menu, tearoff=0)
        lang_menu.add_command(label="English", command=lambda: self.set_lang("en_us"))
        lang_menu.add_command(label="正體中文", command=lambda: self.set_lang("zh_tw"))
        lang_menu.add_command(label="简体中文", command=lambda: self.set_lang("zh_cn"))

        self.popup_menu.add_cascade(label=txt["menu_lang"], menu=lang_menu)

    def show_menu(self):
        self.create_popup_menu()
        # Post menu right under the button
        x = self.btn_menu.winfo_rootx()
        y = self.btn_menu.winfo_rooty() + self.btn_menu.winfo_height()
        self.popup_menu.post(x, y)

    def set_lang(self, lang_code):
        self.app_config.set_language(lang_code)
        self.refresh_ui()

    def updata_stats(self):
        """Fetches latest stats from the database."""
        self.stats["index_fonts"] = int(self.db.metadata_get("file_count") or 0)
        self.stats["index_names"] = sum(
            int(self.db.metadata_get(key) or 0)
            for key in [
                "ps_name_count",
                "family_name_count",
                "full_name_count",
                "unique_id_count",
            ]
        )

    def action_update_index(self):
        """Run the font directory scan in a background thread."""
        self.btn_menu.configure(state="disabled")
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start(10)

        txt = self.locales[self.app_config.get_language()]
        self.lbl_progress_task.config(text=txt["menu_update"] + "...")
        self.lbl_progress_task.pack(anchor="w")

        def run_scan():
            try:
                scan_fonts_in_directory(self.db, Path(self.font_path))
                self.root.after(0, self._on_index_complete)
            except Exception as e:
                logger.error(f"Scan failed: {e}")
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))

        threading.Thread(target=run_scan, daemon=True).start()

    def _on_index_complete(self):
        """Handle UI cleanup after index update."""
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.lbl_progress_task.pack_forget()
        self.btn_menu.configure(state="normal")
        self.refresh_ui()

        txt = self.locales[self.app_config.get_language()]
        messagebox.showinfo(txt["menu_update"], txt["msg_update_complete"])
    def action_set_font_base(self):
        txt = self.locales[self.app_config.get_language()]
        logger.debug("Action: Set Font Base")

        # Create Modal Toplevel Window
        top = tk.Toplevel(self.root)
        top.title(txt["title_font_base"])
        top.geometry("420x130")
        top.resizable(False, False)

        # Center relative to parent
        try:
            x = self.root.winfo_rootx() + (self.root.winfo_width() // 2) - 210
            y = self.root.winfo_rooty() + (self.root.winfo_height() // 2) - 65
            top.geometry(f"+{x}+{y}")
        except Exception:
            pass  # Fallback to default placement if calculation fails

        # UI Elements
        ttk.Label(top, text=txt["lbl_current_path"]).pack(pady=(15, 5), padx=15, anchor="w")

        path_var = tk.StringVar(value=str(self.font_path))
        entry = ttk.Entry(top, textvariable=path_var, state="readonly")
        entry.pack(fill=tk.X, padx=15, pady=5)

        def do_change():
            new_path = filedialog.askdirectory(initialdir=self.font_path, parent=top)
            if new_path:
                self.app_config.set_font_base_path(Path(new_path))
                self.font_path = Path(new_path)
                path_var.set(str(self.font_path))
                logger.info(f"Font base changed to: {self.font_path}")
                top.lift() # Bring focus back to dialog

        # Button Row
        btn_frame = ttk.Frame(top)
        btn_frame.pack(fill=tk.X, pady=10, padx=15, side=tk.BOTTOM)

        ttk.Button(btn_frame, text=txt["btn_close"], command=top.destroy).pack(side=tk.RIGHT)
        ttk.Button(btn_frame, text=txt["btn_change"], command=do_change).pack(side=tk.RIGHT, padx=5)

        # Make Modal
        top.transient(self.root)
        top.grab_set()
        self.root.wait_window(top)

        self.db = FontDatabase(self.app_config.get_db_path())
        self.refresh_ui()

    def action_export(self):
        txt = self.locales[self.app_config.get_language()]
        logger.debug("Action: Export Fonts")

        # Ask for export directory
        export_dir = filedialog.askdirectory(
            title=txt["menu_export"],
            initialdir=str(Path.home()  ),
        )
        if not export_dir:
            return  # User cancelled
        export_path = Path(export_dir)
        # Export loaded fonts
        for font_hash, info in self.font_manager.status.items():
            if info["loaded"]:
                src_path = info["sys_path"]
                if src_path and src_path.exists():
                    dest_path = export_path / src_path.name
                    try:
                        shutil.copy(src_path, dest_path)
                    except Exception as e:
                        logger.error("Failed to export font %s: %s", src_path.name, e)

        messagebox.showinfo(txt["menu_export"], txt["msg_export"])

    def action_help(self):
        txt = self.locales[self.app_config.get_language()]
        logger.debug("Action: Help opened")
        messagebox.showinfo(txt["menu_help"], txt["msg_help"])

    def action_close(self):
        logger.debug("Action: Close clicked")
        self.font_manager.cleanup()
        self.root.destroy()
    def _parse_drop_files(self, data):
        """Parses the string returned by tkinterdnd2.

        Handles paths with spaces (wrapped in {}) and multiple files.
        Returns a list of paths.
        """
        # Regex to capture content inside {} or non-whitespace sequences
        pattern = r'\{(.+?)\}|(\S+)'
        files = []
        for match in re.finditer(pattern, data):
            # group(1) is inside {}, group(2) is a standard word
            path = match.group(1) or match.group(2)
            if path:
                files.append(Path(path))
        return files

    def on_drop(self, event):
        """Handle file drops from Finder/Explorer"""
        dropped_files = self._parse_drop_files(event.data)

        if not dropped_files:
            return

        # Take the first dropped file/folder
        new_path = dropped_files[0]
        logger.info(f"File dropped: {new_path}")

        # Update sub_path and reload
        self.sub_path = new_path
        self.load_fonts()

    def load_fonts(self):
        """Begin the font loading process asynchronously."""
        file_path = self.sub_path
        sub_count, font_list = extract_ass_fonts(file_path)

        if not font_list:
            self.refresh_ui()
            return

        self.txt_details.config(state="normal")
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        self.progress_bar.configure(mode="determinate")
        self.progress_var.set(0)

        # Start incremental processing
        self._process_font_queue(font_list, 0, sub_count, [])

    def _process_font_queue(self, font_list: list[str], index: int, sub_count: int, log_lines: list[str]):
        """Process one font at a time using root.after to keep GUI alive."""
        if index >= len(font_list):
            # Finalize
            self._finalize_load(log_lines, sub_count)
            return

        font = font_list[index]
        self.lbl_progress_task.config(text=f"Loading: {font}")
        self.lbl_progress_task.pack(anchor="w")

        # Update progress bar
        progress = (index / len(font_list)) * 100
        self.progress_var.set(progress)

        # Logic for matching and loading
        res = self.db.search_by_font(font)
        log_line = ""
        if res:
            relative_path_str = res[0][0]
            font_path = (Path(self.font_path) / Path(relative_path_str)).absolute()
            font_hash = res[0][1]
            success = self.font_manager.load_font(font_path, font_hash)

            if success:
                self.stats["loaded"] += 1
                log_line = f"[ok] {font} > {relative_path_str}\n"
            else:
                self.stats["errors"] += 1
                msg = self.font_manager.status.get(font_hash, {}).get("message", "Error")
                log_line = f"[xx] {font} > {relative_path_str} ({msg})\n"
        else:
            self.stats["no_match"] += 1
            log_line = f"[??] {font}\n"

        log_lines.append(log_line)

        # Schedule next font in 10ms (allows UI to breathe)
        self.root.after(10, self._process_font_queue, font_list, index + 1, sub_count, log_lines)

    def _finalize_load(self, log_lines: list[str], sub_count: int):
        """Update logs and UI after all fonts are processed."""
        log_lines.sort(key=lambda x: (0 if x.startswith("[ok]") else 1 if x.startswith("[xx]") else 2))

        for line in log_lines:
            self.txt_details.insert(tk.END, line)

        self.txt_details.see(tk.END)
        self.txt_details.config(state="disabled")

        self.stats["subtitles"] += sub_count
        self.progress_bar.pack_forget()
        self.lbl_progress_task.pack_forget()
        self.refresh_ui()


class AppConfig:
    """Unified interface for application configuration management."""

    APP_NAME = "FontLoaderSubRe"
    APP_AUTHOR = "HighDoping"

    DB_NAME = "FontLoaderSubRe.db"

    def __init__(self):

        self.config_dir = Path(user_config_dir(self.APP_NAME, self.APP_AUTHOR))
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "settings.json"
        self._settings = self._load_settings()

    def _load_settings(self) -> dict:
        """Load settings from file or create defaults."""
        if self.config_file.exists():
            try:
                with self.config_file.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                logger.warning("Failed to load settings, using defaults")

        return {
            "font_base_path": str(Path.cwd()),
            "ui_language": "en_us",
        }

    def _save_settings(self):
        """Persist settings to disk."""
        try:
            with self.config_file.open("w", encoding="utf-8") as f:
                json.dump(self._settings, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")

    def get(self, key: str, default=None):
        """Get a configuration value."""
        return self._settings.get(key, default)

    def set(self, key: str, value):
        """Set a configuration value and save."""
        self._settings[key] = value
        self._save_settings()

    def get_font_base_path(self) -> Path:
        """Get the font base directory."""
        if platform.system() == "Windows" and Path(self.DB_NAME).exists():
            return Path.cwd()
        return Path(self._settings.get("font_base_path", Path.cwd()))

    def set_font_base_path(self, path: Path):
        """Set the font base directory."""
        self.set("font_base_path", str(path))

    def get_db_path(self) -> Path:
        """Get the database file path."""
        return self.get_font_base_path() / self.DB_NAME

    def get_language(self) -> str:
        """Get the UI language."""
        return self._settings.get("ui_language", "en_us")

    def set_language(self, lang_code: str):
        """Set the UI language."""
        self.set("ui_language", lang_code)


if __name__ == "__main__":
    freeze_support()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = TkinterDnD.Tk()

    style = ttk.Style()
    available_themes = style.theme_names()
    if platform.system() == "Windows":
        if "winnative" in available_themes:
            style.theme_use("winnative")
    elif platform.system() == "Darwin":
        if "aqua" in available_themes:
            style.theme_use("aqua")
    elif platform.system() == "Linux":
        if "clam" in available_themes:
            style.theme_use("clam")
    else:
        style.theme_use(available_themes[0])

    sub_path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    sub_path = Path(sub_path_arg) if sub_path_arg else None
    app = FontLoaderApp(root, sub_path=sub_path)

    root.mainloop()
