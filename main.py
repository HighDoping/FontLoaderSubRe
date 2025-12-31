import builtins
import contextlib
import ctypes
import hashlib
import json
import logging
import platform
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import webbrowser
from multiprocessing import Pool, cpu_count, freeze_support
from pathlib import Path
from typing import Any, ClassVar, Optional

import imohash
from fontTools.ttLib import TTCollection, TTFont, TTLibFileIsCollectionError
from platformdirs import user_config_dir
from PySide6.QtCore import Qt, QThread, QTimer, Signal,QObject
from PySide6.QtGui import QCursor, QIcon

# PySide6 Imports
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

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
                font = TTFont(pt, lazy=True)
                data = self._extract_font_details(font)
                font.close()
                return data

            try:
                # Attempt to open as single font first (faster common case)
                fonts_data.append(process_ttfont(path_obj))
            except TTLibFileIsCollectionError:
                # Fallback to collection
                ttc = TTCollection(path_obj, lazy=True)
                for i, font in enumerate(ttc):
                    fonts_data.append(self._extract_font_details(font, i))
            except Exception:
                # If TTFont fails, it might be a true collection or invalid
                # Try collection explicitly one last time
                try:
                    ttc = TTCollection(path_obj, lazy=True)
                    for i, font in enumerate(ttc):
                        fonts_data.append(self._extract_font_details(font, i))
                except Exception as e:
                    return {
                        "error": f"Font parsing error: {e}",
                        "filepath": str(path_obj),
                    }

            file_hash = imohash.hashfile(path_obj,hexdigest=True)

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
        self.db_name = Path(db_name)
        logger.debug(f"Init db from {str(self.db_name)}")
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_name)
        # PERFORMANCE PRAGMAS
        conn.execute("PRAGMA journal_mode = DELETE")
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

    # def _calculate_file_hash(self, path: Path) -> str:
    #     """Calculates the MD5 hash of a file by reading the whole file at once."""
    #     try:
    #         return imohash.hashfile(path,hexdigest=True)
    #     except OSError as e:
    #         logger.error("Failed to read file for hashing '%s': %s", path, e)
    #         return ""

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

        # # 4. Integrity Check
        # logger.debug("Verifying integrity of %s...", font_path.name)
        # actual_hash = self._calculate_file_hash(font_path)

        # if not actual_hash:
        #     self.status[expected_hash]["message"] = "Read error during hashing"
        #     return False

        # if actual_hash != expected_hash:
        #     msg = f"Hash Mismatch. Expected: {expected_hash}, Found: {actual_hash}"
        #     logger.error(msg)
        #     self.status[expected_hash]["message"] = msg
        #     return False

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
            return True
        except Exception as e:
            logger.error("Error removing font %s: %s", path, e)
            return False

    def cache_refresh(self):
        if self.current_os == "Linux":
            try:
                subprocess.run(["fc-cache", "-f"], check=True, capture_output=True)
                logger.info("Linux font cache refreshed.")
            except Exception as e:
                logger.error("Error refreshing Linux font cache: %s", e)

    def cleanup(self):
        """Iterates through status and unloads all loaded fonts."""
        logger.debug("Starting cleanup...")
        hashes = list(self.status.keys())
        for h in hashes:
            if self.status[h]["loaded"]:
                self.unload_font(h)
        self.cache_refresh()
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


def scan_fonts_in_directory(db: FontDatabase, dir_path: Path, progress_callback=None):
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

    total_files = len(font_files)
    if progress_callback:
        progress_callback(0, total_files)

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
                final_stats["files_processed"] += 1
                if progress_callback:
                    progress_callback(final_stats["files_processed"], total_files)
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

            # Report progress
            if progress_callback:
                progress_callback(final_stats["files_processed"], total_files)

            # Flush Batch if full
            if len(batch_files) >= BATCH_SIZE:
                db.bulk_insert_batch(batch_files, batch_names)
                # Reset buffers
                batch_files = []
                batch_names = {"full": [], "family": [], "ps": [], "unique": []}
                logger.debug(
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


def extract_mkv_resources(mkv_path, extract_subs=True, extract_fonts=True):
    """Extracts ASS/SSA subtitles and fonts from an MKV into a temp directory."""
    # 1. Dependency Check
    tools = ["mkvmerge", "mkvextract"]
    for tool in tools:
        if not shutil.which(tool):
            logger.error(f"Missing dependency: '{tool}' not found in system PATH.")
            return None

    mkv_path = Path(mkv_path)
    if not mkv_path.exists():
        logger.error(f"Input file not found: {mkv_path}")
        return None

    # 2. Get Metadata via mkvmerge
    logger.info(f"Reading metadata from: {mkv_path.name}")
    try:
        info_cmd = ["mkvmerge", "-J", str(mkv_path)]
        result = subprocess.run(
            info_cmd,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
        )
        data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        logger.exception("Failed to parse MKV metadata.")
        return None

    temp_dir = Path(tempfile.mkdtemp(prefix="mkv_extract_"))
    logger.debug(f"Created temporary directory: {temp_dir}")

    extracted_files = []

    # 3. Identify Subtitle Tracks
    if extract_subs:
        track_args = []
        for track in data.get("tracks", []):
            codec = track.get("properties", {}).get("codec_id", "").upper()
            if track.get("type") == "subtitles" and (
                "S_TEXT/ASS" in codec or "S_TEXT/SSA" in codec
            ):
                t_id = track.get("id")
                ext = ".ass" if "ASS" in codec else ".ssa"
                name = track.get("properties", {}).get("track_name") or f"track_{t_id}"
                # Sanitize filename
                safe_name = "".join(
                    x for x in name if x.isalnum() or x in "._- "
                ).strip()
                out_path = temp_dir / f"{safe_name}{ext}"

                track_args.append(f"{t_id}:{out_path}")
                extracted_files.append(out_path)

        if track_args:
            logger.info(f"Found {len(track_args)} subtitle track(s). Extracting...")
            cmd = ["mkvextract", "tracks", str(mkv_path), *track_args]
            logger.debug(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, encoding="utf-8")
        else:
            logger.info("No ASS/SSA subtitle tracks found.")

    # 4. Identify Font Attachments
    if extract_fonts:
        attachment_args = []
        font_mimetypes = [
            "application/x-truetype-font",
            "application/vnd.ms-opentype",
            "application/x-font-ttf",
            "application/x-font-otf",
        ]

        for attach in data.get("attachments", []):
            mime = attach.get("content_type", "").lower()
            if mime in font_mimetypes or mime.startswith("font/"):
                a_id = attach.get("id")
                filename = attach.get("file_name")
                safe_filename = "".join(
                    x for x in filename if x.isalnum() or x in "._- "
                ).strip()
                out_path = temp_dir / safe_filename

                attachment_args.append(f"{a_id}:{out_path}")
                extracted_files.append(out_path)

        if attachment_args:
            logger.info(
                f"Found {len(attachment_args)} font attachment(s). Extracting..."
            )
            cmd = ["mkvextract", "attachments", str(mkv_path), *attachment_args]
            logger.debug(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, encoding="utf-8")
        else:
            logger.info("No font attachments found.")

    if not extracted_files:
        logger.warning("No resources were extracted.")
        with contextlib.suppress(OSError):
            temp_dir.rmdir()
        return None

    logger.info(f"Successfully extracted {len(extracted_files)} files to {temp_dir}")
    return temp_dir, extracted_files


class ExtractWorker(QThread):
    """Background thread for extracting font names from subtitle files."""

    progress = Signal(int, int)  # current, total
    finished = Signal(int, list, list)  # total_subs, all_fonts

    def __init__(self, file_list: list[Path]):
        super().__init__()
        self.file_list = file_list

    def run(self):
        all_fonts = set()
        all_font_paths = set()
        total_subs = 0
        total_files = len(self.file_list)

        for i, file_path in enumerate(self.file_list):
            ext = file_path.suffix.lower()

            if ext in {".ass", ".ssa"}:
                sub_count, font_list = extract_ass_fonts(file_path)
                all_fonts.update(font_list)
                total_subs += sub_count
            elif ext in {".ttf", ".otf", ".ttc"}:
                all_font_paths.add(file_path.absolute())
            elif ext == ".mkv":
                result = extract_mkv_resources(
                    file_path, extract_fonts=True, extract_subs=True
                )
                if not result:
                    continue
                temp_folder, files = result
                for file in files:
                    if not isinstance(file, Path):
                        continue
                    file_ext = file.suffix.lower()
                    if file_ext in {".ass", ".ssa"}:
                        sub_count, font_list = extract_ass_fonts(file)
                        all_fonts.update(font_list)
                        total_subs += sub_count
                    elif file_ext in {".ttf", ".otf", ".ttc"}:
                        all_font_paths.add((temp_folder / file).absolute())

            self.progress.emit(i + 1, total_files)

        self.finished.emit(total_subs, list(all_fonts), list(all_font_paths))


class ScanWorker(QThread):
    """Background thread for scanning fonts."""

    progress = Signal(int, int)  # current, total
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, db, font_path):
        super().__init__()
        self.db = db
        self.font_path = font_path

    def run(self):
        try:
            stats = scan_fonts_in_directory(
                self.db,
                Path(self.font_path),
                progress_callback=lambda current, total: self.progress.emit(
                    current, total
                ),
            )
            self.finished.emit(stats)
        except Exception as e:
            self.error.emit(str(e))


class LoadWorker(QThread):
    """Background thread for loading fonts into the session."""

    progress = Signal(int, str, str)  # index, font_name, log_line
    finished = Signal(list)

    def __init__(
        self,
        font_list: list[str],
        font_path_list: list[Path],
        font_manager: SessionFontManager,
        db: FontDatabase,
        font_base_path: Path,
    ):
        super().__init__()
        self.font_list = font_list
        self.font_path_list = font_path_list
        self.fm = font_manager
        self.db = db
        self.font_base = font_base_path

    def run(self):
        log_lines = []
        # load from paths first
        loaded_font_from_path = set()
        for i, font_path in enumerate(self.font_path_list):
            font_path = font_path.absolute()
            f_fullname = (
                FontMetadataExtractor()
                .db_decode_file(font_path)
                .get("full_names", [])[0]
            )
            f_hash = FontMetadataExtractor().db_decode_file(font_path).get("hash", "")
            success = self.fm.load_font(font_path, f_hash)
            if success:
                log_line = f"[ok] {f_fullname} > {str(font_path)}\n"
                loaded_font_from_path.add(f_fullname)
            else:
                msg = self.fm.status.get(f_hash, {}).get("message", "Error")
                log_line = f"[xx] {f_fullname} > {str(font_path)} ({msg})\n"
            log_lines.append(log_line)
            self.progress.emit(i + 1 + len(self.font_list), font_path.name, log_line)
        # then load from db
        for i, font in enumerate(self.font_list):
            # first check if already loaded
            log_line = ""
            if font in loaded_font_from_path:
                # log_line = f"[ok] {font} > Loaded from path\n"
                # log_lines.append(log_line)
                self.progress.emit(i + 1, font, log_line)
                continue
            res = self.db.search_by_font(font)
            log_line = ""
            if res:
                relative_path_str = res[0][0]
                f_path = (Path(self.font_base) / Path(relative_path_str)).absolute()
                f_hash = res[0][1]
                success = self.fm.load_font(f_path, f_hash)
                if success:
                    log_line = f"[ok] {font} > {relative_path_str}\n"
                else:
                    msg = self.fm.status.get(f_hash, {}).get("message", "Error")
                    log_line = f"[xx] {font} > {relative_path_str} ({msg})\n"
            else:
                log_line = f"[??] {font}\n"

            log_lines.append(log_line)
            self.progress.emit(i + 1, font, log_line)

        self.fm.cache_refresh()
        self.finished.emit(log_lines)


class SettingsDialog(QDialog):
    """Popup for changing font base path."""

    def __init__(self, parent, current_path, locale):
        super().__init__(parent)
        self.locales = locale
        self.setWindowTitle(locale["title_font_base"])
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(locale["lbl_current_path"]))
        self.entry = QLineEdit(str(current_path))
        self.entry.setReadOnly(True)
        layout.addWidget(self.entry)

        btn_layout = QHBoxLayout()
        self.btn_change = QPushButton(locale["btn_change"])
        self.btn_close = QPushButton(locale["btn_close"])
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_change)
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)

        self.btn_close.clicked.connect(self.accept)
        self.btn_change.clicked.connect(self.select_dir)
        self.result_path = current_path

    def select_dir(self):
        new_dir = QFileDialog.getExistingDirectory(
            self, self.locales["select_font_base"], str(self.result_path)
        )
        if new_dir:
            self.result_path = new_dir
            self.entry.setText(new_dir)
class QLogSignal(QObject):
    log = Signal(str)


class QLogHandler(logging.Handler):
    def __init__(self, emitter):
        super().__init__()
        self._emitter = emitter

    @property
    def emitter(self):
        return self._emitter

    def emit(self, record):
        msg = self.format(record)
        self.emitter.log.emit(msg)

class FontLoaderApp(QMainWindow):

    def __init__(self, sub_paths: Optional[list[Path]] = None):
        super().__init__()
        self.setAcceptDrops(True)

        q_log_signal = QLogSignal()
        h = QLogHandler(q_log_signal)
        h.setLevel(logging.ERROR)
        logging.getLogger().addHandler(h)
        q_log_signal.log.connect(self.error_pop_up)

        # Configuration & Managers
        self.app_config = AppConfig()
        try:
            self.db = FontDatabase(self.app_config.get_db_path())
        except Exception as e:
            logger.error("Database error: %s", e)
            QMessageBox.critical(
                self,
                "Database Error",
                f"Failed to open database:\n{e}\n Default to current folder.",
            )
            self.app_config.set_font_base_path(Path.cwd())
            self.db = FontDatabase(self.app_config.get_db_path())

        self.font_manager = SessionFontManager()
        self.project_link = "https://github.com/HighDoping/FontLoaderSubRe"

        # Localization
        self.locales = {
            "en_us": {
                "header": "Finished",
                "status_1": "{loaded} fonts loaded, {errors} errors, {no_match} unmatched.",
                "status_2": "Index: {index_fonts} fonts, {index_names} names; {subtitles} subtitles, {imported_fonts} imported fonts.",
                "btn_menu": "Menu",
                "btn_details": "Details",
                "btn_close": "Close",
                "btn_change": "Change...",
                "menu_update": "Update Index",
                "menu_font_base": "Set Font Base",
                "select_font_base": "Select Font Base Directory",
                "menu_export": "Export Fonts",
                "menu_mkv": "Extract from MKV",
                "menu_help": "Help",
                "menu_lang": "Language",
                "menu_clear": "Clear Settings",
                "title_font_base": "Font Base Settings",
                "lbl_current_path": "Current Base Path:",
                "msg_export": "Export complete.",
                "msg_help": "1. Open FontLoaderSubRe, set font directory path, index the fonts from menu. (Only needed first time)\n2. Drag-and-drop subtitles SSA/ASS files or folders onto the FontLoaderSubRe window.",
                "msg_update_complete": "Index update complete.",
                "msg_extracting_subs": "Extracting font names from subtitles...",
                "msg_clear": "Settings cleared",
                "footer": f"GPLv2: {self.project_link}",
                "error": "Error",
            },
            "zh_tw": {
                "header": "完成",
                "status_1": "{loaded} 個字型載入成功，{errors} 個出錯，{no_match} 個無匹配。",
                "status_2": "索引中有 {index_fonts} 個字型，{index_names} 種名稱；當前共 {subtitles} 個字幕，{imported_fonts}個匯入字型。",
                "btn_menu": "選單",
                "btn_details": "詳情",
                "btn_close": "關閉",
                "btn_change": "變更...",
                "menu_update": "更新索引",
                "menu_font_base": "設定字型庫",
                "select_font_base": "設定字型庫資料夾",
                "menu_export": "匯出字型",
                "menu_mkv": "從 MKV 提取",
                "menu_help": "說明",
                "menu_clear": "清除設定",
                "menu_lang": "語言",
                "title_font_base": "字型庫路徑設定",
                "lbl_current_path": "目前字型庫路徑:",
                "msg_export": "匯出完成。",
                "msg_help": "1. 開啟 FontLoaderSubRe，設定字型库路徑，並從選單索引字型。（僅需首次執行）\n2. 將字幕 SSA/ASS 檔案或資料夾拖放至 FontLoaderSubRe 視窗。",
                "msg_update_complete": "索引更新完成。",
                "msg_extracting_subs": "從字幕中獲取字型名稱...",
                "msg_clear": "設定已清除",
                "footer": f"GPLv2: {self.project_link}",
                "error": "錯誤",
            },
            "zh_cn": {
                "header": "完成",
                "status_1": "{loaded} 个字体加载成功，{errors} 个出错，{no_match} 个无匹配。",
                "status_2": "索引中有 {index_fonts} 个字体，{index_names} 种名称；当前共 {subtitles} 个字幕，{imported_fonts}个导入字体。",
                "btn_menu": "菜单",
                "btn_details": "详情",
                "btn_close": "关闭",
                "btn_change": "更改...",
                "menu_update": "更新索引",
                "menu_font_base": "设置字体库",
                "select_font_base": "设置字体库文件夹",
                "menu_export": "导出字体",
                "menu_mkv": "从MKV提取",
                "menu_help": "帮助",
                "menu_clear": "清除设置",
                "menu_lang": "语言",
                "title_font_base": "字体库路径设置",
                "lbl_current_path": "当前字体库路径:",
                "msg_export": "导出完成。",
                "msg_help": "1. 打开FontLoaderSubRe，设置字体库路径，从菜单中索引字体。（仅首次需要）\n2. 将字幕SSA/ASS文件或文件夹拖拽至 FontLoaderSubRe 窗口。",
                "msg_update_complete": "索引更新完成。",
                "msg_extracting_subs": "从字幕中获取字体名称...",
                "msg_clear": "设置已清除",
                "footer": f"GPLv2: {self.project_link}",
                "error": "错误",
            },
        }

        self.stats = {
            "loaded": 0,
            "errors": 0,
            "no_match": 0,
            "index_fonts": 0,
            "index_names": 0,
            "subtitles": 0,
            "imported_fonts": 0,
        }
        self.init_ui()
        self.refresh_ui()

        if sub_paths:
            QTimer.singleShot(100, lambda: self.process_dropped_files(sub_paths))

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QVBoxLayout(central)

        # Labels
        self.lbl_header = QLabel()
        self.lbl_header.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #0033CC;"
        )
        self.main_layout.addWidget(self.lbl_header)

        self.lbl_status1 = QLabel()
        self.lbl_status2 = QLabel()
        self.main_layout.addWidget(self.lbl_status1)
        self.main_layout.addWidget(self.lbl_status2)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        self.lbl_progress_task = QLabel()
        self.lbl_progress_task.hide()
        self.main_layout.addWidget(self.progress_bar)
        self.main_layout.addWidget(self.lbl_progress_task)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_menu = QPushButton()
        self.btn_details = QPushButton()
        self.btn_close = QPushButton()

        self.btn_menu.clicked.connect(self.show_popup_menu)
        self.btn_details.clicked.connect(self.toggle_details)
        self.btn_close.clicked.connect(self.close)

        btn_layout.addWidget(self.btn_menu)
        btn_layout.addWidget(self.btn_details)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_close)
        self.main_layout.addLayout(btn_layout)

        # Footer
        self.lbl_footer = QLabel()
        self.lbl_footer.setStyleSheet("color: blue; text-decoration: underline;")
        self.lbl_footer.setCursor(Qt.PointingHandCursor)
        self.lbl_footer.mousePressEvent = lambda e: webbrowser.open(self.project_link)
        self.main_layout.addWidget(self.lbl_footer)

        # Details Area
        self.txt_details = QPlainTextEdit()
        self.txt_details.setReadOnly(True)
        self.txt_details.hide()
        self.main_layout.addWidget(self.txt_details)

    def refresh_ui(self):
        txt = self.locales.get(self.app_config.get_language())

        self.lbl_header.setText(txt["header"])
        self.update_stats_from_db()

        self.lbl_status1.setText(txt["status_1"].format(**self.stats))
        self.lbl_status2.setText(txt["status_2"].format(**self.stats))
        self.btn_menu.setText(txt["btn_menu"])
        self.btn_details.setText(txt["btn_details"])
        self.btn_close.setText(txt["btn_close"])
        self.lbl_footer.setText(txt["footer"])

    def update_stats_from_db(self):
        self.stats["index_fonts"] = int(self.db.metadata_get("file_count") or 0)
        self.stats["index_names"] = sum(
            int(self.db.metadata_get(k) or 0)
            for k in [
                "ps_name_count",
                "family_name_count",
                "full_name_count",
                "unique_id_count",
            ]
        )

    def toggle_details(self):
        if self.txt_details.isVisible():
            self.hide_txt_details()
        else:
            self.txt_details.show()

    def show_popup_menu(self):
        menu = QMenu(self)
        txt = self.locales.get(self.app_config.get_language())

        act_update = menu.addAction(txt["menu_update"])
        act_update.triggered.connect(self.action_update_index)

        act_base = menu.addAction(txt["menu_font_base"])
        act_base.triggered.connect(self.action_set_font_base)

        act_export = menu.addAction(txt["menu_export"])
        act_export.triggered.connect(self.action_export)

        act_mkv_checkbox = menu.addAction(txt["menu_mkv"])
        act_mkv_checkbox.setCheckable(True)
        act_mkv_checkbox.setChecked(self.app_config.get("mkv_extraction", False))
        act_mkv_checkbox.triggered.connect(
            lambda: self.app_config.set("mkv_extraction", act_mkv_checkbox.isChecked())
        )

        menu.addSeparator()
        lang_menu = menu.addMenu(txt["menu_lang"])
        lang_menu.addAction("English").triggered.connect(lambda: self.set_lang("en_us"))
        lang_menu.addAction("正體中文").triggered.connect(
            lambda: self.set_lang("zh_tw")
        )
        lang_menu.addAction("简体中文").triggered.connect(
            lambda: self.set_lang("zh_cn")
        )

        menu.addSeparator()

        menu.addAction(txt["menu_help"]).triggered.connect(self.action_help)
        menu.addAction(txt["menu_clear"]).triggered.connect(self.action_clear)

        menu.exec(QCursor.pos())

    def set_lang(self, code):
        self.app_config.set_language(code)
        self.refresh_ui()

    def action_update_index(self):
        self.btn_menu.setEnabled(False)

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.lbl_progress_task.show()

        self.scan_thread = ScanWorker(self.db, self.app_config.get_font_base_path())

        self.scan_thread.progress.connect(self._on_scan_progress)

        self.scan_thread.finished.connect(self._on_scan_finished)
        txt = self.locales.get(self.app_config.get_language())
        self.scan_thread.error.connect(
            lambda e: QMessageBox.critical(self, txt["error"], e)
        )
        self.scan_thread.start()

    def _on_scan_progress(self, current, total):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        txt = self.locales.get(self.app_config.get_language())
        task_text = f"{txt['menu_update']}... ({current} / {total})"
        self.lbl_progress_task.setText(task_text)

    def _on_scan_finished(self, stats):
        self.hide_progress()
        self.btn_menu.setEnabled(True)
        self.refresh_ui()

        txt = self.locales.get(self.app_config.get_language())
        QMessageBox.information(self, txt["menu_update"], txt["msg_update_complete"])

    def action_set_font_base(self):
        txt = self.locales.get(self.app_config.get_language())
        dialog = SettingsDialog(self, self.app_config.get_font_base_path(), txt)
        if dialog.exec():
            new_path = Path(dialog.result_path)
            self.app_config.set_font_base_path(new_path)
            try:
                self.db = FontDatabase(self.app_config.get_db_path())
            except Exception as e:
                logger.error("Database error: %s", e)
                QMessageBox.critical(
                    self,
                    "Database Error",
                    f"Failed to open database:\n{e}\n Default to current folder.",
                )
                self.app_config.set_font_base_path(Path.cwd())
                self.db = FontDatabase(self.app_config.get_db_path())
            self.refresh_ui()

    def action_help(self):
        txt = self.locales.get(self.app_config.get_language())
        QMessageBox.information(self, txt["menu_help"], txt["msg_help"])

    def action_clear(self):
        self.app_config.clear()
        txt = self.locales.get(self.app_config.get_language())
        QMessageBox.information(self, txt["menu_clear"], txt["msg_clear"])
        self.refresh_ui()

    def action_export(self):
        target = QFileDialog.getExistingDirectory(self, "Export Fonts")
        if not target:
            return
        target_path = Path(target)
        for h, info in self.font_manager.status.items():
            if info["loaded"] and info["sys_path"]:
                with contextlib.suppress(builtins.BaseException):
                    shutil.copy(info["sys_path"], target_path / info["sys_path"].name)
        txt = self.locales.get(self.app_config.get_language())
        QMessageBox.information(self, txt["menu_export"], txt["msg_export"])

    # --- Drag & Drop ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [Path(u.toLocalFile()) for u in event.mimeData().urls()]
        if not files:
            return

        # Determine valid extensions based on mkv_extraction setting
        valid_exts = {".ass", ".ssa", ".ttf", ".otf", ".ttc"}
        if self.app_config.get("mkv_extraction", False):
            valid_exts.add(".mkv")

        # Collect all valid files from dropped items
        all_files = []
        for f in files:
            if f.is_dir():
                for ext in valid_exts:
                    all_files.extend(f.rglob(f"*{ext}"))
                    all_files.extend(f.rglob(f"*{ext.upper()}"))
            elif f.is_file() and f.suffix.lower() in valid_exts:
                all_files.append(f)

        if all_files:
            # Process the collected files
            logger.debug(f"Dropped files to process: {all_files}")
            self.process_dropped_files(all_files)

    def error_pop_up(self, msg):
        txt = self.locales.get(self.app_config.get_language())
        QMessageBox.critical(self, txt["error"], msg)

    def hide_progress(self):
        self.progress_bar.hide()
        self.lbl_progress_task.hide()
        if self.centralWidget().layout():
            self.centralWidget().layout().activate()
        self.adjustSize()

    def hide_txt_details(self):
        self.txt_details.hide()
        if self.centralWidget().layout():
            self.centralWidget().layout().activate()
        self.adjustSize()

    def process_dropped_files(self, file_list: list[Path]):
        """Process multiple subtitle files using a background worker."""
        if not file_list:
            return

        self.progress_bar.setRange(0, len(file_list))
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.lbl_progress_task.show()
        txt = self.locales.get(self.app_config.get_language())
        self.lbl_progress_task.setText(txt["msg_extracting_subs"])

        self.extract_thread = ExtractWorker(file_list)
        self.extract_thread.progress.connect(
            lambda current, total: self.progress_bar.setValue(current)
        )
        self.extract_thread.finished.connect(self._on_extract_finished)
        self.extract_thread.start()

    def _on_extract_finished(
        self, total_subs: int, font_list: list[str], font_file_list: list[Path]
    ):
        if not font_list and not font_file_list:
            self.hide_progress()
            return

        logger.debug("Extracted fonts: %s", font_list)
        logger.debug("Extracted font files: %s", font_file_list)

        self.stats["subtitles"] += total_subs
        self.stats["imported_fonts"] += len(font_file_list)

        # Transition to Loading phase
        self.progress_bar.setRange(0, len(font_list) + len(font_file_list))
        self.progress_bar.setValue(0)

        self.load_thread = LoadWorker(
            font_list,
            font_file_list,
            self.font_manager,
            self.db,
            self.app_config.get_font_base_path(),
        )
        self.load_thread.progress.connect(self._on_load_progress)
        self.load_thread.finished.connect(self._on_load_finished)
        self.load_thread.start()

    def _on_load_progress(self, idx, font_name, log_line):
        self.progress_bar.setValue(idx)
        self.lbl_progress_task.setText(f"Loading: {font_name}")

        if "[ok]" in log_line:
            self.stats["loaded"] += 1
        elif "[xx]" in log_line:
            self.stats["errors"] += 1
        elif "[??]" in log_line:
            self.stats["no_match"] += 1
        else:
            pass

    def _on_load_finished(self, log_lines):
        self.hide_progress()

        previously_loaded = self.txt_details.toPlainText().splitlines()
        all_lines = previously_loaded + [line.strip() for line in log_lines]

        # Remove duplicates while preserving order
        seen = set()
        unique_lines = []
        for line in all_lines:
            if line and line not in seen:
                seen.add(line)
                unique_lines.append(line)

        # Reorder log lines by status: [ok], [xx], [??]
        ok_lines = [line for line in unique_lines if line.startswith("[ok]")]
        error_lines = [line for line in unique_lines if line.startswith("[xx]")]
        no_match_lines = [line for line in unique_lines if line.startswith("[??]")]

        # Recalculate stats based on unique lines
        self.stats["loaded"] = len(ok_lines)
        self.stats["errors"] = len(error_lines)
        self.stats["no_match"] = len(no_match_lines)

        lines = ""
        for line in ok_lines + error_lines + no_match_lines:
            lines += line.strip() + "\n"
        self.txt_details.setPlainText(lines)

        self.refresh_ui()

    def closeEvent(self, event):
        self.font_manager.cleanup()
        event.accept()


class AppConfig:
    """Unified interface for application configuration management."""

    APP_NAME = "FontLoaderSubRe"
    APP_AUTHOR = "HighDoping"

    DB_NAME = "FontLoaderSubRe.db"

    DEFAULT_SETTINGS = {
        "font_base_path": str(Path.cwd()),
        "ui_language": "en_us",
        "mkv_extraction": False,
    }

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

        return self.DEFAULT_SETTINGS.copy()

    def _save_settings(self):
        """Persist settings to disk."""
        try:
            with self.config_file.open("w", encoding="utf-8") as f:
                json.dump(self._settings, f, indent=4)
        except Exception as e:
            logger.exception(f"Failed to save settings: {e}")

    def get(self, key: str, default=None):
        """Get a configuration value."""
        return self._settings.get(key, default)

    def set(self, key: str, value):
        """Set a configuration value and save."""
        self._settings[key] = value
        self._save_settings()

    def get_font_base_path(self) -> Path:
        """Get the font base directory."""
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

    def clear(self):
        """Clear all settings"""
        self._settings = self.DEFAULT_SETTINGS.copy()
        self._save_settings()


if __name__ == "__main__":
    freeze_support()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    app = QApplication(sys.argv)

    app.setApplicationName("FontLoaderSubRe")
    app.setDesktopFileName("FontLoaderSubRe")

    sub_paths = [Path(arg) for arg in sys.argv[1:]] if len(sys.argv) > 1 else []
    window = FontLoaderApp(sub_paths=sub_paths)
    window.setWindowTitle("FontLoaderSubRe 0.2.0")

    # add icon
    if platform.system() == "Darwin":
        icns_path = Path(__file__).parent / "resources" / "icon.icns"
        if icns_path.exists():
            icon = QIcon(str(icns_path))
            window.setWindowIcon(icon)
            # Set dock icon on macOS
            app.setWindowIcon(icon)
    else:
        icon_path = Path(__file__).parent / "resources" / "icon.png"
        if icon_path.exists():
            icon = QIcon(str(icon_path))
            window.setWindowIcon(icon)
            app.setWindowIcon(icon)

    window.show()

    sys.exit(app.exec())
