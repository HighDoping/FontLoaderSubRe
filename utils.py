import contextlib
import ctypes
import json
import logging
import platform
import shutil
import sqlite3
import subprocess
import tempfile
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, ClassVar, Optional

import imohash
from fontTools.ttLib import TTCollection, TTFont, TTLibFileIsCollectionError

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

            file_hash = imohash.hashfile(path_obj, hexdigest=True)

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

    def table_length(self, table_name: str) -> int:
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
        rel_path = path.relative_to(dir_path).as_posix()
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
    db.metadata_set("family_name_count", str(db.table_length("file_families")))
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
