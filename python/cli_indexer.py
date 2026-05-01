import argparse
import logging
import sys
import time
from multiprocessing import freeze_support
from pathlib import Path

try:
    from utils import FontDatabase, scan_fonts_in_directory
except ImportError:
    print("Error: Could not find main.py in the current directory.")
    sys.exit(1)


def print_progress(current, total):
    """Text-based progress bar for the console."""
    if total == 0:
        return
    percent = (current / total) * 100
    bar_length = 40
    filled_length = int(bar_length * current // total)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)

    # \r moves the cursor back to the start of the line
    sys.stdout.write(f"\rIndexing: |{bar}| {percent:.1f}% ({current}/{total})")
    sys.stdout.flush()
    if current == total:
        print()


def main():
    parser = argparse.ArgumentParser(
        description="FontLoaderSubRe CLI - Update Font Index Database"
    )

    # Required: Directory to scan
    parser.add_argument(
        "font_dir", type=str, help="Path to the directory containing your fonts."
    )

    # Optional: Custom DB path
    parser.add_argument(
        "--db",
        type=str,
        help="Optional: Path to the SQLite database file. (Default: [font_dir]/FontLoaderSubRe.db)",
    )

    # Optional: Verbose logging
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show debug logging"
    )

    args = parser.parse_args()

    # 1. Setup Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")

    # 2. Resolve Paths
    font_path = Path(args.font_dir).absolute()
    if not font_path.is_dir():
        print(f"Error: Path is not a directory: {font_path}")
        sys.exit(1)

    db_path = Path(args.db).absolute() if args.db else font_path / "FontLoaderSubRe.db"

    print(f"Starting index update...")
    print(f"Target Directory: {font_path}")
    print(f"Database Path:    {db_path}")
    print("-" * 30)

    # 3. Initialize Database
    try:
        db = FontDatabase(db_path)
    except Exception as e:
        print(f"Database Error: {e}")
        sys.exit(1)

    # 4. Run Scan
    start_time = time.time()
    try:
        stats = scan_fonts_in_directory(db, font_path, progress_callback=print_progress)

        elapsed = time.time() - start_time

        # 5. Report Stats
        print("-" * 30)
        print("Scan Complete!")
        print(f"Time Elapsed:    {elapsed:.2f} seconds")
        print(f"Files Processed: {stats['files_processed']}")
        print(f"Unique Names Found:")
        print(f"  - Full Names:   {stats['unique_font_names']['full']}")
        print(f"  - Family Names: {stats['unique_font_names']['family']}")
        print(f"  - PS Names:     {stats['unique_font_names']['ps']}")
        print(f"  - Unique IDs:   {stats['unique_font_names']['unique']}")

    except KeyboardInterrupt:
        print("\n\nIndexing cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nAn error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    freeze_support()
    main()
