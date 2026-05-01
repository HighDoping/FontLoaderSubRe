# FontLoaderSubRe — Go/Fyne Edition

A complete reimplementation of [FontLoaderSubRe](../README.md) in Go with a [Fyne](https://fyne.io/) GUI.  
All original Python files are untouched; this implementation lives entirely inside `fontloader-go/`.

---

## Requirements

| Tool | Version | Notes |
|------|---------|-------|
| Go | ≥ 1.21 | <https://go.dev/dl/> |
| Xcode Command Line Tools | any | macOS only — `xcode-select --install` |
| [MKVToolNix](https://mkvtoolnix.download/downloads.html) | any | Optional — only needed for MKV extraction |

> **No CGo, no Python, no system SQLite.** The SQLite driver (`modernc.org/sqlite`) is pure Go.

---

## Build

```sh
cd fontloader-go
go mod tidy          # first time only, downloads all dependencies
```

### macOS / Linux

```sh
make build-mac       # produces ./FontLoaderSubRe and ./cli_indexer
```

Or build individually:

```sh
go build -o FontLoaderSubRe ./cmd/fontloader/
go build -o cli_indexer     ./cmd/cli_indexer/
```

### macOS .app bundle (requires `fyne` CLI)

```sh
go install fyne.io/fyne/v2/cmd/fyne@latest
make bundle-mac      # produces FontLoaderSubRe.app
```

### Windows (cross-compile from macOS/Linux)

```sh
make build-win       # produces FontLoaderSubRe.exe and cli_indexer.exe
```

> CGo is disabled for the Windows target so no MinGW is required.

---

## GUI Usage

### First run

1. Launch `FontLoaderSubRe`.
2. Click **Menu → Set Font Directory** and select the folder that contains your font files.
3. Click **Menu → Update Index** to scan and index all fonts.  
   A progress bar shows indexing progress. This is only needed when your font collection changes.

### Loading fonts for subtitles

Drag and drop any combination of:

| File type | What happens |
|-----------|-------------|
| `*.ass` / `*.ssa` | Font names are parsed from the subtitle styles |
| `*.ttf` / `*.otf` / `*.ttc` | Font files are loaded directly |
| `*.mkv` | Subtitles and fonts are extracted first (requires MKVToolNix) |
| Folder | Recursively scanned for all of the above |

The status line updates with counts of loaded / failed / unmatched fonts.

### Details panel

Click **Details** to expand a log showing the result for every font:

```
[ok] Arial > fonts/Arial.ttf           ← found and loaded
[xx] Arial > fonts/Arial.ttf           ← found but load failed
[??] SomeMissingFont                   ← not found in the index
```

### Menu options

| Item | Description |
|------|-------------|
| Update Index | Rescan the font base directory |
| Set Font Directory | Change the font base path |
| Export Loaded Fonts | Copy all loaded fonts to a chosen folder |
| MKV Extraction | Toggle MKV subtitle/font extraction on or off |
| Language | Switch UI language: English / 正體中文 / 简体中文 |
| Clear Settings | Reset all settings to defaults |

### CLI arguments

Files can also be passed directly on the command line — useful for shell scripting or file-manager associations:

```sh
./FontLoaderSubRe /path/to/episode.ass /path/to/episode2.ass
```

---

## CLI Indexer

`cli_indexer` scans a font directory and builds (or updates) the SQLite database without opening a GUI.

```
Usage: cli_indexer <font_dir> [--db <path>] [-v]

Arguments:
  font_dir          Directory containing font files (required)

Options:
  --db <path>       Path to the SQLite database
                    (default: <font_dir>/FontLoaderSubRe.db)
  -v                Verbose / debug output
```

### Example

```sh
# Index ~/Fonts, store DB in the default location
./cli_indexer ~/Fonts

# Index a network share, write DB to a custom path
./cli_indexer /mnt/nas/fonts --db ~/Library/Application\ Support/FontLoaderSubRe/FontLoaderSubRe.db
```

Sample output:

```
Starting index update...
Target Directory: /Users/you/Fonts
Database Path:    /Users/you/Fonts/FontLoaderSubRe.db
------------------------------
Indexing: |████████████████████--------------------| 51.2% (1024/2000)
...
------------------------------
Scan Complete!
Time Elapsed:    3.42 seconds
Files Processed: 2000
Unique Names Found:
  - Full Names:   4812
  - Family Names: 1203
  - PS Names:     2401
  - Unique IDs:   2398
```

---

## Configuration

Settings are stored as JSON in the platform config directory:

| OS | Path |
|----|------|
| macOS | `~/Library/Application Support/FontLoaderSubRe/config.json` |
| Linux | `~/.config/FontLoaderSubRe/config.json` |
| Windows | `%AppData%\FontLoaderSubRe\config.json` |

The SQLite database is stored in the same directory as `FontLoaderSubRe.db`.

---

## Project Layout

```
fontloader-go/
├── cmd/
│   ├── fontloader/       # GUI entry point
│   └── cli_indexer/      # CLI indexer
├── internal/
│   ├── config/           # Settings persistence
│   ├── db/               # SQLite schema + font search
│   ├── fontmeta/         # Binary TTF/OTF/TTC name-table parser
│   ├── i18n/             # Localised strings (EN / zh-TW / zh-CN)
│   ├── loader/           # OS-specific font installation
│   ├── mkv/              # MKVToolNix CLI wrapper
│   ├── scanner/          # Parallel font directory scanner
│   ├── session/          # Per-run font-load tracking + export
│   └── subtitle/         # ASS/SSA parser (multi-encoding)
├── ui/
│   ├── app.go            # Main Fyne window & layout
│   └── workers.go        # Background goroutines
├── go.mod
├── Makefile
└── README_go.md
```

---

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make build-mac` | Build GUI + CLI for macOS/Linux |
| `make build-cli` | Build CLI indexer only |
| `make build-win` | Cross-compile both binaries for Windows |
| `make bundle-mac` | Package `FontLoaderSubRe.app` with `fyne` tool |
| `make test` | Run all tests |
| `make vet` | Run `go vet` |
| `make tidy` | Run `go mod tidy` |
| `make clean` | Remove build artefacts |
