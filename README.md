# FontLoaderSubRe

Heavily inspired by [FontLoaderSub](https://github.com/yzwduck/FontLoaderSub).

Drag-and-drop ASS/SSA subtitles and it will load corresponding font files from a font base.

> The primary implementation is now written in Go (Fyne GUI).
> The old Python/PySide6 version is preserved in [`python/`](python/) for reference.

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

Files can also be passed directly on the command line:

```sh
./FontLoaderSubRe /path/to/episode.ass /path/to/episode2.ass
```

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

## Configuration

Settings are stored as JSON in the platform config directory:

| OS | Path |
|----|------|
| macOS | `~/Library/Application Support/FontLoaderSubRe/config.json` |
| Linux | `~/.config/FontLoaderSubRe/config.json` |
| Windows | `%AppData%\FontLoaderSubRe\config.json` |

The SQLite database is stored in the font base directory as `FontLoaderSubRe.db`

## Build

```sh
go mod tidy
```

### macOS / Linux

```sh
make build-unix
```

### macOS .app bundle (requires `fyne` CLI)

```sh
go install fyne.io/fyne/v2/cmd/fyne@latest
make bundle-mac      # produces FontLoaderSubRe.app
```

### Windows

```sh
make build-win       # produces FontLoaderSubRe.exe and cli_indexer.exe
```
