package ui

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"fyne.io/fyne/v2/dialog"

	"fontloader-go/internal/fontmeta"
	"fontloader-go/internal/mkv"
	"fontloader-go/internal/scanner"
	"fontloader-go/internal/subtitle"
)

// runExtractWorker extracts font names/paths from the provided file list,
// then hands off to runLoadWorker.
// Runs in a background goroutine; Fyne widget methods are goroutine-safe.
func (a *App) runExtractWorker(paths []string) {
	var fontNames []string
	var fontPaths []string
	totalSubs := 0
	total := len(paths)

	for i, p := range paths {
		ext := strings.ToLower(filepath.Ext(p))
		switch ext {
		case ".ass", ".ssa":
			count, names, err := subtitle.ExtractFonts(p)
			if err == nil {
				totalSubs += count
				fontNames = append(fontNames, names...)
				fmt.Fprintf(os.Stdout, "[sub] %s  (%d font name(s) found)\n", filepath.Base(p), len(names))
				for _, n := range names {
					fmt.Fprintf(os.Stdout, "      - %s\n", n)
				}
			} else {
				fmt.Fprintf(os.Stderr, "[err] %s: %v\n", filepath.Base(p), err)
			}
		case ".ttf", ".otf", ".ttc":
			fontPaths = append(fontPaths, p)
		case ".mkv":
			result, err := mkv.Extract(p, true, true)
			if err == nil && result != nil {
				defer result.Cleanup()
				for _, f := range result.Files {
					fext := strings.ToLower(filepath.Ext(f))
					if fext == ".ass" || fext == ".ssa" {
						count, names, err := subtitle.ExtractFonts(f)
						if err == nil {
							totalSubs += count
							fontNames = append(fontNames, names...)
							fmt.Fprintf(os.Stdout, "[sub] %s  (%d font name(s) found)\n", filepath.Base(f), len(names))
							for _, n := range names {
								fmt.Fprintf(os.Stdout, "      - %s\n", n)
							}
						}
					} else if fext == ".ttf" || fext == ".otf" || fext == ".ttc" {
						fontPaths = append(fontPaths, f)
					}
				}
			}
		}
		a.progressBar.SetValue(float64(i+1) / float64(total))
	}

	dedupNames := dedup(fontNames)
	a.subtitles += totalSubs
	a.importedFonts += len(fontPaths)

	if len(dedupNames) > 0 {
		fmt.Fprintf(os.Stdout, "\n[info] %d unique font name(s) to load\n", len(dedupNames))
	}

	if len(dedupNames) > 0 {
		fmt.Fprintf(os.Stdout, "\n[info] %d unique font name(s) to load\n", len(dedupNames))
	}

	if len(dedupNames) == 0 && len(fontPaths) == 0 {
		a.progressBar.Hide()
		a.lblProgressTask.Hide()
		return
	}

	a.progressBar.SetValue(0)
	a.lblProgressTask.SetText("Loading...")

	go a.runLoadWorker(dedupNames, fontPaths)
}

// runLoadWorker resolves font names via DB, loads them, and updates the UI.
func (a *App) runLoadWorker(fontNames []string, fontPaths []string) {
	totalItems := len(fontNames) + len(fontPaths)
	done := 0
	var logLines []string

	fontBase := a.cfg.FontBasePath

	// 1. Load font files referenced by absolute path (from MKV or direct drop).
	for i, p := range fontPaths {
		// Emit "Loading" progress before doing the work.
		done = i + 1 + len(fontNames)
		a.progressBar.SetValue(float64(done) / float64(totalItems))
		a.lblProgressTask.SetText("Loading: " + filepath.Base(p))

		fd, err := fontmeta.Extract(p)
		var firstName string
		if err == nil && len(fd.FullNames) > 0 {
			firstName = fd.FullNames[0]
		} else {
			firstName = filepath.Base(p)
		}

		var line string
		if err == nil && a.mgr.Load(p, fd.Hash) {
			line = "[ok] " + firstName + " > " + p
		} else {
			line = "[xx] " + firstName + " > " + p
		}
		logLines = append(logLines, line)
		fmt.Fprintln(os.Stdout, line)
		a.lblProgressTask.SetText("Loaded: " + firstName)
		a.updateStats(line)
		a.refreshUI()
	}

	// 2. Look up font names in the database.
	for i, name := range fontNames {
		// Emit "Loading" progress before doing the work.
		done = i + 1
		a.progressBar.SetValue(float64(done) / float64(totalItems))
		a.lblProgressTask.SetText("Loading: " + name)

		results, err := a.db.SearchByFont(name)
		var line string
		if err != nil || len(results) == 0 {
			line = "[??] " + name
		} else {
			r := results[0]
			absPath := filepath.Join(fontBase, r.RelativePath)
			if a.mgr.Load(absPath, r.Hash) {
				line = "[ok] " + name + " > " + r.RelativePath
			} else {
				line = "[xx] " + name + " > " + r.RelativePath
			}
		}
		logLines = append(logLines, line)
		fmt.Fprintln(os.Stdout, line)
		a.lblProgressTask.SetText("Loaded: " + name)
		a.updateStats(line)
		a.refreshUI()
	}

	// Signal cache refresh in progress.
	a.progressBar.SetValue(1)
	a.lblProgressTask.SetText("Refreshing font cache...")
	a.mgr.CacheRefresh()

	a.progressBar.Hide()
	a.lblProgressTask.Hide()

	// Merge with existing lines, deduplicate, sort by status.
	existing := strings.Split(strings.TrimRight(a.txtDetails.Text, "\n"), "\n")
	all := append(existing, logLines...)
	unique := uniqueLines(all)

	var ok, errs, nm []string
	for _, l := range unique {
		switch {
		case strings.HasPrefix(l, "[ok]"):
			ok = append(ok, l)
		case strings.HasPrefix(l, "[xx]"):
			errs = append(errs, l)
		case strings.HasPrefix(l, "[??]"):
			nm = append(nm, l)
		}
	}
	a.loaded = len(ok)
	a.errors = len(errs)
	a.noMatch = len(nm)

	combined := strings.Join(append(append(ok, errs...), nm...), "\n")
	if len(combined) > 0 {
		combined += "\n"
	}
	a.txtDetails.SetText(combined)
	a.refreshUI()
}

// runScanWorker indexes fonts in the configured base directory.
func (a *App) runScanWorker() {
	progress := func(current, t int) {
		if t > 0 {
			a.progressBar.SetValue(float64(current) / float64(t))
		}
		s := a.strings()
		a.lblProgressTask.SetText(fmt.Sprintf("%s... (%d / %d)", s.MenuUpdate, current, t))
	}

	_, err := scanner.Scan(a.db, a.cfg.FontBasePath, progress)

	a.progressBar.Hide()
	a.lblProgressTask.Hide()
	a.btnMenu.Enable()
	a.refreshUI()

	s := a.strings()
	if err != nil {
		dialog.ShowError(err, a.win)
	} else {
		dialog.ShowInformation(s.MenuUpdate, s.MsgUpdateComplete, a.win)
	}
}

// --- helpers ---

// updateStats increments the appropriate counter for a single log line.
func (a *App) updateStats(line string) {
	switch {
	case strings.Contains(line, "[ok]"):
		a.loaded++
	case strings.Contains(line, "[xx]"):
		a.errors++
	case strings.Contains(line, "[??]"):
		a.noMatch++
	}
}

func dedup(ss []string) []string {
	seen := make(map[string]bool, len(ss))
	var out []string
	for _, s := range ss {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}

func uniqueLines(lines []string) []string {
	seen := make(map[string]bool, len(lines))
	var out []string
	for _, l := range lines {
		l = strings.TrimSpace(l)
		if l != "" && !seen[l] {
			seen[l] = true
			out = append(out, l)
		}
	}
	return out
}
