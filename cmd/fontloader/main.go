// FontLoaderSubRe GUI entry point.
// Usage:
//
//	fontloader [file1.ass file2.ass ...]
package main

import (
	_ "embed"
	"os"
	"path/filepath"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"

	"FontLoaderSubRe/internal/config"
	"FontLoaderSubRe/internal/db"
	"FontLoaderSubRe/internal/session"
	"FontLoaderSubRe/ui"
)

//go:embed icon.png
var iconBytes []byte

func main() {
	cfg, err := config.Load()
	if err != nil {
		// Fall back to defaults; don't abort.
		cfg, _ = config.Load()
	}

	database, err := db.Open(cfg.DBPath())
	if err != nil {
		// If the configured DB path fails, fall back to CWD.
		cfg.FontBasePath, _ = os.Getwd()
		database, err = db.Open(cfg.DBPath())
		if err != nil {
			panic("cannot open database: " + err.Error())
		}
	}
	defer database.Close()

	mgr := session.New()

	a := app.NewWithID("com.highdoping.fontloadersubr")
	a.SetIcon(fyne.NewStaticResource("icon.png", iconBytes))

	uiApp := ui.New(a, cfg, database, mgr)

	// Process files passed as CLI arguments.
	if len(os.Args) > 1 {
		var paths []string
		for _, arg := range os.Args[1:] {
			abs, err := filepath.Abs(arg)
			if err == nil {
				paths = append(paths, abs)
			}
		}
		if len(paths) > 0 {
			// Delay until the window is visible.
			go func() {
				uiApp.ProcessFiles(paths)
			}()
		}
	}

	uiApp.Show()
	a.Run()
}
