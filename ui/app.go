// Package ui provides the Fyne-based graphical user interface for FontLoaderSubRe.
package ui

import (
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/ncruces/zenity"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/layout"
	"fyne.io/fyne/v2/widget"

	"FontLoaderSubRe/internal/config"
	"FontLoaderSubRe/internal/db"
	"FontLoaderSubRe/internal/i18n"
	"FontLoaderSubRe/internal/session"
)

// App holds all GUI state.
type App struct {
	fyneApp fyne.App
	win     fyne.Window

	cfg *config.Config
	db  *db.DB
	mgr *session.Manager

	// UI widgets
	lblHeader       *canvas.Text
	lblStatus1      *widget.Label
	lblStatus2      *widget.Label
	progressBar     *widget.ProgressBar
	lblProgressTask *widget.Label
	btnMenu         *widget.Button
	btnDetails      *widget.Button
	btnClose        *widget.Button
	lblFooter       *widget.Hyperlink
	txtDetails      *widget.Entry

	detailsVisible bool
	content        *fyne.Container

	// stats
	loaded        int
	errors        int
	noMatch       int
	subtitles     int
	importedFonts int
}

// New creates and returns a new App. Call Show() to display the window.
func New(a fyne.App, cfg *config.Config, database *db.DB, mgr *session.Manager) *App {
	app := &App{
		fyneApp: a,
		cfg:     cfg,
		db:      database,
		mgr:     mgr,
	}
	app.buildUI()
	return app
}

// Show displays the main window.
func (a *App) Show() { a.win.Show() }

// Window returns the underlying Fyne window.
func (a *App) Window() fyne.Window { return a.win }

// ProcessFiles triggers the extract→load pipeline for the given paths.
func (a *App) ProcessFiles(paths []string) {
	if len(paths) == 0 {
		return
	}
	a.progressBar.Show()
	a.lblProgressTask.Show()
	a.lblProgressTask.SetText(a.strings().MsgExtractingSubs)
	a.progressBar.SetValue(0)
	go a.runExtractWorker(paths)
}

// --- UI construction ---

func (a *App) buildUI() {
	a.win = a.fyneApp.NewWindow("FontLoaderSubRe " + config.Version)
	a.win.SetOnClosed(func() { a.mgr.Cleanup() })

	s := a.strings()

	a.lblHeader = canvas.NewText(s.Header, nil)
	a.lblHeader.TextStyle = fyne.TextStyle{Bold: true}
	a.lblHeader.TextSize = 18

	a.lblStatus1 = widget.NewLabel("")
	a.lblStatus2 = widget.NewLabel("")

	a.progressBar = widget.NewProgressBar()
	a.progressBar.Hide()
	a.lblProgressTask = widget.NewLabel("")
	a.lblProgressTask.Hide()

	a.btnMenu = widget.NewButton(s.BtnMenu, a.showMenu)
	a.btnDetails = widget.NewButton(s.BtnDetails, a.toggleDetails)
	a.btnClose = widget.NewButton(s.BtnClose, func() { a.win.Close() })

	btnRow := container.NewHBox(
		a.btnMenu,
		a.btnDetails,
		layout.NewSpacer(),
		a.btnClose,
	)

	footerURL, _ := url.Parse(i18n.ProjectLink)
	a.lblFooter = widget.NewHyperlink(s.Footer, footerURL)

	a.txtDetails = widget.NewMultiLineEntry()
	a.txtDetails.SetMinRowsVisible(10)
	a.txtDetails.Wrapping = fyne.TextTruncate
	a.txtDetails.Hide()

	a.content = container.NewVBox(
		a.lblHeader,
		a.lblStatus1,
		a.lblStatus2,
		a.progressBar,
		a.lblProgressTask,
		btnRow,
		a.lblFooter,
		a.txtDetails,
	)

	a.win.SetContent(a.content)
	a.win.Resize(fyne.NewSize(480, 200))

	a.win.SetOnDropped(func(_ fyne.Position, uris []fyne.URI) {
		a.handleDrop(uris)
	})

	a.refreshUI()
}

// --- Menu ---

func (a *App) showMenu() {
	s := a.strings()

	mkvLabel := s.MenuMKV
	if a.cfg.MKVExtraction {
		mkvLabel = "✓ " + mkvLabel
	}

	langItem := fyne.NewMenuItem(s.MenuLang, nil)
	langItem.ChildMenu = fyne.NewMenu("",
		fyne.NewMenuItem("English", func() { a.setLang("en_us") }),
		fyne.NewMenuItem("正體中文", func() { a.setLang("zh_tw") }),
		fyne.NewMenuItem("简体中文", func() { a.setLang("zh_cn") }),
	)

	menu := fyne.NewMenu("",
		fyne.NewMenuItem(s.MenuUpdate, a.actionUpdateIndex),
		fyne.NewMenuItem(s.MenuFontBase, a.actionSetFontBase),
		fyne.NewMenuItem(s.MenuExport, a.actionExport),
		fyne.NewMenuItemSeparator(),
		fyne.NewMenuItem(mkvLabel, func() {
			a.cfg.MKVExtraction = !a.cfg.MKVExtraction
			a.cfg.Save() //nolint:errcheck
		}),
		fyne.NewMenuItemSeparator(),
		langItem,
		fyne.NewMenuItemSeparator(),
		fyne.NewMenuItem(s.MenuHelp, a.actionHelp),
		fyne.NewMenuItem(s.MenuClear, a.actionClear),
	)

	widget.NewPopUpMenu(menu, a.win.Canvas()).ShowAtPosition(fyne.NewPos(0, 80))
}

func (a *App) setLang(code string) {
	a.cfg.UILanguage = code
	a.cfg.Save() //nolint:errcheck
	a.refreshUI()
}

// --- Actions ---

func (a *App) actionUpdateIndex() {
	a.btnMenu.Disable()
	a.progressBar.Show()
	a.lblProgressTask.Show()
	go a.runScanWorker()
}

func (a *App) actionSetFontBase() {
	go func() {
		path, err := zenity.SelectFile(
			zenity.Directory(),
			zenity.Title("Select Font Base Folder"),
		)
		if err != nil || path == "" {
			return
		}
		a.cfg.FontBasePath = path
		a.cfg.Save() //nolint:errcheck
		newDB, err := db.Open(a.cfg.DBPath())
		if err != nil {
			dialog.ShowError(err, a.win)
			return
		}
		a.db.Close() //nolint:errcheck
		a.db = newDB
		a.refreshUI()
	}()
}

func (a *App) actionExport() {
	go func() {
		path, err := zenity.SelectFile(
			zenity.Directory(),
			zenity.Title("Select Export Folder"),
		)
		if err != nil || path == "" {
			return
		}
		if exportErr := a.mgr.ExportAll(path); exportErr != nil {
			dialog.ShowError(exportErr, a.win)
			return
		}
		s := a.strings()
		dialog.ShowInformation(s.MenuExport, s.MsgExport, a.win)
	}()
}

func (a *App) actionHelp() {
	s := a.strings()
	dialog.ShowInformation(s.MenuHelp, s.MsgHelp, a.win)
}

func (a *App) actionClear() {
	s := a.strings()
	a.cfg.Reset() //nolint:errcheck
	dialog.ShowInformation(s.MenuClear, s.MsgClear, a.win)
	a.refreshUI()
}

// --- Drag & Drop ---

func (a *App) handleDrop(uris []fyne.URI) {
	validExts := map[string]bool{
		".ass": true, ".ssa": true,
		".ttf": true, ".otf": true, ".ttc": true,
	}
	if a.cfg.MKVExtraction {
		validExts[".mkv"] = true
	}

	var paths []string
	for _, u := range uris {
		p := u.Path()
		fi, err := os.Stat(p)
		if err != nil {
			continue
		}
		if fi.IsDir() {
			filepath.WalkDir(p, func(fp string, d os.DirEntry, e error) error { //nolint:errcheck
				if e == nil && !d.IsDir() && validExts[strings.ToLower(filepath.Ext(fp))] {
					paths = append(paths, fp)
				}
				return nil
			})
		} else if validExts[strings.ToLower(filepath.Ext(p))] {
			paths = append(paths, p)
		}
	}
	if len(paths) > 0 {
		a.ProcessFiles(paths)
	}
}

// --- Details toggle ---

func (a *App) toggleDetails() {
	if a.detailsVisible {
		a.txtDetails.Hide()
		a.win.Resize(fyne.NewSize(480, 200))
	} else {
		a.txtDetails.Show()
		a.win.Resize(fyne.NewSize(480, 400))
	}
	a.detailsVisible = !a.detailsVisible
	a.content.Refresh()
}

// --- UI refresh ---

func (a *App) refreshUI() {
	s := a.strings()
	a.lblHeader.Text = s.Header
	a.lblHeader.Refresh()

	indexFonts := 0
	indexNames := 0
	if v := a.db.MetaGet("file_count"); v != "" {
		fmt.Sscanf(v, "%d", &indexFonts)
	}
	for _, k := range []string{"ps_name_count", "family_name_count", "full_name_count", "unique_id_count"} {
		var n int
		fmt.Sscanf(a.db.MetaGet(k), "%d", &n)
		indexNames += n
	}

	a.lblStatus1.SetText(formatStatus(s.Status1, map[string]int{
		"loaded":   a.loaded,
		"errors":   a.errors,
		"no_match": a.noMatch,
	}))
	a.lblStatus2.SetText(formatStatus(s.Status2, map[string]int{
		"index_fonts":    indexFonts,
		"index_names":    indexNames,
		"subtitles":      a.subtitles,
		"imported_fonts": a.importedFonts,
	}))
	a.btnMenu.SetText(s.BtnMenu)
	a.btnDetails.SetText(s.BtnDetails)
	a.btnClose.SetText(s.BtnClose)
	a.lblFooter.SetText(s.Footer)
}

// strings returns the current locale strings.
func (a *App) strings() i18n.Strings {
	return i18n.Get(a.cfg.UILanguage)
}

// formatStatus replaces {key} tokens in a template with integer values.
func formatStatus(tmpl string, vals map[string]int) string {
	r := tmpl
	for k, v := range vals {
		r = strings.ReplaceAll(r, "{"+k+"}", fmt.Sprintf("%d", v))
	}
	return r
}
