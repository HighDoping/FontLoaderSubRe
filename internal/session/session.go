// Package session tracks fonts loaded during the current application session.
package session

import (
	"io"
	"os"
	"path/filepath"
	"sync"

	"FontLoaderSubRe/internal/loader"
)

// Status constants for a font entry.
const (
	StatusPending   = "Initializing"
	StatusOK        = "Success"
	StatusFailed    = "OS Load Failed"
	StatusUnloaded  = "Unloaded"
	StatusAlreadyOK = "Already loaded"
)

// Entry records the state of one font file.
type Entry struct {
	Hash          string
	SourcePath    string
	InstalledPath string
	Loaded        bool
	Message       string
}

// Manager tracks all font install attempts for the session.
type Manager struct {
	mu     sync.Mutex
	status map[string]*Entry
}

// New creates a new Manager.
func New() *Manager {
	return &Manager{status: make(map[string]*Entry)}
}

// Load installs a font and records the result. Returns true on success.
func (m *Manager) Load(srcPath, hash string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()

	e, exists := m.status[hash]
	if !exists {
		e = &Entry{Hash: hash, SourcePath: srcPath, Message: StatusPending}
		m.status[hash] = e
	} else if !e.Loaded {
		e.SourcePath = srcPath
	}
	if e.Loaded {
		e.Message = StatusAlreadyOK
		return true
	}

	installed, err := loader.Load(srcPath)
	if err != nil {
		e.Message = err.Error()
		return false
	}
	e.InstalledPath = installed
	e.Loaded = true
	e.Message = StatusOK
	return true
}

// GetStatus returns a copy of the entry for a given hash, or nil.
func (m *Manager) GetStatus(hash string) *Entry {
	m.mu.Lock()
	defer m.mu.Unlock()
	if e, ok := m.status[hash]; ok {
		cp := *e
		return &cp
	}
	return nil
}

// Cleanup unloads all loaded fonts and refreshes the font cache.
func (m *Manager) Cleanup() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, e := range m.status {
		if e.Loaded && e.InstalledPath != "" {
			loader.Unload(e.InstalledPath) //nolint:errcheck
			e.Loaded = false
			e.Message = StatusUnloaded
		}
	}
	loader.CacheRefresh()
}

// CacheRefresh updates the font cache (Linux only).
func (m *Manager) CacheRefresh() { loader.CacheRefresh() }

// CountLogLines counts [ok]/[xx]/[??] prefixes in a slice of log lines.
func CountLogLines(lines []string) (ok, errors, noMatch int) {
	for _, l := range lines {
		if len(l) >= 4 {
			switch l[:4] {
			case "[ok]":
				ok++
			case "[xx]":
				errors++
			case "[??]":
				noMatch++
			}
		}
	}
	return
}

// ExportAll copies all installed fonts to destDir.
func (m *Manager) ExportAll(destDir string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, e := range m.status {
		if !e.Loaded || e.InstalledPath == "" {
			continue
		}
		dst := filepath.Join(destDir, filepath.Base(e.InstalledPath))
		copyFile(e.InstalledPath, dst) //nolint:errcheck
	}
	return nil
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o640)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, in)
	return err
}
