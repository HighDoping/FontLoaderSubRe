// Package loader installs and removes fonts from the user font directory
// on macOS, Linux, and Windows.
package loader

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
)

// Load copies (or registers on Windows) a font file into the user font directory.
// Returns the path where the font was installed.
func Load(srcPath string) (string, error) {
	switch runtime.GOOS {
	case "darwin", "linux":
		return loadUnix(srcPath)
	case "windows":
		return loadWindows(srcPath)
	default:
		return "", fmt.Errorf("unsupported OS: %s", runtime.GOOS)
	}
}

// Unload removes a previously installed font (Unix: delete file, Windows: remove registration).
// A missing file on Unix is silently ignored (mirrors Python's missing_ok=True).
func Unload(installedPath string) error {
	switch runtime.GOOS {
	case "darwin", "linux":
		err := os.Remove(installedPath)
		if err != nil && !os.IsNotExist(err) {
			return err
		}
		return nil
	case "windows":
		return unloadWindows(installedPath)
	default:
		return fmt.Errorf("unsupported OS: %s", runtime.GOOS)
	}
}

// CacheRefresh updates the font cache after loading/unloading.
// On Windows: broadcasts WM_FONTCHANGE via SendMessageTimeoutW.
// On Linux: runs fc-cache.
// On macOS: no-op (fonts in ~/Library/Fonts are picked up automatically).
func CacheRefresh() {
	switch runtime.GOOS {
	case "windows":
		cacheRefreshWindows()
	case "linux":
		runFcCache()
	}
}

// FontDir returns the user-level font installation directory for the current OS.
func FontDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	switch runtime.GOOS {
	case "darwin":
		return filepath.Join(home, "Library", "Fonts"), nil
	case "linux":
		return filepath.Join(home, ".local", "share", "fonts"), nil
	case "windows":
		// On Windows we load in-place, but we still return the Fonts directory
		// as a fallback destination.
		if d := os.Getenv("LOCALAPPDATA"); d != "" {
			return filepath.Join(d, "Microsoft", "Windows", "Fonts"), nil
		}
		return filepath.Join(home, "AppData", "Local", "Microsoft", "Windows", "Fonts"), nil
	default:
		return "", fmt.Errorf("unsupported OS: %s", runtime.GOOS)
	}
}

// --- Unix (macOS / Linux) ---

func loadUnix(srcPath string) (string, error) {
	dir, err := FontDir()
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return "", err
	}
	destPath := filepath.Join(dir, filepath.Base(srcPath))
	if _, err := os.Stat(destPath); err == nil {
		// Already exists — treat as success.
		return destPath, nil
	}
	if err := copyFile(srcPath, destPath); err != nil {
		return "", err
	}
	return destPath, nil
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
