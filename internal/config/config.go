// Package config manages persistent application settings via a JSON file
// stored in the platform-appropriate user config directory.
package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
)

const (
	AppName = "FontLoaderSubRe"
	DBName  = "FontLoaderSubRe.db"
	Version = "0.2.2"
)

// Config holds all persisted application settings.
type Config struct {
	FontBasePath  string `json:"font_base_path"`
	UILanguage    string `json:"ui_language"`
	MKVExtraction bool   `json:"mkv_extraction"`

	configFile string `json:"-"`
}

// defaults returns a fresh Config with sensible defaults.
func defaults() Config {
	cwd, _ := os.Getwd()
	return Config{
		FontBasePath:  cwd,
		UILanguage:    "en_us",
		MKVExtraction: false,
	}
}

// Load loads (or creates) the application config from disk.
func Load() (*Config, error) {
	dir, err := configDir()
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return nil, err
	}

	path := filepath.Join(dir, "settings.json")
	cfg := defaults()
	cfg.configFile = path

	data, err := os.ReadFile(path)
	if err == nil {
		// Best-effort decode; keep defaults for missing keys.
		_ = json.Unmarshal(data, &cfg)
		cfg.configFile = path // unmarshal would overwrite the field tag "-" but we set it back
	}
	return &cfg, nil
}

// Save writes the current settings to disk.
func (c *Config) Save() error {
	data, err := json.MarshalIndent(c, "", "    ")
	if err != nil {
		return err
	}
	return os.WriteFile(c.configFile, data, 0o640)
}

// Reset restores factory defaults and persists them.
func (c *Config) Reset() error {
	d := defaults()
	c.FontBasePath = d.FontBasePath
	c.UILanguage = d.UILanguage
	c.MKVExtraction = d.MKVExtraction
	return c.Save()
}

// DBPath returns the full path to the SQLite database.
func (c *Config) DBPath() string {
	return filepath.Join(c.FontBasePath, DBName)
}

// configDir returns the OS-appropriate user config directory for the app.
func configDir() (string, error) {
	switch runtime.GOOS {
	case "windows":
		if d := os.Getenv("APPDATA"); d != "" {
			return filepath.Join(d, AppName), nil
		}
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		return filepath.Join(home, "AppData", "Roaming", AppName), nil
	case "darwin":
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		return filepath.Join(home, "Library", "Application Support", AppName), nil
	default: // Linux / BSD
		if d := os.Getenv("XDG_CONFIG_HOME"); d != "" {
			return filepath.Join(d, AppName), nil
		}
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		return filepath.Join(home, ".config", AppName), nil
	}
}
