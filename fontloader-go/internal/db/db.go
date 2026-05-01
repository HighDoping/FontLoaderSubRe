// Package db manages the SQLite font-metadata database.
// Schema matches the original Python implementation exactly.
package db

import (
	"database/sql"
	"fmt"
	"strings"

	_ "modernc.org/sqlite" // register "sqlite" driver
)

// DB wraps the SQLite connection and exposes the operations needed by the app.
type DB struct {
	path string
	conn *sql.DB
}

// Open opens (or creates) the database at the given path.
func Open(path string) (*DB, error) {
	conn, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	conn.SetMaxOpenConns(1) // SQLite: one writer at a time

	_, err = conn.Exec("PRAGMA journal_mode = DELETE")
	if err != nil {
		conn.Close()
		return nil, err
	}
	_, err = conn.Exec("PRAGMA synchronous = NORMAL")
	if err != nil {
		conn.Close()
		return nil, err
	}
	_, err = conn.Exec("PRAGMA foreign_keys = ON")
	if err != nil {
		conn.Close()
		return nil, err
	}

	d := &DB{path: path, conn: conn}
	if err := d.initSchema(); err != nil {
		conn.Close()
		return nil, err
	}
	return d, nil
}

// Close releases the database connection.
func (d *DB) Close() error {
	return d.conn.Close()
}

func (d *DB) initSchema() error {
	_, err := d.conn.Exec(`
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
		CREATE INDEX IF NOT EXISTS idx_font_name   ON file_fullname (font_name);
		CREATE INDEX IF NOT EXISTS idx_family_name ON file_families (family_name);
		CREATE INDEX IF NOT EXISTS idx_ps_name     ON file_psnames  (ps_name);
		CREATE INDEX IF NOT EXISTS idx_unique_id   ON file_uniqueids(unique_id);
	`)
	return err
}

// --- Batch Insert ---

// FileRecord represents one font file in the database.
type FileRecord struct {
	Hash        string
	RelativePath string
	FullNames   []string
	FamilyNames []string
	PSNames     []string
	UniqueIDs   []string
}

// BulkInsert inserts (or replaces) a batch of font file records in a single transaction.
func (d *DB) BulkInsert(records []FileRecord) error {
	if len(records) == 0 {
		return nil
	}

	tx, err := d.conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback() //nolint:errcheck

	upsertFile, err := tx.Prepare(`
		INSERT INTO files (file_hash, path) VALUES (?, ?)
		ON CONFLICT(file_hash) DO UPDATE SET path=excluded.path
	`)
	if err != nil {
		return err
	}
	defer upsertFile.Close()

	insFull, err := tx.Prepare(`INSERT OR IGNORE INTO file_fullname   (file_hash, font_name)   VALUES (?, ?)`)
	if err != nil {
		return err
	}
	defer insFull.Close()

	insFam, err := tx.Prepare(`INSERT OR IGNORE INTO file_families  (file_hash, family_name) VALUES (?, ?)`)
	if err != nil {
		return err
	}
	defer insFam.Close()

	insPS, err := tx.Prepare(`INSERT OR IGNORE INTO file_psnames   (file_hash, ps_name)     VALUES (?, ?)`)
	if err != nil {
		return err
	}
	defer insPS.Close()

	insUID, err := tx.Prepare(`INSERT OR IGNORE INTO file_uniqueids (file_hash, unique_id)   VALUES (?, ?)`)
	if err != nil {
		return err
	}
	defer insUID.Close()

	for _, r := range records {
		if _, err := upsertFile.Exec(r.Hash, r.RelativePath); err != nil {
			return err
		}
		for _, n := range r.FullNames {
			if _, err := insFull.Exec(r.Hash, n); err != nil {
				return err
			}
		}
		for _, n := range r.FamilyNames {
			if _, err := insFam.Exec(r.Hash, n); err != nil {
				return err
			}
		}
		for _, n := range r.PSNames {
			if _, err := insPS.Exec(r.Hash, n); err != nil {
				return err
			}
		}
		for _, n := range r.UniqueIDs {
			if _, err := insUID.Exec(r.Hash, n); err != nil {
				return err
			}
		}
	}

	return tx.Commit()
}

// --- Queries ---

// SearchResult is a (relativePath, hash) pair returned by font lookup.
type SearchResult struct {
	RelativePath string
	Hash         string
}

// SearchByFont searches all name tables for the given font name.
// The logic matches the Python search_by_font exactly:
//   - Exact match on full_name
//   - LIKE match on family_name, ps_name, unique_id
//   - If no results and name starts with "@", retry without "@"
func (d *DB) SearchByFont(fontName string) ([]SearchResult, error) {
	results, err := d.searchByFontExact(fontName)
	if err != nil {
		return nil, err
	}
	if len(results) == 0 && strings.HasPrefix(fontName, "@") {
		results, err = d.searchByFontExact(fontName[1:])
		if err != nil {
			return nil, err
		}
	}
	return results, nil
}

func (d *DB) searchByFontExact(fontName string) ([]SearchResult, error) {
	wild := "%" + fontName + "%"
	rows, err := d.conn.Query(`
		SELECT DISTINCT f.path, f.file_hash
		FROM files f
		LEFT JOIN file_fullname  ff   ON f.file_hash = ff.file_hash
		LEFT JOIN file_families  fam  ON f.file_hash = fam.file_hash
		LEFT JOIN file_psnames   fps  ON f.file_hash = fps.file_hash
		LEFT JOIN file_uniqueids fuid ON f.file_hash = fuid.file_hash
		WHERE ff.font_name    = ?
		   OR fam.family_name LIKE ?
		   OR fps.ps_name     LIKE ?
		   OR fuid.unique_id  LIKE ?
	`, fontName, wild, wild, wild)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []SearchResult
	for rows.Next() {
		var r SearchResult
		if err := rows.Scan(&r.RelativePath, &r.Hash); err != nil {
			return nil, err
		}
		results = append(results, r)
	}
	return results, rows.Err()
}

// --- Metadata ---

// MetaGet retrieves a metadata value, returning "" when not set.
func (d *DB) MetaGet(key string) string {
	var val string
	d.conn.QueryRow(`SELECT value FROM metadata WHERE key = ?`, key).Scan(&val) //nolint:errcheck
	return val
}

// MetaSet stores a metadata key/value pair.
func (d *DB) MetaSet(key, value string) error {
	_, err := d.conn.Exec(`
		INSERT INTO metadata (key, value) VALUES (?, ?)
		ON CONFLICT(key) DO UPDATE SET value=excluded.value
	`, key, value)
	return err
}

// TableLen returns the row count of a whitelisted table.
func (d *DB) TableLen(tableName string) (int64, error) {
	allowed := map[string]bool{
		"files": true, "file_fullname": true, "file_families": true,
		"file_psnames": true, "file_uniqueids": true, "metadata": true,
	}
	if !allowed[tableName] {
		return 0, fmt.Errorf("invalid table name: %q", tableName)
	}
	var count int64
	err := d.conn.QueryRow(`SELECT COUNT(*) FROM ` + tableName).Scan(&count)
	return count, err
}

// Clean removes all font data from the database and runs VACUUM.
func (d *DB) Clean() error {
	_, err := d.conn.Exec(`
		DELETE FROM files;
		DELETE FROM file_fullname;
		DELETE FROM file_families;
		DELETE FROM file_psnames;
		DELETE FROM file_uniqueids;
		VACUUM;
	`)
	if err != nil {
		return err
	}
	for _, k := range []string{
		"last_scan", "last_scan_time", "file_count",
		"ps_name_count", "family_name_count", "full_name_count", "unique_id_count",
	} {
		d.conn.Exec(`DELETE FROM metadata WHERE key = ?`, k) //nolint:errcheck
	}
	return nil
}

// UpdateScanStats writes post-scan row counts to the metadata table.
func (d *DB) UpdateScanStats() error {
	for _, pair := range []struct{ key, table string }{
		{"file_count", "files"},
		{"ps_name_count", "file_psnames"},
		{"family_name_count", "file_families"},
		{"full_name_count", "file_fullname"},
		{"unique_id_count", "file_uniqueids"},
	} {
		n, err := d.TableLen(pair.table)
		if err != nil {
			return err
		}
		if err := d.MetaSet(pair.key, fmt.Sprintf("%d", n)); err != nil {
			return err
		}
	}
	return nil
}
