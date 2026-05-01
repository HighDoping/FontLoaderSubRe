// Package scanner scans a directory of font files and populates the database.
package scanner

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"FontLoaderSubRe/internal/db"
	"FontLoaderSubRe/internal/fontmeta"
)

// Stats holds the results of a scan operation.
type Stats struct {
	FilesProcessed int
	FullNames      int
	FamilyNames    int
	PSNames        int
	UniqueIDs      int
}

// ProgressFunc is called with (current, total) after each file is processed.
type ProgressFunc func(current, total int)

const batchSize = 500

// Scan scans fontDir for TTF/OTF/TTC files, rebuilds the database, and
// writes updated metadata statistics.
func Scan(database *db.DB, fontDir string, progress ProgressFunc) (Stats, error) {
	var stats Stats

	// Collect font files.
	files, err := collectFontFiles(fontDir)
	if err != nil {
		return stats, err
	}
	total := len(files)

	if progress != nil {
		progress(0, total)
	}

	// Clean DB before inserting.
	if err := database.Clean(); err != nil {
		return stats, err
	}

	// Worker pool.
	numWorkers := runtime.NumCPU()
	if numWorkers > 16 {
		numWorkers = 16
	}

	type job struct {
		path    string
		fontDir string
	}
	type result struct {
		record *db.FileRecord
	}

	jobs := make(chan job, numWorkers*2)
	results := make(chan result, numWorkers*2)

	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				fd, err := fontmeta.Extract(j.path)
				if err != nil {
					results <- result{nil}
					continue
				}
				rel, err := filepath.Rel(j.fontDir, j.path)
				if err != nil {
					rel = j.path
				}
				rel = filepath.ToSlash(rel)
				rec := &db.FileRecord{
					Hash:         fd.Hash,
					RelativePath: rel,
					FullNames:    fd.FullNames,
					FamilyNames:  fd.FamilyNames,
					PSNames:      fd.PSNames,
					UniqueIDs:    fd.UniqueIDs,
				}
				results <- result{rec}
			}
		}()
	}

	// Close results when all workers done.
	go func() {
		wg.Wait()
		close(results)
	}()

	// Feed jobs.
	go func() {
		for _, p := range files {
			jobs <- job{path: p, fontDir: fontDir}
		}
		close(jobs)
	}()

	// Collect results and batch-insert.
	var batch []db.FileRecord
	processed := 0

	flush := func() error {
		if len(batch) == 0 {
			return nil
		}
		err := database.BulkInsert(batch)
		batch = batch[:0]
		return err
	}

	for r := range results {
		processed++
		if r.record != nil {
			stats.FullNames += len(r.record.FullNames)
			stats.FamilyNames += len(r.record.FamilyNames)
			stats.PSNames += len(r.record.PSNames)
			stats.UniqueIDs += len(r.record.UniqueIDs)
			batch = append(batch, *r.record)
			if len(batch) >= batchSize {
				if err := flush(); err != nil {
					return stats, err
				}
			}
		}
		stats.FilesProcessed = processed
		if progress != nil {
			progress(processed, total)
		}
	}

	if err := flush(); err != nil {
		return stats, err
	}

	// Write metadata.
	if err := database.MetaSet("last_scan", fontDir); err != nil {
		return stats, err
	}
	if err := database.MetaSet("last_scan_time", fmt.Sprintf("%d", time.Now().Unix())); err != nil {
		return stats, err
	}
	if err := database.UpdateScanStats(); err != nil {
		return stats, err
	}

	return stats, nil
}

// collectFontFiles returns all TTF/OTF/TTC file paths under dir.
func collectFontFiles(dir string) ([]string, error) {
	var files []string
	err := filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		ext := strings.ToLower(filepath.Ext(p))
		if ext == ".ttf" || ext == ".otf" || ext == ".ttc" {
			files = append(files, p)
		}
		return nil
	})
	return files, err
}
