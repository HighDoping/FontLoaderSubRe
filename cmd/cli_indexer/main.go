// CLI indexer for FontLoaderSubRe.
// Mirrors the behaviour of the Python cli_indexer.py exactly.
//
// Usage:
//
//	cli_indexer <font_dir> [--db <path>] [-v]
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"FontLoaderSubRe/internal/db"
	"FontLoaderSubRe/internal/scanner"
)

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: cli_indexer <font_dir> [--db <path>] [-v]\n\n")
		flag.PrintDefaults()
	}

	dbPath := flag.String("db", "", "Path to SQLite database (default: <font_dir>/FontLoaderSubRe.db)")
	verbose := flag.Bool("v", false, "Show debug logging")
	flag.Parse()

	_ = verbose // reserved for future structured logging

	if flag.NArg() < 1 {
		flag.Usage()
		os.Exit(1)
	}

	fontDir, err := filepath.Abs(flag.Arg(0))
	if err != nil || !isDir(fontDir) {
		fmt.Fprintf(os.Stderr, "Error: %q is not a directory\n", flag.Arg(0))
		os.Exit(1)
	}

	target := *dbPath
	if target == "" {
		target = filepath.Join(fontDir, "FontLoaderSubRe.db")
	}
	target, _ = filepath.Abs(target)

	fmt.Println("Starting index update...")
	fmt.Printf("Target Directory: %s\n", fontDir)
	fmt.Printf("Database Path:    %s\n", target)
	fmt.Println(line())

	database, err := db.Open(target)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Database Error: %v\n", err)
		os.Exit(1)
	}
	defer database.Close()

	start := time.Now()
	stats, err := scanner.Scan(database, fontDir, printProgress)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\n\nAn error occurred: %v\n", err)
		os.Exit(1)
	}

	elapsed := time.Since(start)
	fmt.Println()
	fmt.Println(line())
	fmt.Println("Scan Complete!")
	fmt.Printf("Time Elapsed:    %.2f seconds\n", elapsed.Seconds())
	fmt.Printf("Files Processed: %d\n", stats.FilesProcessed)
	fmt.Println("Unique Names Found:")
	fmt.Printf("  - Full Names:   %d\n", stats.FullNames)
	fmt.Printf("  - Family Names: %d\n", stats.FamilyNames)
	fmt.Printf("  - PS Names:     %d\n", stats.PSNames)
	fmt.Printf("  - Unique IDs:   %d\n", stats.UniqueIDs)
}

func printProgress(current, total int) {
	if total == 0 {
		return
	}
	pct := float64(current) / float64(total) * 100
	barLen := 40
	filled := int(float64(barLen) * float64(current) / float64(total))
	bar := repeatChar('█', filled) + repeatChar('-', barLen-filled)
	fmt.Printf("\rIndexing: |%s| %.1f%% (%d/%d)", bar, pct, current, total)
	if current == total {
		fmt.Println()
	}
}

func repeatChar(r rune, n int) string {
	b := make([]rune, n)
	for i := range b {
		b[i] = r
	}
	return string(b)
}

func line() string { return "------------------------------" }

func isDir(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && fi.IsDir()
}
