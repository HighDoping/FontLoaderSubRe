// Package subtitle parses ASS/SSA subtitle files and extracts font names
// from the [V4+ Styles] / [V4 Styles] sections and inline \fn tags in [Events].
//
// Encoding fallback chain (matches Python): UTF-8 → UTF-16 → GB18030 → UTF-8-sig.
package subtitle

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"golang.org/x/text/encoding"
	"golang.org/x/text/encoding/simplifiedchinese"
	"golang.org/x/text/encoding/unicode"
	"golang.org/x/text/transform"
)

// reFnTag matches inline \fn font-override tags inside Dialogue lines, e.g. {\fnArial}.
var reFnTag = regexp.MustCompile(`\\fn([^\\}]+)`)

// ExtractFonts returns the number of subtitle files processed and a deduplicated
// slice of font names found in all of them.
func ExtractFonts(path string) (fileCount int, fonts []string, err error) {
	info, err := os.Stat(path)
	if err != nil {
		return 0, nil, err
	}

	var files []string
	if info.IsDir() {
		entries, err := walkSubtitles(path)
		if err != nil {
			return 0, nil, err
		}
		files = entries
	} else {
		files = []string{path}
	}

	fontSet := make(map[string]bool)
	for _, f := range files {
		names, err := extractFromFile(f)
		if err != nil {
			continue // skip unreadable files
		}
		for _, n := range names {
			fontSet[n] = true
		}
	}

	result := make([]string, 0, len(fontSet))
	for n := range fontSet {
		result = append(result, n)
	}
	return len(files), result, nil
}

func walkSubtitles(dir string) ([]string, error) {
	var files []string
	err := filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		ext := strings.ToLower(filepath.Ext(p))
		if ext == ".ass" || ext == ".ssa" {
			files = append(files, p)
		}
		return nil
	})
	return files, err
}

// extractFromFile reads one subtitle file (trying multiple encodings) and
// returns all font names from [V4+ Styles] / [V4 Styles] sections.
func extractFromFile(path string) ([]string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var lines []string
	for _, decode := range []func([]byte) (string, bool){
		tryUTF8,
		tryUTF16,
		tryGB18030,
		tryUTF8SIG,
	} {
		if text, ok := decode(raw); ok {
			lines = strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
			break
		}
	}
	if lines == nil {
		lines = strings.Split(string(raw), "\n") // last resort
	}

	return parseStyles(lines), nil
}

// parseStyles scans line-by-line looking for V4/V4+ Styles sections (Fontname
// column) and [Events] Dialogue lines (inline \fn override tags).
func parseStyles(lines []string) []string {
	fontSet := make(map[string]bool)
	section := ""
	fontNameIdx := -1

	for _, rawLine := range lines {
		line := strings.TrimSpace(rawLine)

		// Section detection — every line starting with '[' switches context.
		if strings.HasPrefix(line, "[") {
			if line == "[V4+ Styles]" || line == "[V4 Styles]" {
				section = "styles"
				fontNameIdx = -1
			} else if line == "[Events]" {
				section = "events"
			} else {
				section = ""
			}
			continue
		}

		switch section {
		case "styles":
			if strings.HasPrefix(line, "Format:") {
				headers := strings.Split(line[7:], ",")
				for i, h := range headers {
					if strings.TrimSpace(h) == "Fontname" {
						fontNameIdx = i
						break
					}
				}
				if fontNameIdx == -1 {
					fontNameIdx = 1 // fallback
				}
			} else if strings.HasPrefix(line, "Style:") && fontNameIdx >= 0 {
				parts := strings.Split(line[6:], ",")
				if len(parts) > fontNameIdx {
					name := strings.TrimSpace(parts[fontNameIdx])
					if name != "" {
						fontSet[name] = true
					}
				}
			}
		case "events":
			if strings.HasPrefix(line, "Dialogue:") {
				for _, m := range reFnTag.FindAllStringSubmatch(line, -1) {
					if name := strings.TrimSpace(m[1]); name != "" {
						fontSet[name] = true
					}
				}
			}
		}
	}

	result := make([]string, 0, len(fontSet))
	for n := range fontSet {
		result = append(result, n)
	}
	return result
}

// --- encoding attempts ---

func tryUTF8(b []byte) (string, bool) {
	// Reject if it contains replacement character sequences common in mis-decoded text.
	if !isValidUTF8(b) {
		return "", false
	}
	return string(b), true
}

func isValidUTF8(b []byte) bool {
	for _, r := range string(b) {
		if r == '\uFFFD' {
			return false
		}
	}
	return true
}

func tryUTF8SIG(b []byte) (string, bool) {
	bom := []byte{0xEF, 0xBB, 0xBF}
	if !bytes.HasPrefix(b, bom) {
		return "", false
	}
	rest := b[3:]
	if !isValidUTF8(rest) {
		return "", false
	}
	return string(rest), true
}

func tryUTF16(b []byte) (string, bool) {
	// Check for BOM
	var enc encoding.Encoding
	if len(b) >= 2 {
		if b[0] == 0xFF && b[1] == 0xFE {
			enc = unicode.UTF16(unicode.LittleEndian, unicode.ExpectBOM)
		} else if b[0] == 0xFE && b[1] == 0xFF {
			enc = unicode.UTF16(unicode.BigEndian, unicode.ExpectBOM)
		}
	}
	if enc == nil {
		return "", false
	}
	decoded, err := io.ReadAll(transform.NewReader(bytes.NewReader(b), enc.NewDecoder()))
	if err != nil {
		return "", false
	}
	return string(decoded), true
}

func tryGB18030(b []byte) (string, bool) {
	decoded, err := io.ReadAll(transform.NewReader(
		bytes.NewReader(b),
		simplifiedchinese.GB18030.NewDecoder(),
	))
	if err != nil {
		return "", false
	}
	s := string(decoded)
	if !isValidUTF8([]byte(s)) {
		return "", false
	}
	return s, true
}
