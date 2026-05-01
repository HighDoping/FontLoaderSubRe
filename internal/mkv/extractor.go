// Package mkv extracts ASS/SSA subtitles and font attachments from MKV files
// using the external mkvtoolnix tools (mkvmerge + mkvextract).
package mkv

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// Result holds the output of an MKV extraction.
type Result struct {
	TempDir string
	Files   []string // absolute paths of extracted files
}

// Extract extracts subtitle tracks and/or font attachments from an MKV file
// into a temporary directory. The caller is responsible for cleaning up TempDir.
func Extract(mkvPath string, extractSubs, extractFonts bool) (*Result, error) {
	for _, tool := range []string{"mkvmerge", "mkvextract"} {
		if _, err := exec.LookPath(tool); err != nil {
			return nil, fmt.Errorf("missing dependency: %q not found in PATH", tool)
		}
	}

	if _, err := os.Stat(mkvPath); err != nil {
		return nil, fmt.Errorf("input file not found: %s", mkvPath)
	}

	// 1. Read metadata via mkvmerge -J
	out, err := exec.Command("mkvmerge", "-J", mkvPath).Output()
	if err != nil {
		return nil, fmt.Errorf("mkvmerge metadata failed: %w", err)
	}
	var meta mkvMeta
	if err := json.Unmarshal(out, &meta); err != nil {
		return nil, fmt.Errorf("failed to parse mkvmerge JSON: %w", err)
	}

	tempDir, err := os.MkdirTemp("", "mkv_extract_")
	if err != nil {
		return nil, err
	}

	res := &Result{TempDir: tempDir}

	// 2. Subtitle tracks
	if extractSubs {
		var trackArgs []string
		for _, tr := range meta.Tracks {
			codec := strings.ToUpper(tr.Properties.CodecID)
			if tr.Type != "subtitles" {
				continue
			}
			if !strings.Contains(codec, "S_TEXT/ASS") && !strings.Contains(codec, "S_TEXT/SSA") {
				continue
			}
			ext := ".ass"
			if strings.Contains(codec, "SSA") {
				ext = ".ssa"
			}
			name := tr.Properties.TrackName
			if name == "" {
				name = fmt.Sprintf("track_%d", tr.ID)
			}
			safeName := sanitizeFilename(name)
			outPath := filepath.Join(tempDir, safeName+ext)
			trackArgs = append(trackArgs, fmt.Sprintf("%d:%s", tr.ID, outPath))
			res.Files = append(res.Files, outPath)
		}
		if len(trackArgs) > 0 {
			args := append([]string{"tracks", mkvPath}, trackArgs...)
			if err := exec.Command("mkvextract", args...).Run(); err != nil {
				// Non-fatal: continue with fonts.
				_ = err
			}
		}
	}

	// 3. Font attachments
	if extractFonts {
		var attachArgs []string
		for _, att := range meta.Attachments {
			mime := strings.ToLower(att.ContentType)
			if !isFontMIME(mime) {
				continue
			}
			safeName := sanitizeFilename(att.FileName)
			outPath := filepath.Join(tempDir, safeName)
			attachArgs = append(attachArgs, fmt.Sprintf("%d:%s", att.ID, outPath))
			res.Files = append(res.Files, outPath)
		}
		if len(attachArgs) > 0 {
			args := append([]string{"attachments", mkvPath}, attachArgs...)
			if err := exec.Command("mkvextract", args...).Run(); err != nil {
				_ = err
			}
		}
	}

	if len(res.Files) == 0 {
		os.RemoveAll(tempDir) //nolint:errcheck
		return nil, nil       // nothing extracted
	}
	return res, nil
}

// Cleanup removes the temporary extraction directory.
func (r *Result) Cleanup() {
	if r != nil && r.TempDir != "" {
		os.RemoveAll(r.TempDir) //nolint:errcheck
	}
}

// --- mkvmerge JSON types ---

type mkvMeta struct {
	Tracks      []mkvTrack      `json:"tracks"`
	Attachments []mkvAttachment `json:"attachments"`
}

type mkvTrack struct {
	ID         int    `json:"id"`
	Type       string `json:"type"`
	Properties struct {
		CodecID   string `json:"codec_id"`
		TrackName string `json:"track_name"`
	} `json:"properties"`
}

type mkvAttachment struct {
	ID          int    `json:"id"`
	FileName    string `json:"file_name"`
	ContentType string `json:"content_type"`
}

// --- helpers ---

func isFontMIME(mime string) bool {
	fontMIMEs := []string{
		"application/x-truetype-font",
		"application/vnd.ms-opentype",
		"application/x-font-ttf",
		"application/x-font-otf",
		"font/",
	}
	for _, m := range fontMIMEs {
		if strings.HasPrefix(mime, m) || mime == m {
			return true
		}
	}
	return false
}

func sanitizeFilename(name string) string {
	var sb strings.Builder
	for _, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') ||
			r == '.' || r == '_' || r == '-' || r == ' ' {
			sb.WriteRune(r)
		}
	}
	return strings.TrimSpace(sb.String())
}
