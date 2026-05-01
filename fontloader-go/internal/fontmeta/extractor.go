// Package fontmeta extracts OpenType/TrueType name-table metadata from font
// files (TTF, OTF, TTC).
//
// Only the bytes required to locate and read the 'name' table are loaded from
// disk — the sfnt header, table directory, and name table itself — so large
// CJK fonts are not fully buffered into memory.
//
// Extracted name IDs:
//   - 1  Family name
//   - 3  Unique font identifier
//   - 4  Full font name
//   - 6  PostScript name
package fontmeta

import (
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"os"
	"sort"
	"strings"
	"unicode/utf16"
	"unicode/utf8"

	"github.com/kalafut/imohash"
	"golang.org/x/text/encoding"
	"golang.org/x/text/encoding/charmap"
	"golang.org/x/text/encoding/japanese"
	"golang.org/x/text/encoding/korean"
	"golang.org/x/text/encoding/simplifiedchinese"
	"golang.org/x/text/encoding/traditionalchinese"
)

// SubFontData holds metadata for one sub-font within a font file.
// For TTF/OTF files there is exactly one entry; TTC files have one per
// embedded font.
type SubFontData struct {
	FullNames   []string
	FamilyNames []string
	PSNames     []string
	UniqueIDs   []string
}

// FontData holds the extracted metadata for a single font file.
type FontData struct {
	FilePath    string
	Hash        string // imohash hex string (lowercase)
	FullNames   []string
	FamilyNames []string
	PSNames     []string
	UniqueIDs   []string
	// SubFonts contains per-sub-font metadata. For TTC files each element
	// corresponds to one embedded font in declaration order.
	SubFonts []SubFontData
	IsTTC    bool
	TTCCount int // number of sub-fonts (0 for non-TTC)
}

// Extract reads a font file (TTF / OTF / TTC) and returns its metadata.
// Only the sfnt headers, table directories, and 'name' tables are read from
// disk — the rest of the file is never loaded into memory.
func Extract(path string) (*FontData, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	fileSize := fi.Size()

	h, err := imohash.SumFile(path)
	if err != nil {
		return nil, err
	}

	fd := &FontData{
		FilePath: path,
		Hash:     hex.EncodeToString(h[:]),
	}

	// Per-field seen maps so a value in FullNames does not suppress the same
	// value in FamilyNames (and vice versa) — they are semantically distinct.
	seenFull := make(map[string]bool)
	seenFam := make(map[string]bool)
	seenPS := make(map[string]bool)
	seenUID := make(map[string]bool)
	addUniq := func(slice *[]string, seen map[string]bool, val string) {
		val = strings.TrimSpace(val)
		if val == "" || val == "N/A" || seen[val] {
			return
		}
		seen[val] = true
		*slice = append(*slice, val)
	}
	merge := func(sub SubFontData) {
		for _, v := range sub.FullNames {
			addUniq(&fd.FullNames, seenFull, v)
		}
		for _, v := range sub.FamilyNames {
			addUniq(&fd.FamilyNames, seenFam, v)
		}
		for _, v := range sub.PSNames {
			addUniq(&fd.PSNames, seenPS, v)
		}
		for _, v := range sub.UniqueIDs {
			addUniq(&fd.UniqueIDs, seenUID, v)
		}
	}

	// Read the first 12 bytes to detect TTC and retrieve the font count.
	var hdr [12]byte
	if _, err := f.ReadAt(hdr[:], 0); err != nil {
		return nil, fmt.Errorf("fontmeta: reading header: %w", err)
	}

	if string(hdr[:4]) == "ttcf" {
		// TTC header: tag[4] + majorVer[2] + minorVer[2] + numFonts[4] + offsets[…]
		numFonts := binary.BigEndian.Uint32(hdr[8:12])
		if numFonts > 1000 {
			return nil, fmt.Errorf("fontmeta: suspiciously high TTC font count (%d)", numFonts)
		}
		fd.IsTTC = true
		fd.TTCCount = int(numFonts)

		offBuf := make([]byte, numFonts*4)
		if _, err := f.ReadAt(offBuf, 12); err != nil {
			return nil, fmt.Errorf("fontmeta: reading TTC offset table: %w", err)
		}
		for i := 0; i < int(numFonts); i++ {
			sfntOff := int64(binary.BigEndian.Uint32(offBuf[i*4:]))
			nameBytes, err := readNameTableBytes(f, fileSize, sfntOff)
			if err != nil {
				return nil, err
			}
			sub := parseAllNamesFromBytes(nameBytes)
			fd.SubFonts = append(fd.SubFonts, sub)
			merge(sub)
		}
	} else {
		nameBytes, err := readNameTableBytes(f, fileSize, 0)
		if err != nil {
			return nil, err
		}
		sub := parseAllNamesFromBytes(nameBytes)
		fd.SubFonts = []SubFontData{sub}
		merge(sub)
	}

	return fd, nil
}

// readNameTableBytes reads only the 'name' table from f for the sfnt at
// sfntOffset. It issues three small ReadAt calls: sfnt header, table directory,
// and the name table itself. Returns nil if the table is absent.
func readNameTableBytes(f *os.File, fileSize int64, sfntOffset int64) ([]byte, error) {
	// sfnt offset table: sfVersion[4] + numTables[2] + …
	var sfntHdr [6]byte
	if _, err := f.ReadAt(sfntHdr[:], sfntOffset); err != nil {
		return nil, fmt.Errorf("fontmeta: reading sfnt header at %d: %w", sfntOffset, err)
	}
	numTables := int(binary.BigEndian.Uint16(sfntHdr[4:]))

	// Table directory: numTables × 16 bytes, starts at sfntOffset + 12.
	dirBuf := make([]byte, numTables*16)
	if _, err := f.ReadAt(dirBuf, sfntOffset+12); err != nil {
		return nil, fmt.Errorf("fontmeta: reading table directory: %w", err)
	}
	for i := 0; i < numTables; i++ {
		rec := dirBuf[i*16:]
		if string(rec[0:4]) == "name" {
			tblOff := int64(binary.BigEndian.Uint32(rec[8:12]))
			tblLen := int64(binary.BigEndian.Uint32(rec[12:16]))
			if tblOff+tblLen > fileSize {
				return nil, fmt.Errorf("fontmeta: name table out of bounds")
			}
			data := make([]byte, tblLen)
			if _, err := f.ReadAt(data, tblOff); err != nil {
				return nil, fmt.Errorf("fontmeta: reading name table: %w", err)
			}
			return data, nil
		}
	}
	return nil, nil
}

// parseAllNamesFromBytes decodes all name strings from a raw 'name' table.
func parseAllNamesFromBytes(nameTable []byte) SubFontData {
	if len(nameTable) == 0 {
		return SubFontData{}
	}
	full, fam, ps, uid := parseNameTable(nameTable, 0)
	return SubFontData{
		FullNames:   full,
		FamilyNames: fam,
		PSNames:     ps,
		UniqueIDs:   uid,
	}
}

// parseNameTable walks every record in an OpenType 'name' table and returns
// unique decoded strings for nameIDs 1, 3, 4, 6 across all
// (platform, encoding, language) combinations.
func parseNameTable(src []byte, tableOff int) (full, fam, ps, uid []string) {
	// name table header: format[2] + count[2] + stringOffset[2]
	if tableOff+6 > len(src) {
		return
	}
	count := int(binary.BigEndian.Uint16(src[tableOff+2:]))
	strOff := int(binary.BigEndian.Uint16(src[tableOff+4:]))
	strBase := tableOff + strOff

	recBase := tableOff + 6
	if recBase+count*12 > len(src) {
		return
	}

	type nameKey struct {
		platformID uint16
		langID     uint16
		nameID     uint16
	}
	collected := make(map[nameKey]string)

	for i := 0; i < count; i++ {
		rec := src[recBase+i*12:]
		platformID := binary.BigEndian.Uint16(rec[0:])
		encodingID := binary.BigEndian.Uint16(rec[2:])
		langID := binary.BigEndian.Uint16(rec[4:])
		nameID := binary.BigEndian.Uint16(rec[6:])
		length := int(binary.BigEndian.Uint16(rec[8:]))
		relOff := int(binary.BigEndian.Uint16(rec[10:]))

		if nameID != 1 && nameID != 3 && nameID != 4 && nameID != 6 {
			continue
		}
		if length == 0 {
			continue
		}
		start := strBase + relOff
		if start < 0 || start+length > len(src) {
			continue
		}
		raw := src[start : start+length]

		var decoded string
		switch platformID {
		case 0: // Unicode — UTF-16 BE
			decoded = decodeUTF16BE(raw)
		case 3: // Windows — UTF-16 BE for eid 0/1/10; legacy CJK otherwise
			decoded = decodeWindows(encodingID, raw)
		case 1: // Macintosh
			decoded = decodeMac(encodingID, raw)
		default:
			// Unknown platform — skip to avoid storing garbage
		}
		decoded = strings.TrimSpace(decoded)
		if decoded == "" {
			continue
		}
		k := nameKey{platformID, langID, nameID}
		if _, ok := collected[k]; !ok {
			collected[k] = decoded
		}
	}

	// Sort: Windows (3) first, then by langID asc so English (0x0409) leads.
	keys := make([]nameKey, 0, len(collected))
	for k := range collected {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		a, b := keys[i], keys[j]
		if a.platformID != b.platformID {
			return a.platformID > b.platformID
		}
		if a.langID != b.langID {
			return a.langID < b.langID
		}
		return a.nameID < b.nameID
	})

	// Each nameID has its own seen-set so that a value present in family
	// (nameID=1) is not suppressed when it also appears in full (nameID=4).
	seenFull := make(map[string]bool)
	seenFam := make(map[string]bool)
	seenPS := make(map[string]bool)
	seenUID := make(map[string]bool)
	addUniq := func(s *[]string, seen map[string]bool, val string) {
		if !seen[val] {
			seen[val] = true
			*s = append(*s, val)
		}
	}
	for _, k := range keys {
		val := collected[k]
		switch k.nameID {
		case 4:
			addUniq(&full, seenFull, val)
		case 1:
			addUniq(&fam, seenFam, val)
		case 6:
			addUniq(&ps, seenPS, val)
		case 3:
			addUniq(&uid, seenUID, val)
		}
	}
	return
}

// decodeWindows decodes a Windows-platform (platformID=3) name string.
// eid 0/1/10 are defined as UTF-16 BE by the spec.
// eid 2-6 are legacy CJK code pages; some fonts store them as raw CJK bytes,
// others store each CJK byte zero-padded to a uint16 BE — we handle both.
func decodeWindows(encodingID uint16, b []byte) string {
	switch encodingID {
	case 0, 1, 10:
		return decodeUTF16BE(b)
	case 2:
		return decodeCJKorUTF16(b, japanese.ShiftJIS)
	case 3:
		return decodeCJKorUTF16(b, simplifiedchinese.GBK)
	case 4:
		return decodeCJKorUTF16(b, traditionalchinese.Big5)
	case 5:
		return decodeCJKorUTF16(b, korean.EUCKR)
	default:
		return decodeUTF16BE(b)
	}
}

// decodeMac decodes a Macintosh-platform (platformID=1) name string.
func decodeMac(encodingID uint16, b []byte) string {
	switch encodingID {
	case 0:
		return decodeMacRoman(b)
	case 1:
		return decodeCJKbytes(b, japanese.ShiftJIS)
	case 2:
		return decodeCJKbytes(b, traditionalchinese.Big5)
	case 3:
		return decodeCJKbytes(b, korean.EUCKR)
	case 25:
		return decodeCJKbytes(b, simplifiedchinese.GBK)
	default:
		s := string(b)
		if !utf8.ValidString(s) || strings.ContainsRune(s, 0) {
			return ""
		}
		return s
	}
}

// decodeCJKbytes decodes raw CJK bytes (Mac platform) with the given codec.
func decodeCJKbytes(b []byte, enc encoding.Encoding) string {
	out, err := enc.NewDecoder().Bytes(b)
	if err != nil || !utf8.ValidString(string(out)) || strings.ContainsRune(string(out), 0) {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// decodeCJKorUTF16 decodes Windows-platform CJK name strings.
//
// Some old fonts store raw CJK bytes (e.g. GBK pairs BB AA BF B5…); others
// store each CJK byte zero-padded to a uint16 BE (00BB 00AA 00BF 00B5…).
// We detect the zero-padded pattern first, then try raw CJK, then fall back
// to UTF-16 BE so modern Unicode fonts are unaffected.
func decodeCJKorUTF16(b []byte, enc encoding.Encoding) string {
	// Detect zero-padded pattern: every even-indexed byte is 0x00.
	if len(b)%2 == 0 && len(b) > 0 {
		allZero := true
		for i := 0; i < len(b); i += 2 {
			if b[i] != 0 {
				allZero = false
				break
			}
		}
		if allZero {
			odd := make([]byte, len(b)/2)
			for i := range odd {
				odd[i] = b[i*2+1]
			}
			if s := decodeCJKbytes(odd, enc); s != "" {
				return s
			}
		}
	}
	// Try raw CJK bytes.
	if s := decodeCJKbytes(b, enc); s != "" {
		return s
	}
	// Fall back to UTF-16 BE (correct for modern Unicode fonts).
	return decodeUTF16BE(b)
}

func decodeUTF16BE(b []byte) string {
	if len(b)%2 != 0 {
		return ""
	}
	u16 := make([]uint16, len(b)/2)
	for i := range u16 {
		u16[i] = binary.BigEndian.Uint16(b[i*2:])
	}
	return string(utf16.Decode(u16))
}

func decodeMacRoman(b []byte) string {
	out, err := charmap.Macintosh.NewDecoder().Bytes(b)
	if err != nil {
		return string(b)
	}
	return string(out)
}

