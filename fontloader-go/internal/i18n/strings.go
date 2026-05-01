// Package i18n provides localized string tables for EN, Traditional Chinese, and Simplified Chinese.
package i18n

// Strings holds all UI text for one locale.
type Strings struct {
	Header            string
	Status1           string // template: {loaded}, {errors}, {no_match}
	Status2           string // template: {index_fonts}, {index_names}, {subtitles}, {imported_fonts}
	BtnMenu           string
	BtnDetails        string
	BtnClose          string
	BtnChange         string
	MenuUpdate        string
	MenuFontBase      string
	SelectFontBase    string
	MenuExport        string
	MenuMKV           string
	MenuHelp          string
	MenuLang          string
	MenuClear         string
	TitleFontBase     string
	LblCurrentPath    string
	MsgExport         string
	MsgHelp           string
	MsgUpdateComplete string
	MsgExtractingSubs string
	MsgClear          string
	Footer            string
	Error             string
}

// ProjectLink is the canonical project URL.
const ProjectLink = "https://github.com/HighDoping/FontLoaderSubRe"

var locales = map[string]Strings{
	"en_us": {
		Header:            "Finished",
		Status1:           "{loaded} fonts loaded, {errors} errors, {no_match} unmatched.",
		Status2:           "Index: {index_fonts} fonts, {index_names} names; {subtitles} subtitles, {imported_fonts} imported fonts.",
		BtnMenu:           "Menu",
		BtnDetails:        "Details",
		BtnClose:          "Close",
		BtnChange:         "Change...",
		MenuUpdate:        "Update Index",
		MenuFontBase:      "Set Font Base",
		SelectFontBase:    "Select Font Base Directory",
		MenuExport:        "Export Fonts",
		MenuMKV:           "Extract from MKV",
		MenuHelp:          "Help",
		MenuLang:          "Language",
		MenuClear:         "Clear Settings",
		TitleFontBase:     "Font Base Settings",
		LblCurrentPath:    "Current Base Path:",
		MsgExport:         "Export complete.",
		MsgHelp:           "1. Open FontLoaderSubRe, set font directory path, index the fonts from menu. (Only needed first time)\n2. Drag-and-drop subtitles SSA/ASS files or folders onto the FontLoaderSubRe window.",
		MsgUpdateComplete: "Index update complete.",
		MsgExtractingSubs: "Extracting font names from subtitles...",
		MsgClear:          "Settings cleared",
		Footer:            "GPLv2: " + ProjectLink,
		Error:             "Error",
	},
	"zh_tw": {
		Header:            "完成",
		Status1:           "{loaded} 個字型載入成功，{errors} 個出錯，{no_match} 個無匹配。",
		Status2:           "索引中有 {index_fonts} 個字型，{index_names} 種名稱；當前共 {subtitles} 個字幕，{imported_fonts}個匯入字型。",
		BtnMenu:           "選單",
		BtnDetails:        "詳情",
		BtnClose:          "關閉",
		BtnChange:         "變更...",
		MenuUpdate:        "更新索引",
		MenuFontBase:      "設定字型庫",
		SelectFontBase:    "設定字型庫資料夾",
		MenuExport:        "匯出字型",
		MenuMKV:           "從 MKV 提取",
		MenuHelp:          "說明",
		MenuLang:          "語言",
		MenuClear:         "清除設定",
		TitleFontBase:     "字型庫路徑設定",
		LblCurrentPath:    "目前字型庫路徑:",
		MsgExport:         "匯出完成。",
		MsgHelp:           "1. 開啟 FontLoaderSubRe，設定字型库路徑，並從選單索引字型。（僅需首次執行）\n2. 將字幕 SSA/ASS 檔案或資料夾拖放至 FontLoaderSubRe 視窗。",
		MsgUpdateComplete: "索引更新完成。",
		MsgExtractingSubs: "從字幕中獲取字型名稱...",
		MsgClear:          "設定已清除",
		Footer:            "GPLv2: " + ProjectLink,
		Error:             "錯誤",
	},
	"zh_cn": {
		Header:            "完成",
		Status1:           "{loaded} 个字体加载成功，{errors} 个出错，{no_match} 个无匹配。",
		Status2:           "索引中有 {index_fonts} 个字体，{index_names} 种名称；当前共 {subtitles} 个字幕，{imported_fonts}个导入字体。",
		BtnMenu:           "菜单",
		BtnDetails:        "详情",
		BtnClose:          "关闭",
		BtnChange:         "更改...",
		MenuUpdate:        "更新索引",
		MenuFontBase:      "设置字体库",
		SelectFontBase:    "设置字体库文件夹",
		MenuExport:        "导出字体",
		MenuMKV:           "从MKV提取",
		MenuHelp:          "帮助",
		MenuLang:          "语言",
		MenuClear:         "清除设置",
		TitleFontBase:     "字体库路径设置",
		LblCurrentPath:    "当前字体库路径:",
		MsgExport:         "导出完成。",
		MsgHelp:           "1. 打开FontLoaderSubRe，设置字体库路径，从菜单中索引字体。（仅首次需要）\n2. 将字幕SSA/ASS文件或文件夹拖拽至 FontLoaderSubRe 窗口。",
		MsgUpdateComplete: "索引更新完成。",
		MsgExtractingSubs: "从字幕中获取字体名称...",
		MsgClear:          "设置已清除",
		Footer:            "GPLv2: " + ProjectLink,
		Error:             "错误",
	},
}

// Get returns the Strings for the given locale code.
// Falls back to "en_us" if the code is unknown.
func Get(code string) Strings {
	if s, ok := locales[code]; ok {
		return s
	}
	return locales["en_us"]
}
