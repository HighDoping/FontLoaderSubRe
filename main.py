import builtins
import contextlib
import json
import logging
import platform
import shutil
import sys
import webbrowser
from multiprocessing import freeze_support
from pathlib import Path
from typing import Optional

from platformdirs import user_config_dir
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QCursor, QIcon

# PySide6 Imports
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from utils import (
    FontDatabase,
    FontMetadataExtractor,
    SessionFontManager,
    extract_ass_fonts,
    extract_mkv_resources,
    scan_fonts_in_directory,
)


class ExtractWorker(QThread):
    """Background thread for extracting font names from subtitle files."""

    progress = Signal(int, int)  # current, total
    finished = Signal(int, list, list)  # total_subs, all_fonts

    def __init__(self, file_list: list[Path]):
        super().__init__()
        self.file_list = file_list

    def run(self):
        all_fonts = set()
        all_font_paths = set()
        total_subs = 0
        total_files = len(self.file_list)

        for i, file_path in enumerate(self.file_list):
            ext = file_path.suffix.lower()

            if ext in {".ass", ".ssa"}:
                sub_count, font_list = extract_ass_fonts(file_path)
                all_fonts.update(font_list)
                total_subs += sub_count
            elif ext in {".ttf", ".otf", ".ttc"}:
                all_font_paths.add(file_path.absolute())
            elif ext == ".mkv":
                result = extract_mkv_resources(
                    file_path, extract_fonts=True, extract_subs=True
                )
                if not result:
                    continue
                temp_folder, files = result
                for file in files:
                    if not isinstance(file, Path):
                        continue
                    file_ext = file.suffix.lower()
                    if file_ext in {".ass", ".ssa"}:
                        sub_count, font_list = extract_ass_fonts(file)
                        all_fonts.update(font_list)
                        total_subs += sub_count
                    elif file_ext in {".ttf", ".otf", ".ttc"}:
                        all_font_paths.add((temp_folder / file).absolute())

            self.progress.emit(i + 1, total_files)

        self.finished.emit(total_subs, list(all_fonts), list(all_font_paths))


class ScanWorker(QThread):
    """Background thread for scanning fonts."""

    progress = Signal(int, int)  # current, total
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, db, font_path):
        super().__init__()
        self.db = db
        self.font_path = font_path

    def run(self):
        try:
            stats = scan_fonts_in_directory(
                self.db,
                Path(self.font_path),
                progress_callback=lambda current, total: self.progress.emit(
                    current, total
                ),
            )
            self.finished.emit(stats)
        except Exception as e:
            self.error.emit(str(e))


class LoadWorker(QThread):
    """Background thread for loading fonts into the session."""

    progress = Signal(int, str, str)  # index, font_name, log_line
    finished = Signal(list)

    def __init__(
        self,
        font_list: list[str],
        font_path_list: list[Path],
        font_manager: SessionFontManager,
        db: FontDatabase,
        font_base_path: Path,
    ):
        super().__init__()
        self.font_list = font_list
        self.font_path_list = font_path_list
        self.fm = font_manager
        self.db = db
        self.font_base = font_base_path

    def run(self):
        log_lines = []
        # load from paths first
        loaded_font_from_path = set()
        for i, font_path in enumerate(self.font_path_list):
            font_path = font_path.absolute()
            f_fullname = (
                FontMetadataExtractor()
                .db_decode_file(font_path)
                .get("full_names", [])[0]
            )
            f_hash = FontMetadataExtractor().db_decode_file(font_path).get("hash", "")
            success = self.fm.load_font(font_path, f_hash)
            if success:
                log_line = f"[ok] {f_fullname} > {str(font_path)}\n"
                loaded_font_from_path.add(f_fullname)
            else:
                msg = self.fm.status.get(f_hash, {}).get("message", "Error")
                log_line = f"[xx] {f_fullname} > {str(font_path)} ({msg})\n"
            log_lines.append(log_line)
            self.progress.emit(i + 1 + len(self.font_list), font_path.name, log_line)
        # then load from db
        for i, font in enumerate(self.font_list):
            # first check if already loaded
            log_line = ""
            if font in loaded_font_from_path:
                # log_line = f"[ok] {font} > Loaded from path\n"
                # log_lines.append(log_line)
                self.progress.emit(i + 1, font, log_line)
                continue
            res = self.db.search_by_font(font)
            log_line = ""
            if res:
                relative_path_str = res[0][0]
                f_path = (Path(self.font_base) / Path(relative_path_str)).absolute()
                f_hash = res[0][1]
                success = self.fm.load_font(f_path, f_hash)
                if success:
                    log_line = f"[ok] {font} > {relative_path_str}\n"
                else:
                    msg = self.fm.status.get(f_hash, {}).get("message", "Error")
                    log_line = f"[xx] {font} > {relative_path_str} ({msg})\n"
            else:
                log_line = f"[??] {font}\n"

            log_lines.append(log_line)
            self.progress.emit(i + 1, font, log_line)

        self.fm.cache_refresh()
        self.finished.emit(log_lines)


class SettingsDialog(QDialog):
    """Popup for changing font base path."""

    def __init__(self, parent, current_path, locale):
        super().__init__(parent)
        self.locales = locale
        self.setWindowTitle(locale["title_font_base"])
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(locale["lbl_current_path"]))
        self.entry = QLineEdit(str(current_path))
        self.entry.setReadOnly(True)
        layout.addWidget(self.entry)

        btn_layout = QHBoxLayout()
        self.btn_change = QPushButton(locale["btn_change"])
        self.btn_close = QPushButton(locale["btn_close"])
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_change)
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)

        self.btn_close.clicked.connect(self.accept)
        self.btn_change.clicked.connect(self.select_dir)
        self.result_path = current_path

    def select_dir(self):
        new_dir = QFileDialog.getExistingDirectory(
            self, self.locales["select_font_base"], str(self.result_path)
        )
        if new_dir:
            self.result_path = new_dir
            self.entry.setText(new_dir)
class QLogSignal(QObject):
    log = Signal(str)


class QLogHandler(logging.Handler):
    def __init__(self, emitter):
        super().__init__()
        self._emitter = emitter

    @property
    def emitter(self):
        return self._emitter

    def emit(self, record):
        msg = self.format(record)
        self.emitter.log.emit(msg)

class FontLoaderApp(QMainWindow):

    def __init__(self, sub_paths: Optional[list[Path]] = None):
        super().__init__()
        self.setAcceptDrops(True)

        q_log_signal = QLogSignal()
        h = QLogHandler(q_log_signal)
        h.setLevel(logging.ERROR)
        logging.getLogger().addHandler(h)
        q_log_signal.log.connect(self.error_pop_up)

        # Configuration & Managers
        self.app_config = AppConfig()
        try:
            self.db = FontDatabase(self.app_config.get_db_path())
        except Exception as e:
            logger.error("Database error: %s", e)
            QMessageBox.critical(
                self,
                "Database Error",
                f"Failed to open database:\n{e}\n Default to current folder.",
            )
            self.app_config.set_font_base_path(Path.cwd())
            self.db = FontDatabase(self.app_config.get_db_path())

        self.font_manager = SessionFontManager()
        self.project_link = "https://github.com/HighDoping/FontLoaderSubRe"

        # Localization
        self.locales = {
            "en_us": {
                "header": "Finished",
                "status_1": "{loaded} fonts loaded, {errors} errors, {no_match} unmatched.",
                "status_2": "Index: {index_fonts} fonts, {index_names} names; {subtitles} subtitles, {imported_fonts} imported fonts.",
                "btn_menu": "Menu",
                "btn_details": "Details",
                "btn_close": "Close",
                "btn_change": "Change...",
                "menu_update": "Update Index",
                "menu_font_base": "Set Font Base",
                "select_font_base": "Select Font Base Directory",
                "menu_export": "Export Fonts",
                "menu_mkv": "Extract from MKV",
                "menu_help": "Help",
                "menu_lang": "Language",
                "menu_clear": "Clear Settings",
                "title_font_base": "Font Base Settings",
                "lbl_current_path": "Current Base Path:",
                "msg_export": "Export complete.",
                "msg_help": "1. Open FontLoaderSubRe, set font directory path, index the fonts from menu. (Only needed first time)\n2. Drag-and-drop subtitles SSA/ASS files or folders onto the FontLoaderSubRe window.",
                "msg_update_complete": "Index update complete.",
                "msg_extracting_subs": "Extracting font names from subtitles...",
                "msg_clear": "Settings cleared",
                "footer": f"GPLv2: {self.project_link}",
                "error": "Error",
            },
            "zh_tw": {
                "header": "完成",
                "status_1": "{loaded} 個字型載入成功，{errors} 個出錯，{no_match} 個無匹配。",
                "status_2": "索引中有 {index_fonts} 個字型，{index_names} 種名稱；當前共 {subtitles} 個字幕，{imported_fonts}個匯入字型。",
                "btn_menu": "選單",
                "btn_details": "詳情",
                "btn_close": "關閉",
                "btn_change": "變更...",
                "menu_update": "更新索引",
                "menu_font_base": "設定字型庫",
                "select_font_base": "設定字型庫資料夾",
                "menu_export": "匯出字型",
                "menu_mkv": "從 MKV 提取",
                "menu_help": "說明",
                "menu_clear": "清除設定",
                "menu_lang": "語言",
                "title_font_base": "字型庫路徑設定",
                "lbl_current_path": "目前字型庫路徑:",
                "msg_export": "匯出完成。",
                "msg_help": "1. 開啟 FontLoaderSubRe，設定字型库路徑，並從選單索引字型。（僅需首次執行）\n2. 將字幕 SSA/ASS 檔案或資料夾拖放至 FontLoaderSubRe 視窗。",
                "msg_update_complete": "索引更新完成。",
                "msg_extracting_subs": "從字幕中獲取字型名稱...",
                "msg_clear": "設定已清除",
                "footer": f"GPLv2: {self.project_link}",
                "error": "錯誤",
            },
            "zh_cn": {
                "header": "完成",
                "status_1": "{loaded} 个字体加载成功，{errors} 个出错，{no_match} 个无匹配。",
                "status_2": "索引中有 {index_fonts} 个字体，{index_names} 种名称；当前共 {subtitles} 个字幕，{imported_fonts}个导入字体。",
                "btn_menu": "菜单",
                "btn_details": "详情",
                "btn_close": "关闭",
                "btn_change": "更改...",
                "menu_update": "更新索引",
                "menu_font_base": "设置字体库",
                "select_font_base": "设置字体库文件夹",
                "menu_export": "导出字体",
                "menu_mkv": "从MKV提取",
                "menu_help": "帮助",
                "menu_clear": "清除设置",
                "menu_lang": "语言",
                "title_font_base": "字体库路径设置",
                "lbl_current_path": "当前字体库路径:",
                "msg_export": "导出完成。",
                "msg_help": "1. 打开FontLoaderSubRe，设置字体库路径，从菜单中索引字体。（仅首次需要）\n2. 将字幕SSA/ASS文件或文件夹拖拽至 FontLoaderSubRe 窗口。",
                "msg_update_complete": "索引更新完成。",
                "msg_extracting_subs": "从字幕中获取字体名称...",
                "msg_clear": "设置已清除",
                "footer": f"GPLv2: {self.project_link}",
                "error": "错误",
            },
        }

        self.stats = {
            "loaded": 0,
            "errors": 0,
            "no_match": 0,
            "index_fonts": 0,
            "index_names": 0,
            "subtitles": 0,
            "imported_fonts": 0,
        }
        self.init_ui()
        self.refresh_ui()

        if sub_paths:
            QTimer.singleShot(100, lambda: self.process_dropped_files(sub_paths))

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QVBoxLayout(central)

        # Labels
        self.lbl_header = QLabel()
        self.lbl_header.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #0033CC;"
        )
        self.main_layout.addWidget(self.lbl_header)

        self.lbl_status1 = QLabel()
        self.lbl_status2 = QLabel()
        self.main_layout.addWidget(self.lbl_status1)
        self.main_layout.addWidget(self.lbl_status2)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        self.lbl_progress_task = QLabel()
        self.lbl_progress_task.hide()
        self.main_layout.addWidget(self.progress_bar)
        self.main_layout.addWidget(self.lbl_progress_task)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_menu = QPushButton()
        self.btn_details = QPushButton()
        self.btn_close = QPushButton()

        self.btn_menu.clicked.connect(self.show_popup_menu)
        self.btn_details.clicked.connect(self.toggle_details)
        self.btn_close.clicked.connect(self.close)

        btn_layout.addWidget(self.btn_menu)
        btn_layout.addWidget(self.btn_details)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_close)
        self.main_layout.addLayout(btn_layout)

        # Footer
        self.lbl_footer = QLabel()
        self.lbl_footer.setStyleSheet("color: blue; text-decoration: underline;")
        self.lbl_footer.setCursor(Qt.PointingHandCursor)
        self.lbl_footer.mousePressEvent = lambda e: webbrowser.open(self.project_link)
        self.main_layout.addWidget(self.lbl_footer)

        # Details Area
        self.txt_details = QPlainTextEdit()
        self.txt_details.setReadOnly(True)
        self.txt_details.hide()
        self.main_layout.addWidget(self.txt_details)

    def refresh_ui(self):
        txt = self.locales.get(self.app_config.get_language())

        self.lbl_header.setText(txt["header"])
        self.update_stats_from_db()

        self.lbl_status1.setText(txt["status_1"].format(**self.stats))
        self.lbl_status2.setText(txt["status_2"].format(**self.stats))
        self.btn_menu.setText(txt["btn_menu"])
        self.btn_details.setText(txt["btn_details"])
        self.btn_close.setText(txt["btn_close"])
        self.lbl_footer.setText(txt["footer"])

    def update_stats_from_db(self):
        self.stats["index_fonts"] = int(self.db.metadata_get("file_count") or 0)
        self.stats["index_names"] = sum(
            int(self.db.metadata_get(k) or 0)
            for k in [
                "ps_name_count",
                "family_name_count",
                "full_name_count",
                "unique_id_count",
            ]
        )

    def toggle_details(self):
        if self.txt_details.isVisible():
            self.hide_txt_details()
        else:
            self.txt_details.show()

    def show_popup_menu(self):
        menu = QMenu(self)
        txt = self.locales.get(self.app_config.get_language())

        act_update = menu.addAction(txt["menu_update"])
        act_update.triggered.connect(self.action_update_index)

        act_base = menu.addAction(txt["menu_font_base"])
        act_base.triggered.connect(self.action_set_font_base)

        act_export = menu.addAction(txt["menu_export"])
        act_export.triggered.connect(self.action_export)

        act_mkv_checkbox = menu.addAction(txt["menu_mkv"])
        act_mkv_checkbox.setCheckable(True)
        act_mkv_checkbox.setChecked(self.app_config.get("mkv_extraction", False))
        act_mkv_checkbox.triggered.connect(
            lambda: self.app_config.set("mkv_extraction", act_mkv_checkbox.isChecked())
        )

        menu.addSeparator()
        lang_menu = menu.addMenu(txt["menu_lang"])
        lang_menu.addAction("English").triggered.connect(lambda: self.set_lang("en_us"))
        lang_menu.addAction("正體中文").triggered.connect(
            lambda: self.set_lang("zh_tw")
        )
        lang_menu.addAction("简体中文").triggered.connect(
            lambda: self.set_lang("zh_cn")
        )

        menu.addSeparator()

        menu.addAction(txt["menu_help"]).triggered.connect(self.action_help)
        menu.addAction(txt["menu_clear"]).triggered.connect(self.action_clear)

        menu.exec(QCursor.pos())

    def set_lang(self, code):
        self.app_config.set_language(code)
        self.refresh_ui()

    def action_update_index(self):
        self.btn_menu.setEnabled(False)

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.lbl_progress_task.show()

        self.scan_thread = ScanWorker(self.db, self.app_config.get_font_base_path())

        self.scan_thread.progress.connect(self._on_scan_progress)

        self.scan_thread.finished.connect(self._on_scan_finished)
        txt = self.locales.get(self.app_config.get_language())
        self.scan_thread.error.connect(
            lambda e: QMessageBox.critical(self, txt["error"], e)
        )
        self.scan_thread.start()

    def _on_scan_progress(self, current, total):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        txt = self.locales.get(self.app_config.get_language())
        task_text = f"{txt['menu_update']}... ({current} / {total})"
        self.lbl_progress_task.setText(task_text)

    def _on_scan_finished(self, stats):
        self.hide_progress()
        self.btn_menu.setEnabled(True)
        self.refresh_ui()

        txt = self.locales.get(self.app_config.get_language())
        QMessageBox.information(self, txt["menu_update"], txt["msg_update_complete"])

    def action_set_font_base(self):
        txt = self.locales.get(self.app_config.get_language())
        dialog = SettingsDialog(self, self.app_config.get_font_base_path(), txt)
        if dialog.exec():
            new_path = Path(dialog.result_path)
            self.app_config.set_font_base_path(new_path)
            try:
                self.db = FontDatabase(self.app_config.get_db_path())
            except Exception as e:
                logger.error("Database error: %s", e)
                QMessageBox.critical(
                    self,
                    "Database Error",
                    f"Failed to open database:\n{e}\n Default to current folder.",
                )
                self.app_config.set_font_base_path(Path.cwd())
                self.db = FontDatabase(self.app_config.get_db_path())
            self.refresh_ui()

    def action_help(self):
        txt = self.locales.get(self.app_config.get_language())
        QMessageBox.information(self, txt["menu_help"], txt["msg_help"])

    def action_clear(self):
        self.app_config.clear()
        txt = self.locales.get(self.app_config.get_language())
        QMessageBox.information(self, txt["menu_clear"], txt["msg_clear"])
        self.refresh_ui()

    def action_export(self):
        target = QFileDialog.getExistingDirectory(self, "Export Fonts")
        if not target:
            return
        target_path = Path(target)
        for h, info in self.font_manager.status.items():
            if info["loaded"] and info["sys_path"]:
                with contextlib.suppress(builtins.BaseException):
                    shutil.copy(info["sys_path"], target_path / info["sys_path"].name)
        txt = self.locales.get(self.app_config.get_language())
        QMessageBox.information(self, txt["menu_export"], txt["msg_export"])

    # --- Drag & Drop ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [Path(u.toLocalFile()) for u in event.mimeData().urls()]
        if not files:
            return

        # Determine valid extensions based on mkv_extraction setting
        valid_exts = {".ass", ".ssa", ".ttf", ".otf", ".ttc"}
        if self.app_config.get("mkv_extraction", False):
            valid_exts.add(".mkv")

        # Collect all valid files from dropped items
        all_files = []
        for f in files:
            if f.is_dir():
                for ext in valid_exts:
                    all_files.extend(f.rglob(f"*{ext}"))
                    all_files.extend(f.rglob(f"*{ext.upper()}"))
            elif f.is_file() and f.suffix.lower() in valid_exts:
                all_files.append(f)

        if all_files:
            # Process the collected files
            logger.debug(f"Dropped files to process: {all_files}")
            self.process_dropped_files(all_files)

    def error_pop_up(self, msg):
        txt = self.locales.get(self.app_config.get_language())
        QMessageBox.critical(self, txt["error"], msg)

    def hide_progress(self):
        self.progress_bar.hide()
        self.lbl_progress_task.hide()
        if self.centralWidget().layout():
            self.centralWidget().layout().activate()
        self.adjustSize()

    def hide_txt_details(self):
        self.txt_details.hide()
        if self.centralWidget().layout():
            self.centralWidget().layout().activate()
        self.adjustSize()

    def process_dropped_files(self, file_list: list[Path]):
        """Process multiple subtitle files using a background worker."""
        if not file_list:
            return

        self.progress_bar.setRange(0, len(file_list))
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.lbl_progress_task.show()
        txt = self.locales.get(self.app_config.get_language())
        self.lbl_progress_task.setText(txt["msg_extracting_subs"])

        self.extract_thread = ExtractWorker(file_list)
        self.extract_thread.progress.connect(
            lambda current, total: self.progress_bar.setValue(current)
        )
        self.extract_thread.finished.connect(self._on_extract_finished)
        self.extract_thread.start()

    def _on_extract_finished(
        self, total_subs: int, font_list: list[str], font_file_list: list[Path]
    ):
        if not font_list and not font_file_list:
            self.hide_progress()
            return

        logger.debug("Extracted fonts: %s", font_list)
        logger.debug("Extracted font files: %s", font_file_list)

        self.stats["subtitles"] += total_subs
        self.stats["imported_fonts"] += len(font_file_list)

        # Transition to Loading phase
        self.progress_bar.setRange(0, len(font_list) + len(font_file_list))
        self.progress_bar.setValue(0)

        self.load_thread = LoadWorker(
            font_list,
            font_file_list,
            self.font_manager,
            self.db,
            self.app_config.get_font_base_path(),
        )
        self.load_thread.progress.connect(self._on_load_progress)
        self.load_thread.finished.connect(self._on_load_finished)
        self.load_thread.start()

    def _on_load_progress(self, idx, font_name, log_line):
        self.progress_bar.setValue(idx)
        self.lbl_progress_task.setText(f"Loading: {font_name}")

        if "[ok]" in log_line:
            self.stats["loaded"] += 1
        elif "[xx]" in log_line:
            self.stats["errors"] += 1
        elif "[??]" in log_line:
            self.stats["no_match"] += 1
        else:
            pass

    def _on_load_finished(self, log_lines):
        self.hide_progress()

        previously_loaded = self.txt_details.toPlainText().splitlines()
        all_lines = previously_loaded + [line.strip() for line in log_lines]

        # Remove duplicates while preserving order
        seen = set()
        unique_lines = []
        for line in all_lines:
            if line and line not in seen:
                seen.add(line)
                unique_lines.append(line)

        # Reorder log lines by status: [ok], [xx], [??]
        ok_lines = [line for line in unique_lines if line.startswith("[ok]")]
        error_lines = [line for line in unique_lines if line.startswith("[xx]")]
        no_match_lines = [line for line in unique_lines if line.startswith("[??]")]

        # Recalculate stats based on unique lines
        self.stats["loaded"] = len(ok_lines)
        self.stats["errors"] = len(error_lines)
        self.stats["no_match"] = len(no_match_lines)

        lines = ""
        for line in ok_lines + error_lines + no_match_lines:
            lines += line.strip() + "\n"
        self.txt_details.setPlainText(lines)

        self.refresh_ui()

    def closeEvent(self, event):
        self.font_manager.cleanup()
        event.accept()


class AppConfig:
    """Unified interface for application configuration management."""

    APP_NAME = "FontLoaderSubRe"
    APP_AUTHOR = "HighDoping"

    DB_NAME = "FontLoaderSubRe.db"

    DEFAULT_SETTINGS = {
        "font_base_path": str(Path.cwd()),
        "ui_language": "en_us",
        "mkv_extraction": False,
    }

    def __init__(self):

        self.config_dir = Path(user_config_dir(self.APP_NAME, self.APP_AUTHOR))
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "settings.json"
        self._settings = self._load_settings()

    def _load_settings(self) -> dict:
        """Load settings from file or create defaults."""
        if self.config_file.exists():
            try:
                with self.config_file.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                logger.warning("Failed to load settings, using defaults")

        return self.DEFAULT_SETTINGS.copy()

    def _save_settings(self):
        """Persist settings to disk."""
        try:
            with self.config_file.open("w", encoding="utf-8") as f:
                json.dump(self._settings, f, indent=4)
        except Exception as e:
            logger.exception(f"Failed to save settings: {e}")

    def get(self, key: str, default=None):
        """Get a configuration value."""
        return self._settings.get(key, default)

    def set(self, key: str, value):
        """Set a configuration value and save."""
        self._settings[key] = value
        self._save_settings()

    def get_font_base_path(self) -> Path:
        """Get the font base directory."""
        return Path(self._settings.get("font_base_path", Path.cwd()))

    def set_font_base_path(self, path: Path):
        """Set the font base directory."""
        self.set("font_base_path", str(path))

    def get_db_path(self) -> Path:
        """Get the database file path."""
        return self.get_font_base_path() / self.DB_NAME

    def get_language(self) -> str:
        """Get the UI language."""
        return self._settings.get("ui_language", "en_us")

    def set_language(self, lang_code: str):
        """Set the UI language."""
        self.set("ui_language", lang_code)

    def clear(self):
        """Clear all settings"""
        self._settings = self.DEFAULT_SETTINGS.copy()
        self._save_settings()


if __name__ == "__main__":
    freeze_support()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    app = QApplication(sys.argv)

    app.setApplicationName("FontLoaderSubRe")
    app.setDesktopFileName("FontLoaderSubRe")

    sub_paths = [Path(arg) for arg in sys.argv[1:]] if len(sys.argv) > 1 else []
    window = FontLoaderApp(sub_paths=sub_paths)
    window.setWindowTitle("FontLoaderSubRe 0.2.0")

    # add icon
    if platform.system() == "Darwin":
        icns_path = Path(__file__).parent / "resources" / "icon.icns"
        if icns_path.exists():
            icon = QIcon(str(icns_path))
            window.setWindowIcon(icon)
            # Set dock icon on macOS
            app.setWindowIcon(icon)
    else:
        icon_path = Path(__file__).parent / "resources" / "icon.png"
        if icon_path.exists():
            icon = QIcon(str(icon_path))
            window.setWindowIcon(icon)
            app.setWindowIcon(icon)

    window.show()

    sys.exit(app.exec())
