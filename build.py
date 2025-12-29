import subprocess
import sys

NAME = "FontLoaderSubRe"
AUTHOR = "HighDoping"
DESCRIPTION = "Load missing fonts for ASS/SSA subtitles from huge font collections."
COPYRIGHT = "Copyright (c) 2025, HighDoping"
VERSION_BASE = "0.1.0"


nuitka_commands = [
    sys.executable,
    "-m",
    "nuitka",
    "--mode=app",
    "--enable-plugin=pyside6",
    "--product-name=" + DESCRIPTION,
    "--company-name=" + AUTHOR,
    "--file-version=" + VERSION_BASE,
    "--product-version=" + VERSION_BASE,
    "--copyright=" + COPYRIGHT,
    "--include-data-file=resources/icon.ico=resources/icon.ico",
    "--include-data-file=resources/icon.png=resources/icon.png",
    "--windows-icon-from-ico=resources/icon.ico",
    "--macos-app-icon=resources/icon.icns",
    "--windows-console-mode=attach",
    "--output-filename="+NAME,
    "main.py",
]
print(" ".join(nuitka_commands))
subprocess.run(nuitka_commands)
