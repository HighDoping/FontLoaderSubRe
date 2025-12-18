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
    "--onefile",
    "--enable-plugin=tk-inter",
    "--product-name=" + DESCRIPTION,
    "--company-name=" + AUTHOR,
    "--file-version=" + VERSION_BASE,
    "--product-version=" + VERSION_BASE,
    "--copyright=" + COPYRIGHT,
    "--windows-console-mode=attach",
    "main.py",
]
print(" ".join(nuitka_commands))
subprocess.run(nuitka_commands)
