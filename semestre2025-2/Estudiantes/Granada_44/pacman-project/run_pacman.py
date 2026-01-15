import os
import sys
import runpy

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PACMAN_DIR = os.path.join(PROJECT_ROOT, "pacman")

for path in [PROJECT_ROOT, PACMAN_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

os.chdir(PACMAN_DIR)

runpy.run_path("pacman.py", run_name="__main__")
    