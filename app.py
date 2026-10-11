"""
Deepfake desktop app: live webcam face swap with a settings overlay (press Tab).
For the command line (and offline processing from scripts) use main.py.
"""
import os
import sys


def main():
    from src.utils import PROJECT_ROOT

    # pythonw.exe and the packaged app have no console: keep a log file for troubleshooting
    if sys.stdout is None or sys.stderr is None:
        log = open(os.path.join(PROJECT_ROOT, 'deepfake.log'), 'w', encoding='utf-8', buffering=1)
        sys.stdout = sys.stderr = log

    from src.gui.main_window import run
    return run()


if __name__ == "__main__":
    sys.exit(main())
