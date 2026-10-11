# PyInstaller runtime hook: runs before app.py in the packaged app.
import os

# albumentations (imported by insightface) otherwise checks pypi.org for updates on every start (2 s timeout)
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
