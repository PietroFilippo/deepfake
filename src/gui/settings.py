"""
App settings, saved as JSON next to the app (settings.json).
"""
import json
import math
import os
from dataclasses import dataclass, asdict, fields

from ..utils import PROJECT_ROOT, MODELS_DIR, OUTPUTS_DIR

SETTINGS_PATH = os.path.join(PROJECT_ROOT, 'settings.json')

# Same as src.swapper (not imported, so the window can open before the heavy libraries are loaded)
PIXEL_BOOST_SIZES = (128, 256, 384, 512)
FACE_MASKS = ('box', 'parser')
FRINGE_MODES = ('blend', 'hide')

# Model files are recognized by name inside the models folder
SWAP_MODEL_KEYS = ('inswapper', 'hyperswap')
ENHANCER_MODEL_KEYS = ('gfpgan', 'gpen', 'restoreformer')
PARSER_MODEL_NAME = 'face_parser.onnx'
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.jfif', '.png', '.bmp', '.webp', '.tif', '.tiff')
RESOLUTIONS = ('640x480', '1280x720', '1920x1080')


@dataclass
class Settings:
    models_dir: str = MODELS_DIR
    faces_dir: str = os.path.join(PROJECT_ROOT, 'images')
    outputs_dir: str = OUTPUTS_DIR
    source_image: str = ''
    swap_model: str = 'inswapper_128_fp16.onnx'
    enhancer_model: str = 'GFPGANv1.4.onnx'
    enhance: bool = False
    enhance_blend: float = 0.8
    pixel_boost: int = 128
    file_pixel_boost: int = 512
    face_mask: str = 'box'
    fringe: str = 'blend'
    smoothing: bool = True
    detect_interval: int = 1
    det_size: int = 640
    camera_index: int = 0
    resolution: str = '1920x1080'
    camera_fps: int = 30
    mirror: bool = False
    flip_vertical: bool = False
    max_workers: int = 4
    virtual_cam: bool = False
    show_hud: bool = True

    @classmethod
    def load(cls):
        settings = cls()
        try:
            with open(SETTINGS_PATH, encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, ValueError):
            return settings
        if not isinstance(data, dict):
            return settings
        for field in fields(cls):
            value = data.get(field.name)
            default = getattr(settings, field.name)
            if isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
                try:
                    value = float(value)
                except OverflowError:
                    continue
            # Ignore missing or wrongly typed values (e.g. from a hand-edited file)
            if type(value) is type(default):
                setattr(settings, field.name, value)
        settings._validate()
        return settings

    def _validate(self):
        """Resets out-of-range values (e.g. from a hand-edited file) to their defaults."""
        defaults = Settings()
        valid = {
            'enhance_blend': lambda v: math.isfinite(v) and 0.0 <= v <= 1.0,
            'pixel_boost': lambda v: v in PIXEL_BOOST_SIZES,
            'file_pixel_boost': lambda v: v in PIXEL_BOOST_SIZES,
            'face_mask': lambda v: v in FACE_MASKS,
            'fringe': lambda v: v in FRINGE_MODES,
            'detect_interval': lambda v: 1 <= v <= 10,
            'det_size': lambda v: v in (320, 480, 640),
            'camera_index': lambda v: 0 <= v <= 9,
            'resolution': lambda v: v in RESOLUTIONS,
            'camera_fps': lambda v: v in (15, 24, 30, 60),
            'max_workers': lambda v: 1 <= v <= 8,
        }
        for name, check in valid.items():
            if not check(getattr(self, name)):
                setattr(self, name, getattr(defaults, name))

    def save(self):
        try:
            with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
                json.dump(asdict(self), f, indent=2)
        except OSError as e:
            print(f"Warning: could not save settings: {e}")

    # ------------------------------------------------------------------ helpers

    def model_file(self, name):
        return os.path.join(self.models_dir, name) if name else ''

    def resolution_size(self):
        width, height = self.resolution.split('x')
        return int(width), int(height)


def list_models(models_dir, keys):
    """ONNX files in `models_dir` whose name contains one of `keys` (case-insensitive), sorted."""
    try:
        names = os.listdir(models_dir)
    except OSError:
        return []
    return sorted(n for n in names if n.lower().endswith('.onnx') and any(k in n.lower() for k in keys))


def list_images(folder):
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return sorted(os.path.join(folder, n) for n in names if n.lower().endswith(IMAGE_EXTENSIONS))
