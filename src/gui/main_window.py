"""
Main window: live preview with a HUD, and a settings overlay toggled with Tab (or F1).
"""
import os
import subprocess
import sys
import tempfile
import threading
import time

from PySide6.QtCore import QFileSystemWatcher, QObject, QPointF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QImageReader, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame, QGroupBox,
                               QHBoxLayout, QLabel, QListView, QListWidget, QListWidgetItem, QMessageBox,
                               QProgressBar, QPushButton, QScrollArea, QSlider, QSpinBox, QVBoxLayout, QWidget)

# The face swap modules (ONNX Runtime, insightface, ...) are imported in the background, so the window
# opens right away with a loading screen
from .pipeline import LivePipeline
from .settings import (ENHANCER_MODEL_KEYS, FACE_MASKS, FRINGE_MODES, IMAGE_EXTENSIONS, PARSER_MODEL_NAME,
                       PIXEL_BOOST_SIZES, RESOLUTIONS, SWAP_MODEL_KEYS, Settings, list_images, list_models)

FRINGE_LABELS = {'blend': "Model default", 'hide': "Hide (keep your forehead)",
                 'show': "Show (the photo's fringe)"}

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v')
PANEL_WIDTH = 420

PANEL_STYLE = """
#panel { background-color: rgba(18, 18, 22, 228); border-right: 1px solid #333; }
#panel QWidget { color: #e6e6e6; font-size: 13px; }
#panel QScrollArea, #panel #content { background: transparent; }
#panel QGroupBox { border: 1px solid #3a3a42; border-radius: 6px; margin-top: 14px; padding: 8px 6px 6px 6px; }
#panel QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #9ecbff; font-weight: bold; }
#panel QComboBox, #panel QSpinBox, #panel QListWidget {
    background: #26262c; border: 1px solid #44444c; border-radius: 4px; padding: 3px; }
#panel QComboBox QAbstractItemView { background: #26262c; selection-background-color: #3d6fb6; }
#panel QPushButton { background: #2f3a4a; border: 1px solid #46566e; border-radius: 4px; padding: 5px 10px; }
#panel QPushButton:hover { background: #3b4a5f; }
#panel QPushButton:disabled, #panel QComboBox:disabled { color: #777; }
#panel QListWidget::item:selected { background: #3d6fb6; border-radius: 4px; }
#panel QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #6a6a75; border-radius: 3px; background: #1d1d22; }
#panel QCheckBox::indicator:checked { background: #3d8ef0; border-color: #3d8ef0; image: url(CHECK_ICON); }
#panel QCheckBox::indicator:disabled { border-color: #444; background: #2a2a2e; }
#panel QProgressBar { border: 1px solid #44444c; border-radius: 4px; text-align: center; background: #26262c; }
#panel QProgressBar::chunk { background: #3d6fb6; }
#panel #title { font-size: 16px; font-weight: bold; color: #ffffff; }
#panel #hint { color: #9a9aa5; font-size: 12px; }
"""


def _panel_style():
    """PANEL_STYLE with a white tick image for checked checkboxes (stylesheets need an image file)."""
    path = os.path.join(tempfile.gettempdir(), 'deepfake_check.png')
    if not os.path.exists(path):
        image = QImage(28, 28, QImage.Format_ARGB32)
        image.fill(Qt.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor('white'), 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPolyline([QPointF(6, 15), QPointF(12, 21), QPointF(22, 8)])
        painter.end()
        image.save(path)
    return PANEL_STYLE.replace('CHECK_ICON', path.replace('\\', '/'))


class Bridge(QObject):
    """Signals used to hand results from background threads to the GUI thread."""
    status = Signal(str)
    swapper_ready = Signal(object)
    swapper_failed = Signal(str)
    swap_model_loaded = Signal(str, bool, str)
    enhancer_loaded = Signal(str, bool, str)
    source_set = Signal(str, bool, str)
    file_progress = Signal(int, int)
    file_done = Signal(str)
    recording_stopped = Signal(int)
    shutdown_done = Signal()


class VideoView(QWidget):
    """Draws the latest frame (letterboxed) with the HUD on top. The HUD is never part of the frame."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame = None
        self.hud_lines = []
        self.show_hud = True
        self.recording = False
        self.status = ""
        self.hud_left = 12
        # Shown while there is no video yet: (title, text, show spinner)
        self.placeholder = ("Deepfake", "Starting...", True)
        self.placeholder_since = time.time()
        self.setAttribute(Qt.WA_OpaquePaintEvent)

    def set_placeholder(self, title, text, spinner=True):
        if self.placeholder is None or self.placeholder[0] != title:
            self.placeholder_since = time.time()
        self.placeholder = (title, text, spinner)
        self.update()

    def set_frame(self, frame):
        self.frame = frame
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self.frame is not None:
            h, w = self.frame.shape[:2]
            image = QImage(self.frame.data, w, h, self.frame.strides[0], QImage.Format_BGR888)
            scale = min(self.width() / w, self.height() / h)
            tw, th = int(w * scale), int(h * scale)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.drawImage(self.rect().adjusted((self.width() - tw) // 2, (self.height() - th) // 2,
                                                   -((self.width() - tw + 1) // 2), -((self.height() - th + 1) // 2)),
                              image)
        elif self.placeholder:
            self._paint_placeholder(painter)

        font = QFont(self.font())
        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)
        line_h = painter.fontMetrics().height() + 4
        if self.show_hud:
            y = 12
            for text, color in self.hud_lines:
                self._draw_text(painter, self.hud_left, y + line_h - 6, text, color)
                y += line_h
        if self.recording:
            x = self.width() - 90
            painter.setBrush(QColor(230, 30, 30))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(x, 16, 14, 14)
            self._draw_text(painter, x + 22, 29, "REC", QColor(230, 30, 30))
        if self.status:
            self._draw_text(painter, self.hud_left, self.height() - 14, self.status, QColor(235, 235, 235))
        painter.end()

    def _paint_placeholder(self, painter):
        """Loading screen: title, current step, spinner and elapsed time, centered on the free area."""
        title, text, spinner = self.placeholder
        left = self.hud_left - 12
        cx, cy = left + (self.width() - left) // 2, self.height() // 2
        painter.setRenderHint(QPainter.Antialiasing)

        font = QFont(self.font())
        font.setPointSize(22)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(235, 235, 240))
        painter.drawText(left, cy - 110, self.width() - left, 40, Qt.AlignHCenter, title)

        if spinner:
            # Rotating arc, 1 turn per second
            angle = int((time.time() % 1.0) * 360 * 16)
            painter.setPen(QPen(QColor(60, 60, 70), 5))
            painter.drawEllipse(cx - 22, cy - 50, 44, 44)
            painter.setPen(QPen(QColor(61, 142, 240), 5, Qt.SolidLine, Qt.RoundCap))
            painter.drawArc(cx - 22, cy - 50, 44, 44, -angle, 100 * 16)

        font.setPointSize(11)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor(200, 200, 210))
        painter.drawText(left + 40, cy + 10, self.width() - left - 80, 60,
                         Qt.AlignHCenter | Qt.TextWordWrap, text)
        if spinner:
            painter.setPen(QColor(140, 140, 150))
            painter.drawText(left, cy + 72, self.width() - left, 24, Qt.AlignHCenter,
                             f"{int(time.time() - self.placeholder_since)} s")

    @staticmethod
    def _draw_text(painter, x, y, text, color):
        painter.setPen(QColor(0, 0, 0, 200))
        painter.drawText(x + 1, y + 1, text)
        painter.setPen(color)
        painter.drawText(x, y, text)


class MainWindow(QWidget):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.swapper = None
        self.pipeline = None
        self.bridge = Bridge()
        self._loading_model = False
        self._loading_enhancer = False
        self._file_job = None
        self._cancel_file = False
        self._recording = False
        self._recording_session = 0
        self._closing = False
        self._closed = False
        self._source_lock = threading.Lock()
        self._source_generation = 0
        self._last_frame_number = 0
        self._status_time = 0.0

        self.setWindowTitle("Deepfake")
        self.resize(1280, 760)
        self.video = VideoView(self)
        self.panel = self._build_panel()
        self.panel.hide()

        self.bridge.status.connect(self._set_status)
        self.bridge.swapper_ready.connect(self._on_swapper_ready)
        self.bridge.swapper_failed.connect(self._on_swapper_failed)
        self.bridge.swap_model_loaded.connect(self._on_swap_model_loaded)
        self.bridge.enhancer_loaded.connect(self._on_enhancer_loaded)
        self.bridge.source_set.connect(self._on_source_set)
        self.bridge.file_progress.connect(self._on_file_progress)
        self.bridge.file_done.connect(self._on_file_done)
        self.bridge.recording_stopped.connect(self._on_recording_stopped)
        self.bridge.shutdown_done.connect(self._on_shutdown_done)

        self._add_shortcuts()
        # The faces list follows the folder (files added or removed while the app runs)
        self.faces_watcher = QFileSystemWatcher(self)
        self.faces_watcher.directoryChanged.connect(lambda _: self._faces_refresh.start())
        self._faces_refresh = QTimer(self, singleShot=True, interval=500)
        self._faces_refresh.timeout.connect(self._load_faces)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(15)

    # ------------------------------------------------------------------ startup

    def start(self):
        if not self._ensure_models_dir():
            return False
        self._populate_controls()
        self._set_controls_enabled(False)
        self.video.set_placeholder("Loading", "Loading libraries...")
        threading.Thread(target=self._create_swapper, daemon=True).start()
        return True

    def _ensure_models_dir(self):
        s = self.settings
        while not list_models(s.models_dir, SWAP_MODEL_KEYS):
            QMessageBox.information(
                self, "Select your models folder",
                "Choose the folder that contains your models, e.g. inswapper_128_fp16.onnx or "
                "hyperswap_1a_256.onnx (plus optional GFPGANv1.4.onnx, GPEN-BFR-512.onnx and face_parser.onnx).")
            folder = QFileDialog.getExistingDirectory(self, "Models folder", s.models_dir)
            if not folder:
                return False
            s.models_dir = folder
        swap_models = list_models(s.models_dir, SWAP_MODEL_KEYS)
        if s.swap_model not in swap_models:
            s.swap_model = swap_models[0]
        enhancers = list_models(s.models_dir, ENHANCER_MODEL_KEYS)
        if s.enhancer_model not in enhancers:
            s.enhancer_model = enhancers[0] if enhancers else ''
        s.save()
        return True

    def _create_swapper(self):
        s = self.settings
        try:
            from ..swapper import FaceSwapper
            parser_path = s.model_file(PARSER_MODEL_NAME)
            swapper = FaceSwapper(
                s.model_file(s.swap_model), det_size=(s.det_size, s.det_size), max_workers=s.max_workers,
                pixel_boost=s.pixel_boost, face_mask=s.face_mask,
                enhancer_model=s.model_file(s.enhancer_model),
                enhance_blend=s.enhance_blend, smoothing=s.smoothing, fringe=s.fringe,
                parser_model=parser_path if os.path.exists(parser_path) else None,
                on_status=self.bridge.status.emit)
            swapper.enhancement_enabled = s.enhance and swapper.enhancer is not None
            source = s.source_image if os.path.isfile(s.source_image) else next(iter(list_images(s.faces_dir)), '')
            if source:
                try:
                    swapper.set_source_image(source)
                    s.source_image = source
                except Exception as e:
                    self.bridge.status.emit(f"Source face: {e}")
            self.bridge.swapper_ready.emit(swapper)
        except Exception as e:
            self.bridge.swapper_failed.emit(str(e))

    def _on_swapper_ready(self, swapper):
        if self._closing:
            return
        self.swapper = swapper
        s = self.settings
        self.pipeline = LivePipeline(swapper, on_status=self.bridge.status.emit,
                                     on_recording_stopped=self.bridge.recording_stopped.emit)
        self.pipeline.detect_interval = s.detect_interval
        self.pipeline.mirror, self.pipeline.flip_vertical = s.mirror, s.flip_vertical
        self.pipeline.start()
        self.pipeline.open_camera(s.camera_index, *s.resolution_size(), s.camera_fps)
        if s.virtual_cam:
            self.pipeline.set_virtual_cam(True, s.camera_fps)
        self._set_controls_enabled(True)
        self.video.set_placeholder("Starting the camera", f"Camera {s.camera_index} ({s.resolution})")
        if not swapper.source_face:
            self._set_status("Pick a source face in the settings (Tab)")
            self.toggle_panel(True)

    def _on_swapper_failed(self, message):
        if self._closing:
            return
        self.video.set_placeholder("Could not load the models", message, spinner=False)
        QMessageBox.critical(self, "Could not load the models", message)
        self._set_status(f"Error: {message}. Choose another models folder in the settings.")
        self.models_button.setEnabled(True)
        self.toggle_panel(True)

    # ------------------------------------------------------------------ panel

    def _build_panel(self):
        panel = QFrame(self)
        panel.setObjectName("panel")
        panel.setStyleSheet(_panel_style())
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)
        content = QWidget()
        content.setObjectName("content")
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel("Settings")
        title.setObjectName("title")
        hint = QLabel("Tab / F1: show or hide · Esc: close · F11: fullscreen")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(hint)

        # Source face
        group = QGroupBox("Source face  (N / P)")
        box = QVBoxLayout(group)
        self.faces_list = QListWidget()
        self.faces_list.setViewMode(QListView.IconMode)
        self.faces_list.setIconSize(QSize(64, 64))
        self.faces_list.setGridSize(QSize(82, 92))
        self.faces_list.setResizeMode(QListView.Adjust)
        self.faces_list.setMovement(QListView.Static)
        self.faces_list.setFixedHeight(200)
        self.faces_list.itemClicked.connect(lambda item: self.choose_source(item.data(Qt.UserRole)))
        box.addWidget(self.faces_list)
        row = QHBoxLayout()
        self.faces_dir_button = QPushButton("Faces folder...")
        self.faces_dir_button.clicked.connect(self._change_faces_dir)
        add_button = QPushButton("Other image...")
        add_button.clicked.connect(self._choose_source_file)
        row.addWidget(self.faces_dir_button)
        row.addWidget(add_button)
        box.addLayout(row)
        layout.addWidget(group)

        # Face swap
        group = QGroupBox("Face swap")
        form = QFormLayout(group)
        self.swap_check = QCheckBox("Swap enabled  (X)")
        self.swap_check.setChecked(True)
        self.swap_check.toggled.connect(self._set_swap_enabled)
        form.addRow(self.swap_check)
        self.swap_combo = QComboBox()
        self.swap_combo.activated.connect(lambda _: self._change_swap_model(self.swap_combo.currentText()))
        form.addRow("Model", self.swap_combo)
        self.boost_combo = QComboBox()
        self.boost_combo.activated.connect(lambda _: self._set_pixel_boost(int(self.boost_combo.currentText())))
        form.addRow("Pixel boost (live)  (B)", self.boost_combo)
        self.file_boost_combo = QComboBox()
        self.file_boost_combo.addItems([str(s) for s in PIXEL_BOOST_SIZES])
        self.file_boost_combo.activated.connect(
            lambda _: self._update_setting('file_pixel_boost', int(self.file_boost_combo.currentText())))
        form.addRow("Pixel boost (files)", self.file_boost_combo)
        self.mask_combo = QComboBox()
        self.mask_combo.addItems(FACE_MASKS)
        self.mask_combo.activated.connect(lambda _: self._set_face_mask(self.mask_combo.currentText()))
        form.addRow("Face mask  (M)", self.mask_combo)
        self.fringe_combo = QComboBox()
        for mode in FRINGE_MODES:
            self.fringe_combo.addItem(FRINGE_LABELS[mode], mode)
        self.fringe_combo.setToolTip("Hair over the forehead (e.g. a fringe in the source photo) only half comes "
                                     "through the swap and looks see-through.\nHide: keep your own forehead and "
                                     "eyebrows.\nShow: give you the photo's fringe; the swap model draws it in "
                                     "your lighting, starting at your hairline.")
        self.fringe_combo.activated.connect(lambda _: self._set_fringe(self.fringe_combo.currentData()))
        form.addRow("Fringe  (F)", self.fringe_combo)
        layout.addWidget(group)

        # Enhancer
        group = QGroupBox("Face enhancer")
        form = QFormLayout(group)
        self.enhance_check = QCheckBox("Enhancer enabled  (E)")
        self.enhance_check.toggled.connect(self._set_enhance)
        form.addRow(self.enhance_check)
        self.enhancer_combo = QComboBox()
        self.enhancer_combo.activated.connect(lambda _: self._change_enhancer(self.enhancer_combo.currentText()))
        form.addRow("Model", self.enhancer_combo)
        blend_row = QHBoxLayout()
        self.blend_slider = QSlider(Qt.Horizontal)
        self.blend_slider.setRange(0, 100)
        self.blend_label = QLabel()
        self.blend_label.setMinimumWidth(36)
        self.blend_slider.valueChanged.connect(self._set_blend)
        blend_row.addWidget(self.blend_slider)
        blend_row.addWidget(self.blend_label)
        form.addRow("Blend", blend_row)
        layout.addWidget(group)

        # Tracking
        group = QGroupBox("Face detection")
        form = QFormLayout(group)
        self.smooth_check = QCheckBox("Smooth landmarks (less jitter)")
        self.smooth_check.toggled.connect(self._set_smoothing)
        form.addRow(self.smooth_check)
        self.det_combo = QComboBox()
        self.det_combo.addItems(["320", "480", "640"])
        self.det_combo.activated.connect(lambda _: self._set_det_size(int(self.det_combo.currentText())))
        form.addRow("Detector size", self.det_combo)
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 10)
        self.interval_spin.valueChanged.connect(self._set_detect_interval)
        form.addRow("Detect every N frames", self.interval_spin)
        layout.addWidget(group)

        # Camera
        group = QGroupBox("Camera")
        form = QFormLayout(group)
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 9)
        form.addRow("Camera index", self.camera_spin)
        self.res_combo = QComboBox()
        self.res_combo.addItems(RESOLUTIONS)
        form.addRow("Resolution", self.res_combo)
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["15", "24", "30", "60"])
        form.addRow("FPS", self.fps_combo)
        apply_camera = QPushButton("Apply camera settings")
        apply_camera.clicked.connect(self._apply_camera)
        form.addRow(apply_camera)
        self.vcam_check = QCheckBox("Virtual camera (OBS)")
        self.vcam_check.setToolTip("Send the output to OBS Virtual Camera, for Discord, Zoom, Teams, ...")
        self.vcam_check.toggled.connect(self._set_virtual_cam)
        form.addRow(self.vcam_check)
        self.mirror_check = QCheckBox("Mirror (flip left-right)")
        self.mirror_check.toggled.connect(lambda on: self._set_flip('mirror', on))
        form.addRow(self.mirror_check)
        self.flip_check = QCheckBox("Flip upside down")
        self.flip_check.toggled.connect(lambda on: self._set_flip('flip_vertical', on))
        form.addRow(self.flip_check)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 8)
        self.workers_spin.valueChanged.connect(self._set_workers)
        form.addRow("Worker threads", self.workers_spin)
        layout.addWidget(group)

        # Recording and files
        group = QGroupBox("Recording and files")
        box = QVBoxLayout(group)
        self.record_button = QPushButton("Start recording  (R)")
        self.record_button.clicked.connect(self.toggle_recording)
        box.addWidget(self.record_button)
        self.process_button = QPushButton("Process an image, video or GIF...")
        self.process_button.clicked.connect(self._process_file)
        box.addWidget(self.process_button)
        progress_row = QHBoxLayout()
        self.file_progress = QProgressBar()
        self.file_progress.hide()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.hide()
        self.cancel_button.clicked.connect(self._cancel_file_job)
        progress_row.addWidget(self.file_progress)
        progress_row.addWidget(self.cancel_button)
        box.addLayout(progress_row)
        open_outputs = QPushButton("Open outputs folder")
        open_outputs.clicked.connect(lambda: self._open_folder(self.settings.outputs_dir))
        box.addWidget(open_outputs)
        layout.addWidget(group)

        # Folders and view
        group = QGroupBox("Folders and view")
        form = QFormLayout(group)
        self.models_dir_label = QLabel()
        self.models_dir_label.setWordWrap(True)
        self.models_button = QPushButton("Change models folder...")
        self.models_button.clicked.connect(self._change_models_dir)
        form.addRow(self.models_dir_label)
        form.addRow(self.models_button)
        self.outputs_dir_label = QLabel()
        self.outputs_dir_label.setWordWrap(True)
        outputs_button = QPushButton("Change outputs folder...")
        outputs_button.clicked.connect(self._change_outputs_dir)
        form.addRow(self.outputs_dir_label)
        form.addRow(outputs_button)
        self.hud_check = QCheckBox("Show HUD  (U)")
        self.hud_check.toggled.connect(self._set_hud)
        form.addRow(self.hud_check)
        layout.addWidget(group)
        layout.addStretch(1)

        # Long labels and model names must not widen the panel: wrap form rows when needed
        for form in content.findChildren(QFormLayout):
            form.setRowWrapPolicy(QFormLayout.WrapLongRows)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        for combo in content.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
        self.faces_list.setMinimumWidth(0)

        self._controls = [self.faces_list, self.faces_dir_button, add_button, self.swap_check, self.swap_combo,
                          self.boost_combo, self.file_boost_combo, self.mask_combo, self.fringe_combo,
                          self.enhance_check, self.enhancer_combo, self.blend_slider, self.smooth_check,
                          self.det_combo, self.interval_spin, apply_camera, self.vcam_check, self.mirror_check,
                          self.flip_check, self.workers_spin, self.record_button,
                          self.process_button, self.models_button]
        return panel

    def _populate_controls(self):
        """Fills the panel from the settings and the loaded models (signals are blocked meanwhile)."""
        s = self.settings
        widgets = [self.swap_combo, self.boost_combo, self.file_boost_combo, self.mask_combo, self.fringe_combo,
                   self.enhance_check, self.enhancer_combo, self.blend_slider, self.smooth_check, self.det_combo,
                   self.interval_spin, self.camera_spin, self.res_combo, self.fps_combo, self.vcam_check,
                   self.mirror_check, self.flip_check, self.workers_spin, self.hud_check, self.swap_check]
        for w in widgets:
            w.blockSignals(True)

        self.swap_combo.clear()
        self.swap_combo.addItems(list_models(s.models_dir, SWAP_MODEL_KEYS))
        current = self.swapper.model.name if self.swapper else s.swap_model
        self.swap_combo.setCurrentText(current)
        self._refresh_boost_combo()
        self.file_boost_combo.setCurrentText(str(s.file_pixel_boost))
        self.mask_combo.setCurrentText(s.face_mask)
        parser_ok = self.swapper is None or self.swapper.parser is not None
        self.mask_combo.model().item(FACE_MASKS.index('parser')).setEnabled(parser_ok)
        self.fringe_combo.setCurrentIndex(FRINGE_MODES.index(s.fringe))
        self.fringe_combo.model().item(FRINGE_MODES.index('show')).setEnabled(parser_ok)

        enhancers = list_models(s.models_dir, ENHANCER_MODEL_KEYS)
        self.enhancer_combo.clear()
        self.enhancer_combo.addItems(enhancers)
        self.enhancer_combo.setCurrentText(s.enhancer_model)
        enhancer_ok = bool(enhancers) and (self.swapper is None or self.swapper.enhancer is not None)
        self.enhance_check.setEnabled(enhancer_ok)
        self.enhance_check.setChecked(s.enhance and enhancer_ok)
        self.blend_slider.setValue(int(round(s.enhance_blend * 100)))
        self.blend_label.setText(f"{self.blend_slider.value()}%")

        self.smooth_check.setChecked(s.smoothing)
        self.det_combo.setCurrentText(str(s.det_size))
        self.interval_spin.setValue(s.detect_interval)
        self.camera_spin.setValue(s.camera_index)
        self.res_combo.setCurrentText(s.resolution)
        self.fps_combo.setCurrentText(str(s.camera_fps))
        self.vcam_check.setChecked(s.virtual_cam)
        self.mirror_check.setChecked(s.mirror)
        self.flip_check.setChecked(s.flip_vertical)
        self.workers_spin.setValue(s.max_workers)
        self.hud_check.setChecked(s.show_hud)
        self.video.show_hud = s.show_hud
        self.swap_check.setChecked(self.pipeline.swap_enabled if self.pipeline else True)

        for w in widgets:
            w.blockSignals(False)

        self.models_dir_label.setText(f"Models: {s.models_dir}")
        self.outputs_dir_label.setText(f"Outputs: {s.outputs_dir}")
        self._load_faces()

    def _refresh_boost_combo(self):
        self.boost_combo.blockSignals(True)
        self.boost_combo.clear()
        if self.swapper:
            self.boost_combo.addItems([str(s) for s in self.swapper.boost_sizes])
            self.boost_combo.setCurrentText(str(self.swapper.pixel_boost))
        self.boost_combo.blockSignals(False)

    def _load_faces(self):
        watched = self.faces_watcher.directories()
        if watched != [self.settings.faces_dir]:
            if watched:
                self.faces_watcher.removePaths(watched)
            if os.path.isdir(self.settings.faces_dir):
                self.faces_watcher.addPath(self.settings.faces_dir)
        scroll = self.faces_list.verticalScrollBar().value()
        self.faces_list.clear()
        for path in list_images(self.settings.faces_dir):
            reader = QImageReader(path)
            reader.setAutoTransform(True)
            size = reader.size()
            if size.isValid():
                reader.setScaledSize(size.scaled(QSize(96, 96), Qt.KeepAspectRatio))
            icon = QIcon(QPixmap.fromImage(reader.read()))
            name = os.path.basename(path)
            item = QListWidgetItem(icon, self.faces_list.fontMetrics().elidedText(name, Qt.ElideMiddle, 76))
            item.setToolTip(name)
            item.setData(Qt.UserRole, path)
            self.faces_list.addItem(item)
            if os.path.normcase(path) == os.path.normcase(self.settings.source_image):
                item.setSelected(True)
                self.faces_list.setCurrentItem(item)
        self.faces_list.verticalScrollBar().setValue(scroll)

    def _set_controls_enabled(self, enabled):
        for w in self._controls:
            w.setEnabled(enabled)
        if enabled:
            self._populate_controls()

    def toggle_panel(self, visible=None):
        visible = not self.panel.isVisible() if visible is None else visible
        self.panel.setVisible(visible)
        self.video.hud_left = self.panel.width() + 12 if visible else 12
        if visible:
            self.panel.raise_()
        else:
            self.setFocus()
        self.video.update()

    def resizeEvent(self, event):
        self.video.setGeometry(self.rect())
        self.panel.setGeometry(0, 0, min(PANEL_WIDTH, self.width()), self.height())
        super().resizeEvent(event)

    # ------------------------------------------------------------------ actions

    def _update_setting(self, name, value):
        setattr(self.settings, name, value)
        self.settings.save()

    def _set_status(self, text):
        print(text)
        if self.swapper is None and self.video.placeholder and self.video.placeholder[2]:
            self.video.set_placeholder("Loading", text)  # the loading screen shows the current step
        self.video.status = text
        self._status_time = time.time()
        self.video.update()

    def choose_source(self, path):
        if not self.swapper or not path:
            return
        self._set_status(f"Setting source face: {os.path.basename(path)}")
        self._source_generation += 1
        generation = self._source_generation

        def work():
            # One at a time, and only the newest choice: a slow earlier pick can't override a later one
            with self._source_lock:
                if generation != self._source_generation:
                    return
                try:
                    self.swapper.set_source_image(path)
                    self.bridge.source_set.emit(path, True, f"Source face: {os.path.basename(path)}")
                except Exception as e:
                    self.bridge.source_set.emit(path, False, f"{os.path.basename(path)}: {e}")
        threading.Thread(target=work, daemon=True).start()

    def _on_source_set(self, path, ok, message):
        self._set_status(message)
        if ok:
            self._update_setting('source_image', path)

    def _choose_source_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Source face image", self.settings.faces_dir,
                                              "Images (" + " ".join("*" + e for e in IMAGE_EXTENSIONS) + ")")
        if path:
            self.choose_source(os.path.normpath(path))

    def step_source(self, step):
        count = self.faces_list.count()
        if not count:
            return
        row = (self.faces_list.currentRow() + step) % count if self.faces_list.currentRow() >= 0 else 0
        self.faces_list.setCurrentRow(row)
        self.choose_source(self.faces_list.item(row).data(Qt.UserRole))

    def _change_swap_model(self, name):
        if not self.swapper or self._loading_model or not name:
            return
        same_file = os.path.normcase(os.path.abspath(self.settings.model_file(name))) == \
            os.path.normcase(os.path.abspath(self.swapper.model.path))
        if same_file:
            return
        self._loading_model = True
        self.swap_combo.setEnabled(False)
        self._set_status(f"Loading {name}...")

        def work():
            try:
                self._pause_for_build(self.settings.model_file(name), fp16=True)
                self.swapper.set_swap_model(self.settings.model_file(name))
                self.bridge.swap_model_loaded.emit(name, True, f"Swap model: {name}")
            except Exception as e:
                self.bridge.swap_model_loaded.emit(name, False, f"Could not load {name}: {e}")
        threading.Thread(target=work, daemon=True).start()

    def _pause_for_build(self, model_path, fp16):
        """Pauses the live swap while a model that has never been used is prepared for the GPU (any thread)."""
        from ..engines import is_prepared
        if self.pipeline and not is_prepared(model_path, fp16):
            self.pipeline.paused = True

    def _on_swap_model_loaded(self, name, ok, message):
        self._loading_model = False
        if self.pipeline and not self._loading_enhancer:
            self.pipeline.paused = False
        self.swap_combo.setEnabled(True)
        self._set_status(message)
        if ok:
            self._update_setting('swap_model', name)
        self.swap_combo.setCurrentText(self.swapper.model.name)
        self._refresh_boost_combo()

    def _change_enhancer(self, name):
        if not self.swapper or not name:
            return
        self.enhancer_combo.setEnabled(False)
        self._loading_enhancer = True
        self._set_status(f"Loading enhancer {name}...")

        def work():
            try:
                self._pause_for_build(self.settings.model_file(name), fp16=False)
                self.swapper.set_enhancer_model(self.settings.model_file(name))
                self.bridge.enhancer_loaded.emit(name, True, f"Enhancer: {name}")
            except Exception as e:
                self.bridge.enhancer_loaded.emit(name, False, f"Could not load {name}: {e}")
        threading.Thread(target=work, daemon=True).start()

    def _on_enhancer_loaded(self, name, ok, message):
        self._loading_enhancer = False
        if self.pipeline and not self._loading_model:
            self.pipeline.paused = False
        self.enhancer_combo.setEnabled(True)
        self._set_status(message)
        if ok:
            self._update_setting('enhancer_model', name)
            self.enhance_check.setEnabled(True)

    def _set_swap_enabled(self, enabled):
        if self.pipeline:
            self.pipeline.swap_enabled = enabled

    def _set_pixel_boost(self, size):
        if self.swapper:
            self.swapper.pixel_boost = size
            self._update_setting('pixel_boost', size)

    def cycle_pixel_boost(self):
        if self.swapper:
            sizes = self.swapper.boost_sizes
            size = sizes[(sizes.index(self.swapper.pixel_boost) + 1) % len(sizes)]
            self._set_pixel_boost(size)
            self.boost_combo.setCurrentText(str(size))

    def _set_face_mask(self, mask):
        if self.swapper:
            if mask == 'parser' and self.swapper.parser is None:
                self._set_status("Face parser not available (face_parser.onnx missing).")
                return
            self.swapper.face_mask = mask
            self._update_setting('face_mask', mask)

    def _set_fringe(self, mode):
        if mode == 'show' and self.swapper and self.swapper.parser is None:
            self._set_status("Showing the photo's fringe needs the face parser (face_parser.onnx).")
            mode = 'blend'
        if self.swapper:
            self.swapper.fringe = mode
            if mode == 'show' and self.swapper.fringe_patch is None:
                self._set_status("No fringe found in this face's photo.")
        self._update_setting('fringe', mode)
        self.fringe_combo.setCurrentIndex(FRINGE_MODES.index(mode))

    def cycle_fringe(self):
        modes = FRINGE_MODES
        self._set_fringe(modes[(modes.index(self.settings.fringe) + 1) % len(modes)])

    def _set_flip(self, name, enabled):
        if self.pipeline:
            setattr(self.pipeline, name, enabled)
        if self.swapper:
            self.swapper.tracker.reset()  # the faces jump to the other side
        self._update_setting(name, enabled)

    def toggle_face_mask(self):
        mask = 'box' if self.settings.face_mask == 'parser' else 'parser'
        self._set_face_mask(mask)
        self.mask_combo.setCurrentText(self.swapper.face_mask if self.swapper else mask)

    def _set_enhance(self, enabled):
        if self.swapper:
            self.swapper.enhancement_enabled = enabled and self.swapper.enhancer is not None
            self._update_setting('enhance', self.swapper.enhancement_enabled)

    def _set_blend(self, value):
        self.blend_label.setText(f"{value}%")
        if self.swapper:
            self.swapper.set_enhance_blend(value / 100.0)
        self._update_setting('enhance_blend', value / 100.0)

    def _set_smoothing(self, enabled):
        if self.swapper:
            self.swapper.smoothing = enabled
            self.swapper.tracker.reset()
        self._update_setting('smoothing', enabled)

    def _set_det_size(self, size):
        if self.swapper:
            self.swapper.det_size = (size, size)
        self._update_setting('det_size', size)

    def _set_detect_interval(self, value):
        if self.pipeline:
            self.pipeline.detect_interval = value
        self._update_setting('detect_interval', value)

    def _apply_camera(self):
        s = self.settings
        s.camera_index = self.camera_spin.value()
        s.resolution = self.res_combo.currentText()
        s.camera_fps = int(self.fps_combo.currentText())
        s.save()
        if self.pipeline:
            self.pipeline.open_camera(s.camera_index, *s.resolution_size(), s.camera_fps)
            if s.virtual_cam:
                self.pipeline.set_virtual_cam(True, s.camera_fps)

    def _set_virtual_cam(self, enabled):
        if self.pipeline:
            self.pipeline.set_virtual_cam(enabled, self.settings.camera_fps)
        self._update_setting('virtual_cam', enabled)

    def _set_workers(self, value):
        if self.pipeline:
            self.pipeline.set_workers(value)
        self._update_setting('max_workers', value)

    def _set_hud(self, visible):
        self.video.show_hud = visible
        self._update_setting('show_hud', visible)
        self.video.update()

    def toggle_recording(self):
        if not self.pipeline or self._closing:
            return
        # Own flag: the pipeline applies the request a frame later
        self._recording = not self._recording
        if not self._recording:
            self.pipeline.stop_recording()
            self.record_button.setText("Start recording  (R)")
        else:
            os.makedirs(self.settings.outputs_dir, exist_ok=True)
            path = os.path.join(self.settings.outputs_dir, f"output_{int(time.time())}.mp4")
            self._recording_session += 1
            self.pipeline.start_recording(path, self._recording_session)
            self.record_button.setText("Stop recording  (R)")

    def _on_recording_stopped(self, session):
        # Ignore notices about an older recording (a newer one may already have been started)
        if session == self._recording_session:
            self._recording = False
            self.record_button.setText("Start recording  (R)")

    # ------------------------------------------------------------------ files

    def _process_file(self):
        if not self.swapper or self._file_job or self._closing:
            return
        media = " ".join("*" + e for e in IMAGE_EXTENSIONS + VIDEO_EXTENSIONS + ('.gif',))
        path, _ = QFileDialog.getOpenFileName(self, "Image, video or GIF to process", self.settings.faces_dir,
                                              f"Media ({media});;All files (*)")
        if path:
            self.process_file(os.path.normpath(path))

    def process_file(self, path):
        """Swaps the faces in an image, video or GIF in the background; the result goes to the outputs folder."""
        if not self.swapper or self._file_job or self._closing:
            return
        from ..media import process_gif, process_image, process_video
        ext = os.path.splitext(path)[1].lower()
        if ext == '.gif':
            job, out_ext = process_gif, '.gif'
        elif ext in IMAGE_EXTENSIONS:
            job, out_ext = process_image, ext
        else:
            job, out_ext = process_video, '.mp4'
        target = os.path.splitext(os.path.basename(path))[0]
        source = os.path.splitext(os.path.basename(self.settings.source_image))[0] or "source"
        os.makedirs(self.settings.outputs_dir, exist_ok=True)
        out_path = os.path.join(self.settings.outputs_dir, f"processed_{target}_with_{source}_{int(time.time())}{out_ext}")

        self._cancel_file = False
        self.process_button.setEnabled(False)
        self.file_progress.setRange(0, 0)
        self.file_progress.show()
        self.cancel_button.setVisible(job is not process_image)
        boost = self.settings.file_pixel_boost
        self._set_status(f"Processing {os.path.basename(path)} (pixel boost {self.swapper.model.crop_size(boost)})...")

        def work():
            try:
                if job is process_image:
                    process_image(self.swapper, path, out_path, show=False, pixel_boost=boost)
                else:
                    job(self.swapper, path, out_path, pixel_boost=boost,
                        progress=self.bridge.file_progress.emit, cancel=lambda: self._cancel_file)
                done = "Cancelled (partial file kept)" if self._cancel_file else "Saved"
                self.bridge.file_done.emit(f"{done}: {out_path}")
            except Exception as e:
                self.bridge.file_done.emit(f"Error processing {os.path.basename(path)}: {e}")
        self._file_job = threading.Thread(target=work, daemon=True)
        self._file_job.start()

    def _on_file_progress(self, done, total):
        if total > 0:
            self.file_progress.setRange(0, total)
            self.file_progress.setValue(min(done, total))

    def _on_file_done(self, message):
        self._file_job = None
        self.file_progress.hide()
        self.cancel_button.hide()
        self.process_button.setEnabled(not self._closing)
        self._set_status(message)

    def _cancel_file_job(self):
        self._cancel_file = True

    # ------------------------------------------------------------------ folders

    def _change_models_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Models folder", self.settings.models_dir)
        if not folder:
            return
        if not list_models(folder, SWAP_MODEL_KEYS):
            QMessageBox.warning(self, "Models folder", "No swap model (inswapper / hyperswap .onnx) in that folder.")
            return
        self._update_setting('models_dir', folder)
        if self.swapper is None:
            # Loading failed earlier: try again with the new folder
            self.models_button.setEnabled(False)
            self.start()
            return
        self._populate_controls()
        self._set_status("Models folder changed. Pick a model to load it from the new folder.")

    def _change_faces_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Faces folder", self.settings.faces_dir)
        if folder:
            self._update_setting('faces_dir', folder)
            self._load_faces()

    def _change_outputs_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Outputs folder", self.settings.outputs_dir)
        if folder:
            self._update_setting('outputs_dir', folder)
            self.outputs_dir_label.setText(f"Outputs: {folder}")

    @staticmethod
    def _open_folder(folder):
        os.makedirs(folder, exist_ok=True)
        if hasattr(os, 'startfile'):
            os.startfile(folder)
        else:
            subprocess.Popen(['xdg-open', folder])

    # ------------------------------------------------------------------ keys, timer, exit

    def _add_shortcuts(self):
        def key(sequence, handler):
            QShortcut(QKeySequence(sequence), self, activated=handler, context=Qt.WindowShortcut)

        key(Qt.Key_Tab, self.toggle_panel)
        key(Qt.Key_F1, self.toggle_panel)
        key(Qt.Key_Escape, self._escape)
        key(Qt.Key_F11, self._toggle_fullscreen)
        key(Qt.Key_N, lambda: self.step_source(1))
        key(Qt.Key_P, lambda: self.step_source(-1))
        key(Qt.Key_X, self.swap_check.toggle)
        key(Qt.Key_E, lambda: self.enhance_check.isEnabled() and self.enhance_check.toggle())
        key(Qt.Key_B, self.cycle_pixel_boost)
        key(Qt.Key_M, self.toggle_face_mask)
        key(Qt.Key_F, self.cycle_fringe)
        key(Qt.Key_R, self.toggle_recording)
        key(Qt.Key_U, self.hud_check.toggle)
        key(Qt.Key_Q, self.close)

    def _escape(self):
        if self.panel.isVisible():
            self.toggle_panel(False)
        elif self.isFullScreen():
            self.showNormal()

    def _toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def _tick(self):
        if self.pipeline:
            number, frame = self.pipeline.latest()
            if number != self._last_frame_number and frame is not None:
                self._last_frame_number = number
                self.video.set_frame(frame)
            elif self.video.frame is None and self.pipeline.camera_error:
                self.video.set_placeholder("No camera image", self.pipeline.camera_error, spinner=False)
        if self.video.frame is None and self.video.placeholder and self.video.placeholder[2]:
            self.video.update()  # animate the loading spinner
        self._update_hud()

    def _update_hud(self):
        green, red, white = QColor(80, 220, 120), QColor(240, 90, 90), QColor(235, 235, 235)
        lines = []
        if self.swapper and self.pipeline:
            sw = self.swapper
            lines.append((f"FPS: {self.pipeline.fps:.1f}", green))
            if not self.pipeline.swap_enabled:
                lines.append(("Swap: OFF", red))
            elif self.pipeline.paused:
                lines.append(("Swap paused: preparing a model for your GPU (first time only)", red))
            elif sw.source_face is None:
                lines.append(("No source face (Tab)", red))
            line = f"{sw.model.name} | boost {sw.pixel_boost} | mask {sw.face_mask}"
            if sw.fringe != 'blend':
                none = sw.fringe == 'show' and sw.fringe_patch is None
                line += f" | fringe {sw.fringe}" + (" (none in photo)" if none else "")
            lines.append((line, white))
            lines.append((f"Enhancer: {'ON' if sw.enhancement_enabled else 'OFF'}",
                          green if sw.enhancement_enabled else white))
            if self.settings.source_image:
                lines.append((f"Face: {os.path.basename(self.settings.source_image)}", white))
        if not self.panel.isVisible():
            lines.append(("Tab: settings", QColor(170, 170, 180)))
        recording = bool(self.pipeline and self.pipeline.recording)
        if recording != self.video.recording:
            self.record_button.setText("Stop recording  (R)" if self._recording else "Start recording  (R)")
        self.video.recording = recording
        # Status messages fade out after a while
        busy = self._file_job or self._loading_model or self._loading_enhancer or self.swapper is None
        if self.video.status and time.time() - self._status_time > 8 and not busy:
            self.video.status = ""
        if lines != self.video.hud_lines:
            self.video.hud_lines = lines
            self.video.update()

    def closeEvent(self, event):
        if self._closed:
            super().closeEvent(event)
            return
        # Finish the recording and stop a file job in the background, keeping the window responsive,
        # then close for real (see _on_shutdown_done)
        event.ignore()
        if self._closing:
            return
        self._closing = True
        self.settings.save()
        for w in self._controls:
            w.setEnabled(False)
        self._cancel_file = True
        file_job, pipeline, swapper = self._file_job, self.pipeline, self.swapper
        if file_job or (pipeline and pipeline.recording):
            self._set_status("Closing: finishing the recording / file...")

        def shutdown():
            # Stop engine builds first (a TensorRT build can take minutes)
            if 'src.engines' in sys.modules:
                sys.modules['src.engines'].cancel_builds()
            if file_job:
                file_job.join()  # a cancelled video is still finalized
            if pipeline:
                pipeline.stop()  # encodes an ongoing recording
            if swapper:
                swapper.executor.shutdown(wait=False, cancel_futures=True)
            self.bridge.shutdown_done.emit()
        threading.Thread(target=shutdown, daemon=True).start()

    def _on_shutdown_done(self):
        self._closed = True
        self.close()


def run():
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Deepfake")
    app.setStyle("Fusion")
    window = MainWindow(Settings.load())
    window.show()
    app.processEvents()  # paint the loading screen before anything else
    if not window.start():
        return 0
    return app.exec()
