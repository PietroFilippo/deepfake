import cv2
import insightface
import numpy as np
import onnx
import onnxruntime
import os
import threading
import time
import concurrent.futures
from collections import namedtuple
from onnx import numpy_helper
from insightface.app.common import Face
from insightface.utils import face_align
from .utils import setup_dll_directories, get_default_providers, model_path as default_model_path, read_image
from .face_utils import box_mask, paste_back
from .enhancer import FaceEnhancer
from .face_parser import FaceParser
from .tracking import FaceTracker

# Setup DLL directories for Windows
setup_dll_directories()

# Crop sizes supported by pixel boost (multiples of the model's native resolution)
PIXEL_BOOST_SIZES = (128, 256, 384, 512)
FACE_MASKS = ('box', 'parser')

# Everything a worker needs to process one frame, captured when the frame is submitted, so
# settings or models can change at any time without a frame mixing old and new state
SwapJob = namedtuple('SwapJob', 'model latent crop_size face_mask enhancer')


class SwapModel:
    """
    A loaded swap model. Supported:
    - inswapper_128 (InsightFace): 128x128, RGB 0-1, identity = ArcFace embedding projected by the model's emap.
    - hyperswap_1a/1b/1c_256 (FaceFusion): 256x256, RGB -1..1, identity = normalized ArcFace embedding.
    Both use the same ArcFace 5-point alignment, so the rest of the pipeline is shared.
    """

    def __init__(self, model_path, providers):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.path = model_path
        self.name = os.path.basename(model_path)
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        inputs = self.session.get_inputs()
        target = next(i for i in inputs if len(i.shape) == 4)
        source = next(i for i in inputs if len(i.shape) == 2)
        outputs = [o.name for o in self.session.get_outputs()]
        self.target_name, self.source_name, self.output_name = target.name, source.name, outputs[0]
        self.size = target.shape[2]

        if self.size == 128:
            self.type = 'inswapper'
            self.mean, self.std = 0.0, 1.0
            # The identity projection matrix is stored as the graph's last initializer
            self.emap = numpy_helper.to_array(onnx.load(model_path).graph.initializer[-1])
        elif self.size == 256 and 'mask' in outputs:
            self.type = 'hyperswap'
            self.mean, self.std = 0.5, 0.5
            self.emap = None
        else:
            raise ValueError(f"Unsupported swap model: {model_path} (input {target.shape}, outputs {outputs})")
        print(f"[FaceSwapper] Swap model: {self.name} ({self.type}, {self.size}x{self.size})")

    @property
    def boost_sizes(self):
        """Pixel boost sizes usable with this model (multiples of its native resolution)."""
        return tuple(s for s in PIXEL_BOOST_SIZES if s >= self.size and s % self.size == 0)

    def crop_size(self, pixel_boost):
        """The largest supported pixel boost size not above `pixel_boost` (at least the native size)."""
        usable = [s for s in self.boost_sizes if s <= pixel_boost]
        return usable[-1] if usable else self.boost_sizes[0]

    def latent(self, source_face):
        """Identity vector this model expects. inswapper projects the ArcFace embedding with its emap first."""
        latent = source_face.normed_embedding.reshape((1, -1))
        if self.emap is not None:
            latent = np.dot(latent, self.emap)
            latent /= np.linalg.norm(latent)
        return latent.astype(np.float32)

    def run(self, crop, latent):
        """Runs the model on a model-sized BGR crop and returns the swapped BGR crop (float32, 0-1)."""
        # (pixel / 255 - mean) / std, BGR -> RGB, NCHW
        mean, std = self.mean * 255.0, self.std * 255.0
        blob = cv2.dnn.blobFromImage(crop, 1.0 / std, (self.size, self.size), (mean, mean, mean), swapRB=True)
        pred = self.session.run([self.output_name], {self.target_name: blob, self.source_name: latent})[0]
        return (pred[0].transpose(1, 2, 0) * self.std + self.mean)[:, :, ::-1]


class FaceSwapper:
    def __init__(self, model_path, providers=None, det_size=(640, 640), max_workers=None,
                 pixel_boost=None, face_mask='box', enhancer_model=None, enhance_blend=0.8,
                 parser_model=None, smoothing=True):
        if providers is None:
            providers = get_default_providers()
        self.providers = providers
        self.det_size = det_size

        print(f"[FaceSwapper] Initializing FaceAnalysis with providers: {self.providers} and det_size: {self.det_size}")

        # Face detection app. Only detection (target faces) and recognition (source identity)
        # are needed; the 106/68-point landmark and gender/age models are skipped.
        self.app = insightface.app.FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'],
                                                providers=self.providers)
        self.app.prepare(ctx_id=0, det_size=self.det_size)

        self.source_face = None
        # (model, latent) are always replaced together so workers never see a mismatched pair
        self._active = (SwapModel(model_path, self.providers), None)
        self._lock = threading.Lock()

        # Requested pixel boost; each model uses the closest size it supports
        self._pixel_boost = pixel_boost or self.model.size
        self.face_mask = face_mask

        # Face parser for the anatomical masks (optional for the swap, used by the enhancer)
        try:
            self.parser = FaceParser(parser_model)
        except Exception as e:
            print(f"[FaceSwapper] Warning: FaceParser unavailable ({e}). Using box masks.")
            self.parser = None

        # Initialize the Enhancer (GFPGAN by default, '' = no enhancer)
        self.enhancement_enabled = False # Disabled by default
        self.enhance_blend = enhance_blend
        self.enhancer = None
        if enhancer_model is None:
            enhancer_model = default_model_path("GFPGANv1.4.onnx")
        if enhancer_model:
            try:
                self.set_enhancer_model(enhancer_model)
            except Exception as e:
                print(f"[FaceSwapper] Warning: Could not load FaceEnhancer: {e}")

        self._swap_error_logged = False

        # State for periodic detection and landmark smoothing (live mode)
        self.frame_count = 0
        self.last_faces = []
        self.smoothing = smoothing
        self.tracker = FaceTracker()

        # Thread pool for parallel processing
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            max_workers = min(cpu_count, 5)
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        print(f"[FaceSwapper] Initialized with {self.max_workers} worker threads.")

    # ------------------------------------------------------------------ models and settings

    @property
    def model(self):
        return self._active[0]

    @property
    def source_latent(self):
        return self._active[1]

    @property
    def model_size(self):
        return self.model.size

    @property
    def boost_sizes(self):
        return self.model.boost_sizes

    @property
    def pixel_boost(self):
        """Pixel boost actually used with the current model."""
        return self.model.crop_size(self._pixel_boost)

    @pixel_boost.setter
    def pixel_boost(self, size):
        if size not in PIXEL_BOOST_SIZES:
            raise ValueError(f"pixel_boost must be one of {PIXEL_BOOST_SIZES}")
        self._pixel_boost = size
        if self.pixel_boost != size:
            print(f"[FaceSwapper] Pixel boost {size} is not supported by this model, using {self.pixel_boost}")

    def set_swap_model(self, model_path):
        """
        Loads another swap model (can take a while: TensorRT builds an engine the first time)
        and switches to it. Frames already submitted finish with the previous model.
        """
        model = SwapModel(model_path, self.providers)
        with self._lock:
            latent = model.latent(self.source_face) if self.source_face is not None else None
            self._active = (model, latent)

    def set_enhancer_model(self, model_path):
        """Loads another enhancer model (GFPGAN/GPEN style) and switches to it."""
        enhancer = FaceEnhancer(model_path, parser=self.parser, blend=self.enhance_blend)
        self.enhancer = enhancer

    def set_enhance_blend(self, blend):
        self.enhance_blend = float(np.clip(blend, 0.0, 1.0))
        if self.enhancer:
            self.enhancer.blend = self.enhance_blend

    def set_max_workers(self, max_workers):
        old = self.executor
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        old.shutdown(wait=False)

    def set_source_image(self, source_img_path):
        img = read_image(source_img_path)
        if img is None:
            raise ValueError(f"Could not read the source image: {source_img_path}")

        faces = self.app.get(img)
        if len(faces) == 0:
            raise ValueError("No face detected in the source image")

        # Use the largest face
        source_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

        # The identity vector is computed once here instead of for every face in every frame
        with self._lock:
            model = self.model
            self.source_face = source_face
            self._active = (model, model.latent(source_face))
        print("[FaceSwapper] Source face set")

    def job(self, pixel_boost=None):
        """Snapshot of the current models and settings for one frame."""
        model, latent = self._active
        enhancer = self.enhancer if self.enhancement_enabled else None
        crop_size = model.crop_size(pixel_boost or self._pixel_boost)
        return SwapJob(model, latent, crop_size, self.face_mask, enhancer)

    # ------------------------------------------------------------------ processing

    def detect_faces(self, frame, input_size=None):
        """Runs only the face detector. Returns Face objects with bbox, kps and det_score."""
        bboxes, kpss = self.app.det_model.detect(frame, input_size=tuple(input_size or self.det_size),
                                                 max_num=0, metric='default')
        if kpss is None:
            return []
        return [Face(bbox=bbox[:4], kps=kps, det_score=bbox[4]) for bbox, kps in zip(bboxes, kpss)]

    def _swap_face(self, frame, face, job):
        """Swaps one face in `frame` (in place)."""
        model, size = job.model, job.crop_size
        n = size // model.size
        matrix = face_align.estimate_norm(face.kps, size)
        crop = cv2.warpAffine(frame, matrix, (size, size), borderMode=cv2.BORDER_REPLICATE)

        if n == 1:
            swapped = model.run(crop, job.latent)
        else:
            # Pixel boost: split the high-res crop into n*n interleaved model-sized sub-images
            # (every n-th pixel), swap each one, then interleave the results back together.
            # The output keeps n times more detail than a single pass.
            m = model.size
            tiles = crop.reshape(m, n, m, n, 3).transpose(1, 3, 0, 2, 4).reshape(n * n, m, m, 3)
            swapped = np.stack([model.run(np.ascontiguousarray(tile), job.latent) for tile in tiles])
            swapped = swapped.reshape(n, n, m, m, 3).transpose(2, 0, 3, 1, 4).reshape(size, size, 3)

        swapped = np.clip(swapped * 255.0 + 0.5, 0, 255).astype(np.uint8)

        mask = box_mask(size)
        if job.face_mask == 'parser' and self.parser is not None:
            # Parse the target crop so hair, hands and glasses in front of the face are kept
            region = self.parser.parse(crop)
            if region.any():
                mask = np.minimum(mask, region)

        paste_back(frame, swapped, mask, matrix)

    def _swap_worker(self, frame, faces, job):
        res = frame.copy()
        if job.latent is None:
            return res
        for face in faces:
            try:
                self._swap_face(res, face, job)
            except Exception as e:
                if not self._swap_error_logged:
                    print(f"[FaceSwapper] Swap error (further errors are not shown): {e}")
                    self._swap_error_logged = True

        # Apply enhancement if enabled and available
        if job.enhancer:
            try:
                res = job.enhancer.enhance(res, faces)
            except Exception as e:
                print(f"[FaceSwapper] Enhancer error: {e}")

        return res

    def process_frame(self, frame, tracker=None, timestamp=None, pixel_boost=None, det_size=None):
        """
        Detects and swaps every face in a single frame (synchronous, used by the offline modes).
        Pass a FaceTracker and the frame's timestamp (seconds) to smooth landmarks across video frames.
        """
        if self.source_latent is None:
            return frame
        faces = self.detect_faces(frame, input_size=det_size)
        if tracker is not None and timestamp is not None:
            faces = tracker.update(faces, timestamp)
        return self._swap_worker(frame, faces, self.job(pixel_boost))

    def process_frame_async(self, frame, detect_interval=1, timestamp=None):
        if self.source_latent is None:
            # Return a completed future with the original frame
            future = concurrent.futures.Future()
            future.set_result(frame)
            return future

        # Detect faces periodically (synchronous to keep the state simple)
        if self.frame_count % detect_interval == 0:
            try:
                faces = self.detect_faces(frame)
                if self.smoothing:
                    faces = self.tracker.update(faces, time.monotonic() if timestamp is None else timestamp)
                self.last_faces = faces
            except Exception as e:
                print(f"Detection error: {e}")
                self.last_faces = []
        self.frame_count += 1

        # Submit the swap task to the thread pool.
        # The worker copies the frame before drawing on it.
        return self.executor.submit(self._swap_worker, frame, list(self.last_faces), self.job())

    def toggle_enhancer(self):
        if self.enhancer:
            self.enhancement_enabled = not self.enhancement_enabled
            return self.enhancement_enabled
        return False
