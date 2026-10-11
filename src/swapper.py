import cv2
import insightface
import numpy as np
import onnx
import onnxruntime
import os
import concurrent.futures
from onnx import numpy_helper
from insightface.app.common import Face
from insightface.utils import face_align
from .utils import setup_dll_directories, get_default_providers, model_path as default_model_path, read_image
from .face_utils import box_mask, paste_back
from .enhancer import FaceEnhancer
from .face_parser import FaceParser

# Setup DLL directories for Windows
setup_dll_directories()

# Crop sizes supported by pixel boost (multiples of the model's native resolution)
PIXEL_BOOST_SIZES = (128, 256, 384, 512)
FACE_MASKS = ('box', 'parser')


class FaceSwapper:
    def __init__(self, model_path, providers=None, det_size=(320, 320), max_workers=None,
                 pixel_boost=None, face_mask='box', enhancer_model=None, enhance_blend=0.8):
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

        # Face swap model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self._load_swap_model(model_path)

        # Default: the model's native resolution (no pixel boost)
        self.pixel_boost = pixel_boost or self.model_size
        self.face_mask = face_mask

        # Face parser for the anatomical masks (optional for the swap, used by the enhancer)
        try:
            self.parser = FaceParser()
        except Exception as e:
            print(f"[FaceSwapper] Warning: FaceParser unavailable ({e}). Using box masks.")
            self.parser = None

        # Initialize the Enhancer (GFPGAN by default)
        self.enhancement_enabled = False # Disabled by default
        try:
            self.enhancer = FaceEnhancer(enhancer_model or default_model_path("GFPGANv1.4.onnx"),
                                         parser=self.parser, blend=enhance_blend)
        except Exception as e:
            print(f"[FaceSwapper] Warning: Could not load FaceEnhancer: {e}")
            self.enhancer = None

        self.source_face = None
        self.source_latent = None
        self._swap_error_logged = False

        # State for periodic detection
        self.frame_count = 0
        self.last_faces = []

        # Thread pool for parallel processing
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            self.max_workers = min(cpu_count, 5)
        else:
            self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        print(f"[FaceSwapper] Initialized with {self.max_workers} worker threads.")

    def _load_swap_model(self, model_path):
        """
        Loads a swap model. Supported:
        - inswapper_128 (InsightFace): 128x128, RGB 0-1, identity = ArcFace embedding projected by the model's emap.
        - hyperswap_1a/1b/1c_256 (FaceFusion): 256x256, RGB -1..1, identity = normalized ArcFace embedding.
        Both use the same ArcFace 5-point alignment, so the rest of the pipeline is shared.
        """
        self.session = onnxruntime.InferenceSession(model_path, providers=self.providers)
        inputs = self.session.get_inputs()
        target = next(i for i in inputs if len(i.shape) == 4)
        source = next(i for i in inputs if len(i.shape) == 2)
        outputs = [o.name for o in self.session.get_outputs()]
        self.target_name, self.source_name, self.output_name = target.name, source.name, outputs[0]
        self.model_size = target.shape[2]

        if self.model_size == 128:
            self.model_type = 'inswapper'
            self.model_mean, self.model_std = 0.0, 1.0
            # The identity projection matrix is stored as the graph's last initializer
            self.emap = numpy_helper.to_array(onnx.load(model_path).graph.initializer[-1])
        elif self.model_size == 256 and 'mask' in outputs:
            self.model_type = 'hyperswap'
            self.model_mean, self.model_std = 0.5, 0.5
            self.emap = None
        else:
            raise ValueError(f"Unsupported swap model: {model_path} (input {target.shape}, outputs {outputs})")
        print(f"[FaceSwapper] Swap model: {self.model_type} ({self.model_size}x{self.model_size})")

    @property
    def boost_sizes(self):
        """Pixel boost sizes usable with the loaded model (multiples of its native resolution)."""
        return tuple(s for s in PIXEL_BOOST_SIZES if s >= self.model_size and s % self.model_size == 0)

    @property
    def pixel_boost(self):
        return self._pixel_boost

    @pixel_boost.setter
    def pixel_boost(self, size):
        if size not in PIXEL_BOOST_SIZES:
            raise ValueError(f"pixel_boost must be one of {PIXEL_BOOST_SIZES}")
        # Snap to the closest size the model supports (e.g. 128 -> 256 for a 256 model)
        usable = [s for s in self.boost_sizes if s <= size] or [self.boost_sizes[0]]
        if usable[-1] != size:
            print(f"[FaceSwapper] Pixel boost {size} is not supported by this model, using {usable[-1]}")
        self._pixel_boost = usable[-1]

    def set_source_image(self, source_img_path):
        img = read_image(source_img_path)
        if img is None:
            raise ValueError(f"Could not read the source image: {source_img_path}")

        faces = self.app.get(img)
        if len(faces) == 0:
            raise ValueError("No face detected in the source image")

        # Use the largest face
        source_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

        # Identity vector the swap model expects, computed once here instead of for every face
        # in every frame. inswapper projects the ArcFace embedding with its emap first.
        latent = source_face.normed_embedding.reshape((1, -1))
        if self.emap is not None:
            latent = np.dot(latent, self.emap)
            latent /= np.linalg.norm(latent)

        # Assigned together so worker threads never see a mismatched pair
        self.source_face, self.source_latent = source_face, latent.astype(np.float32)
        print("[FaceSwapper] Source face set")

    def detect_faces(self, frame):
        """Runs only the face detector. Returns Face objects with bbox, kps and det_score."""
        bboxes, kpss = self.app.det_model.detect(frame, max_num=0, metric='default')
        if kpss is None:
            return []
        return [Face(bbox=bbox[:4], kps=kps, det_score=bbox[4]) for bbox, kps in zip(bboxes, kpss)]

    def _run_model(self, crop, latent):
        """Runs the swap model on a model-sized BGR crop and returns the swapped BGR crop (float32, 0-1)."""
        # (pixel / 255 - mean) / std, BGR -> RGB, NCHW
        mean, std = self.model_mean * 255.0, self.model_std * 255.0
        blob = cv2.dnn.blobFromImage(crop, 1.0 / std, (self.model_size, self.model_size),
                                     (mean, mean, mean), swapRB=True)
        pred = self.session.run([self.output_name], {self.target_name: blob, self.source_name: latent})[0]
        return (pred[0].transpose(1, 2, 0) * self.model_std + self.model_mean)[:, :, ::-1]

    def _swap_face(self, frame, face, latent):
        """Swaps one face in `frame` (in place)."""
        size = self.pixel_boost
        n = size // self.model_size
        matrix = face_align.estimate_norm(face.kps, size)
        crop = cv2.warpAffine(frame, matrix, (size, size), borderMode=cv2.BORDER_REPLICATE)

        if n == 1:
            swapped = self._run_model(crop, latent)
        else:
            # Pixel boost: split the high-res crop into n*n interleaved model-sized sub-images
            # (every n-th pixel), swap each one, then interleave the results back together.
            # The output keeps n times more detail than a single pass.
            m = self.model_size
            tiles = crop.reshape(m, n, m, n, 3).transpose(1, 3, 0, 2, 4).reshape(n * n, m, m, 3)
            swapped = np.stack([self._run_model(np.ascontiguousarray(tile), latent) for tile in tiles])
            swapped = swapped.reshape(n, n, m, m, 3).transpose(2, 0, 3, 1, 4).reshape(size, size, 3)

        swapped = np.clip(swapped * 255.0 + 0.5, 0, 255).astype(np.uint8)

        mask = box_mask(size)
        if self.face_mask == 'parser' and self.parser is not None:
            # Parse the target crop so hair, hands and glasses in front of the face are kept
            region = self.parser.parse(crop)
            if region.any():
                mask = np.minimum(mask, region)

        paste_back(frame, swapped, mask, matrix)

    def _swap_worker(self, frame, faces, latent):
        res = frame.copy()
        for face in faces:
            try:
                self._swap_face(res, face, latent)
            except Exception as e:
                if not self._swap_error_logged:
                    print(f"[FaceSwapper] Swap error (further errors are not shown): {e}")
                    self._swap_error_logged = True

        # Apply enhancement if enabled and available
        if self.enhancer and self.enhancement_enabled:
            try:
                res = self.enhancer.enhance(res, faces)
            except Exception as e:
                print(f"[FaceSwapper] Enhancer error: {e}")

        return res

    def process_frame(self, frame):
        """Detects and swaps every face in a single frame (synchronous, used by the offline modes)."""
        if self.source_latent is None:
            return frame
        return self._swap_worker(frame, self.detect_faces(frame), self.source_latent)

    def process_frame_async(self, frame, detect_interval=2):
        if self.source_latent is None:
            # Return a completed future with the original frame
            future = concurrent.futures.Future()
            future.set_result(frame)
            return future

        # Detect faces periodically (synchronous to keep the state simple)
        if self.frame_count % detect_interval == 0:
            try:
                self.last_faces = self.detect_faces(frame)
            except Exception as e:
                print(f"Detection error: {e}")
                self.last_faces = []
        self.frame_count += 1

        # Submit the swap task to the thread pool.
        # The worker copies the frame before drawing on it.
        return self.executor.submit(self._swap_worker, frame, list(self.last_faces), self.source_latent)

    def toggle_enhancer(self):
        if self.enhancer:
            self.enhancement_enabled = not self.enhancement_enabled
            return self.enhancement_enabled
        return False
