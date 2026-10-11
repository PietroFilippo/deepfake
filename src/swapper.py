import cv2
import insightface
import numpy as np
import os
import concurrent.futures
from insightface.app.common import Face
from insightface.utils import face_align
from .utils import setup_dll_directories, get_default_providers, model_path as default_model_path, read_image
from .face_utils import box_mask, paste_back
from .enhancer import FaceEnhancer
from .face_parser import FaceParser

# Setup DLL directories for Windows
setup_dll_directories()

# Crop sizes supported by pixel boost (multiples of the model's native 128)
PIXEL_BOOST_SIZES = (128, 256, 384, 512)
FACE_MASKS = ('box', 'parser')


class FaceSwapper:
    def __init__(self, model_path, providers=None, det_size=(320, 320), max_workers=None,
                 pixel_boost=128, face_mask='box', enhancer_model=None, enhance_blend=0.8):
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

        self.swapper = insightface.model_zoo.get_model(model_path, providers=self.providers)
        self.model_size = self.swapper.input_size[0]

        self.pixel_boost = pixel_boost
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

    @property
    def pixel_boost(self):
        return self._pixel_boost

    @pixel_boost.setter
    def pixel_boost(self, size):
        if size not in PIXEL_BOOST_SIZES:
            raise ValueError(f"pixel_boost must be one of {PIXEL_BOOST_SIZES}")
        self._pixel_boost = size

    def set_source_image(self, source_img_path):
        img = read_image(source_img_path)
        if img is None:
            raise ValueError(f"Could not read the source image: {source_img_path}")

        faces = self.app.get(img)
        if len(faces) == 0:
            raise ValueError("No face detected in the source image")

        # Use the largest face
        source_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

        # Identity vector the swap model expects (ArcFace embedding projected by the model's emap).
        # Computed once here instead of for every face in every frame.
        latent = source_face.normed_embedding.reshape((1, -1))
        latent = np.dot(latent, self.swapper.emap)
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
        """Runs inswapper on a 128x128 BGR crop and returns the swapped 128x128 BGR crop (float32, 0-1)."""
        blob = cv2.dnn.blobFromImage(crop, 1.0 / 255.0, (self.model_size, self.model_size),
                                     (0.0, 0.0, 0.0), swapRB=True)
        pred = self.swapper.session.run(self.swapper.output_names,
                                        {self.swapper.input_names[0]: blob, self.swapper.input_names[1]: latent})[0]
        return pred[0].transpose(1, 2, 0)[:, :, ::-1]

    def _swap_face(self, frame, face, latent):
        """Swaps one face in `frame` (in place)."""
        size = self.pixel_boost
        n = size // self.model_size
        matrix = face_align.estimate_norm(face.kps, size)
        crop = cv2.warpAffine(frame, matrix, (size, size), borderMode=cv2.BORDER_REPLICATE)

        if n == 1:
            swapped = self._run_model(crop, latent)
        else:
            # Pixel boost: split the high-res crop into n*n interleaved 128x128 sub-images
            # (every n-th pixel), swap each one, then interleave the results back together.
            # The output keeps n times more detail than a single 128x128 pass.
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
