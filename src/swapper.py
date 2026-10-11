import cv2
import insightface
import numpy as np
import os
import time
import concurrent.futures
from .utils import setup_dll_directories, get_default_providers
from .enhancer import FaceEnhancer

# Setup DLL directories for Windows
setup_dll_directories()

class FaceSwapper:
    def __init__(self, model_path, providers=None, det_size=(320, 320), max_workers=None):
        if providers is None:
            providers = get_default_providers()
        self.providers = providers
        self.det_size = det_size

        print(f"[FaceSwapper] Initializing FaceAnalysis with providers: {self.providers} and det_size: {self.det_size}")
        
        # Face detection app
        self.app = insightface.app.FaceAnalysis(name='buffalo_l', providers=self.providers)
        self.app.prepare(ctx_id=0, det_size=self.det_size)

        # Face swap model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        if not os.path.exists('trt_cache'):
            os.makedirs('trt_cache')
            
        self.swapper = insightface.model_zoo.get_model(model_path, providers=self.providers)
        
        # Initialize the Enhancer (GFPGAN)
        try:
            self.enhancer = FaceEnhancer()
            self.enhancement_enabled = False # Disabled by default
        except Exception as e:
            print(f"[FaceSwapper] Warning: Could not load FaceEnhancer: {e}")
            self.enhancer = None
            self.enhancement_enabled = False

        self.source_face = None

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

    def set_source_image(self, source_img_path):
        img = cv2.imread(source_img_path)
        if img is None:
            raise ValueError(f"Could not read the source image: {source_img_path}")

        faces = self.app.get(img)
        if len(faces) == 0:
            raise ValueError("No face detected in the source image")

        # Use the largest face
        self.source_face = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0])*(x.bbox[3] - x.bbox[1]))[-1]
        print("[FaceSwapper] Source face set")

    def _detect_faces_downscale(self, frame, scale=0.5):
        small = cv2.resize(frame, (0,0), fx=scale, fy=scale)
        faces_small = self.app.get(small)
        faces = []
        for f in faces_small:
            f.bbox = (np.array(f.bbox) / scale).astype(np.float32)
            if hasattr(f, 'kps') and f.kps is not None:
                f.kps = (np.array(f.kps) / scale).astype(np.float32)
            faces.append(f)
        return faces

    def _swap_worker(self, frame, faces, source_face):
        res = frame.copy()
        for face in faces:
            try:
                res = self.swapper.get(res, face, source_face, paste_back=True)
            except Exception:
                continue
        
        # Apply enhancement if enabled and available
        if self.enhancer and self.enhancement_enabled:
            try:
                res = self.enhancer.enhance(res, faces)
            except Exception as e:
                print(f"[FaceSwapper] Enhancer error: {e}")
                
        return res

    def process_frame_async(self, frame, detect_interval=5):
        if self.source_face is None:
            # Return a completed future with the original frame
            future = concurrent.futures.Future()
            future.set_result(frame)
            return future

        # Detect faces periodically (synchronous to keep the state simple)
        if self.frame_count % detect_interval == 0:
            try:
                faces = self._detect_faces_downscale(frame, scale=0.5)
                self.last_faces = faces
            except Exception as e:
                print(f"Detection error: {e}")
                pass
        self.frame_count += 1

        # Submit the swap task to the thread pool
        # Copy the frame and faces to ensure thread safety
        frame_copy = frame.copy()
        faces_copy = list(self.last_faces) 
        
        future = self.executor.submit(self._swap_worker, frame_copy, faces_copy, self.source_face)
        return future

    def toggle_enhancer(self):
        if self.enhancer:
            self.enhancement_enabled = not self.enhancement_enabled
            return self.enhancement_enabled
        return False