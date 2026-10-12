import cv2
import numpy as np
import os
from .engines import create_session
from .utils import model_path as default_model_path

class FaceParser:
    # ImageNet normalization (standard CelebAMask-HQ BiSeNet training)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # CelebAMask-HQ 19 classes:
    #  0 bg | 1 skin | 2 l_brow | 3 r_brow | 4 l_eye | 5 r_eye | 6 eye_g
    #  7 l_ear | 8 r_ear | 9 ear_r | 10 nose | 11 mouth | 12 u_lip | 13 l_lip
    # 14 neck | 15 neck_l | 16 cloth | 17 hair | 18 hat
    # Only include the face regions that the swap/enhance should affect.
    FACE_CLASSES = (1, 2, 3, 4, 5, 10, 11, 12, 13)

    def __init__(self, model_path=None, providers=None, on_status=None):
        model_path = model_path or default_model_path("face_parser.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"face_parser model not found at {model_path}")

        print(f"[FaceParser] Loading {model_path}")
        self.session = create_session(model_path, on_status=on_status, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        # The first output is the main logit; the others are auxiliary heads
        self.output_name = self.session.get_outputs()[0].name
        self._face_class_arr = np.array(self.FACE_CLASSES, dtype=np.uint8)
        print("[FaceParser] Loaded.")

    def labels(self, face_bgr):
        """Class of every pixel (256x256 uint8, CelebAMask-HQ classes above)."""
        # RGB, (x / 255 - mean) / std, NCHW
        blob = cv2.dnn.blobFromImage(face_bgr, 1.0 / 255.0, (512, 512), tuple(float(m) * 255.0 for m in self.MEAN),
                                     swapRB=True)
        blob /= self.STD[None, :, None, None]

        out = self.session.run([self.output_name], {self.input_name: blob})[0]
        # The argmax runs on the CPU: at half resolution it takes ~4 ms instead of ~20 ms, plenty for masks
        return out[0, :, ::2, ::2].argmax(axis=0).astype(np.uint8)

    def parse(self, face_bgr):
        """Mask (H, W) float32 in [0,1]. 1 = face region to be blended."""
        h, w = face_bgr.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((max(h, 1), max(w, 1)), dtype=np.float32)

        parsing = self.labels(face_bgr)
        mask = np.isin(parsing, self._face_class_arr).astype(np.float32)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        # Feather: erode slightly to avoid bleeding into hair/contour,
        # then a Gaussian blur proportional to the crop size.
        k = max(1, int(min(h, w) * 0.01))
        if k > 1:
            kernel = np.ones((k, k), dtype=np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
        sigma = max(3.0, min(h, w) * 0.02)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)
        return np.clip(mask, 0.0, 1.0)
