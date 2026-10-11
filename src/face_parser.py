import cv2
import numpy as np
import onnxruntime
import os
from .utils import get_default_providers, model_path as default_model_path

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

    def __init__(self, model_path=None, providers=None):
        model_path = model_path or default_model_path("face_parser.onnx")
        if providers is None:
            providers = get_default_providers()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"face_parser model not found at {model_path}")

        print(f"[FaceParser] Loading {model_path}")
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        # The first output is the main logit; the others are auxiliary heads
        self.output_name = self.session.get_outputs()[0].name
        self._face_class_arr = np.array(self.FACE_CLASSES, dtype=np.uint8)
        print("[FaceParser] Loaded.")

    def parse(self, face_bgr):
        """Mask (H, W) float32 in [0,1]. 1 = face region to be blended."""
        h, w = face_bgr.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((max(h, 1), max(w, 1)), dtype=np.float32)

        img = cv2.resize(face_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.MEAN) / self.STD
        img = np.transpose(img, (2, 0, 1))[None, ...].astype(np.float32)

        out = self.session.run([self.output_name], {self.input_name: img})[0]
        parsing = out[0].argmax(axis=0).astype(np.uint8)  # (512,512)

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
