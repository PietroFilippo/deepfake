import cv2
import numpy as np
import os
from .engines import create_session
from .utils import model_path as default_model_path
from .face_utils import box_mask, estimate_ffhq_matrix, paste_back


class FaceEnhancer:
    def __init__(self, model_path=None, providers=None, parser=None, blend=0.8, on_status=None):
        """
        Face restoration with GFPGAN-style ONNX models (GFPGANv1.4, GPEN-BFR-512, ...):
        FFHQ-aligned 512x512 RGB input and output, values between -1 and 1.

        Args:
            model_path: Path to the ONNX model (default: models/GFPGANv1.4.onnx).
            providers: ONNX Runtime providers (default: TensorRT FP32 > CUDA > CPU, prebuilt engine).
            parser: Optional FaceParser used to restrict the blend to the face region.
            blend: How much of the restored face is mixed in (0-1). Lower = less "plastic".
        """
        model_path = model_path or default_model_path("GFPGANv1.4.onnx")
        self.parser = parser
        self.blend = float(np.clip(blend, 0.0, 1.0))

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Enhancer model not found at {model_path}")

        print(f"[FaceEnhancer] Loading enhancer model from {model_path}")
        try:
            # TensorRT in FP32 mode (full precision): FP16 makes the face blurry
            self.session = create_session(model_path, fp16=False, on_status=on_status, providers=providers)
        except Exception as e:
            print(f"[FaceEnhancer] Error loading model: {e}")
            raise

        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        size = model_input.shape[2]
        self.size = size if isinstance(size, int) else 512
        print("[FaceEnhancer] Model loaded successfully.")

    def enhance(self, frame, faces, blend=None):
        """
        Enhances the quality of the detected faces in the frame.

        Args:
            frame: BGR image (numpy array).
            faces: List of face objects (InsightFace) that were swapped. Only `kps` is used.
            blend: Overrides self.blend for this call.

        Returns:
            frame: Image with "enhanced" faces.
        """
        if not faces:
            return frame

        enhanced_frame = frame.copy()
        size = self.size
        blend = self.blend if blend is None else blend

        for face in faces:
            try:
                # Align the face to the FFHQ template the model was trained on
                matrix = estimate_ffhq_matrix(face.kps, size)
                if matrix is None:
                    continue
                crop = cv2.warpAffine(enhanced_frame, matrix, (size, size), borderMode=cv2.BORDER_REPLICATE)

                # Pre-processing: BGR -> RGB, values between -1 and 1, NCHW
                face_input = cv2.dnn.blobFromImage(crop, 1.0 / 127.5, (size, size), (127.5, 127.5, 127.5), swapRB=True)

                # Inference
                output = self.session.run(None, {self.input_name: face_input})[0]

                # Post-processing: CHW -> HWC, -1..1 -> 0..255, RGB -> BGR
                output = output[0].transpose(1, 2, 0)
                output = np.clip((output + 1.0) * 127.5 + 0.5, 0, 255).astype(np.uint8)
                output = np.ascontiguousarray(output[:, :, ::-1])

                if blend < 1.0:
                    output = cv2.addWeighted(output, blend, crop, 1.0 - blend, 0)

                # Anatomical mask via BiSeNet (face_parser) on the aligned crop. Failure -> box mask.
                mask = box_mask(size)
                if self.parser is not None:
                    try:
                        region = self.parser.parse(crop)
                        if region.any():  # empty = parser found no face, keep the box mask
                            mask = np.minimum(mask, region)
                    except Exception as e:
                        print(f"[FaceEnhancer] Parser failed, falling back to box mask: {e}")

                paste_back(enhanced_frame, output, mask, matrix)

            except Exception as e:
                print(f"[FaceEnhancer] Error processing face: {e}")
                continue

        return enhanced_frame
