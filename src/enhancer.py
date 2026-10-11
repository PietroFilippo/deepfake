import cv2
import numpy as np
import onnxruntime
import os
from .utils import get_default_providers
from .face_parser import FaceParser

class FaceEnhancer:
    def __init__(self, model_path="models/GFPGANv1.4.onnx", providers=None):
        # Tenta TensorRT em modo FP32 (precisão total) para performance sem artefatos
        if providers is None:
            providers = [
                ('TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': 'trt_cache',
                    'trt_fp16_enable': False, # desativa FP16 para evitar face borrada
                }),
                ('CUDAExecutionProvider', {
                    'cudnn_conv_algo_search': 'HEURISTIC',
                    'arena_extend_strategy': 'kSameAsRequested',
                }),
                'CPUExecutionProvider'
            ]
        self.providers = providers

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo GFPGAN não encontrado em {model_path}")

        print(f"[FaceEnhancer] Carregando modelo GFPGAN de {model_path}")
        try:
            self.session = onnxruntime.InferenceSession(model_path, providers=self.providers)
        except Exception as e:
            print(f"[FaceEnhancer] Erro ao carregar modelo: {e}")
            raise

        self.input_name = self.session.get_inputs()[0].name
        print("[FaceEnhancer] Modelo carregado com sucesso.")

        # Parser de rosto para máscara anatômica (substitui o círculo gaussiano).
        # Falha silenciosa -> cai no fallback de círculo.
        try:
            self.parser = FaceParser()
        except Exception as e:
            print(f"[FaceEnhancer] Aviso: FaceParser indisponível ({e}). Usando máscara circular.")
            self.parser = None

    def _circle_mask(self, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        radius = int(min(h, w) * 0.45)
        cv2.circle(mask, center, radius, 255, -1)
        mask = mask.astype(np.float32) / 255.0
        return cv2.GaussianBlur(mask, (0, 0), sigmaX=min(h, w) * 0.05)

    def enhance(self, frame, faces):
        """
        Melhora a qualidade dos rostos detectados no quadro.
        
        Args:
            frame: Imagem BGR (numpy array).
            faces: Lista de objetos de rosto (InsightFace) que foram trocados.
            
        Returns:
            frame: Imagem com rostos "melhorados".
        """
        if not faces:
            return frame
            
        enhanced_frame = frame.copy()
        
        for face in faces:
            # Obtém bounding box
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Adiciona margem para capturar o rosto inteiro e um pouco do contexto
            # Margem de segurança
            h, w = frame.shape[:2]
            pad_x = int((x2 - x1) * 0.5)
            pad_y = int((y2 - y1) * 0.5)
            
            x1_p = max(0, x1 - pad_x)
            y1_p = max(0, y1 - pad_y)
            x2_p = min(w, x2 + pad_x)
            y2_p = min(h, y2 + pad_y)
            
            face_img = frame[y1_p:y2_p, x1_p:x2_p]
            
            if face_img.size == 0:
                continue
                
            # Pre-processamento
            # espera entrada 512x512, valores entre -1 e 1, RGB
            try:
                face_input = cv2.resize(face_img, (512, 512))
                face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB) # BGR -> RGB
                face_input = face_input.astype(np.float32) / 255.0
                face_input = (face_input - 0.5) / 0.5
                face_input = np.transpose(face_input, (2, 0, 1)) # HWC -> CHW

                face_input = np.expand_dims(face_input, axis=0) # Adiciona batch dimension
                
                # Inferência
                output = self.session.run(None, {self.input_name: face_input})[0]
                
                # Post-processamento
                output = output.squeeze(0).transpose(1, 2, 0) # CHW -> HWC
                output = (output * 0.5 + 0.5) * 255.0
                output = np.clip(output, 0, 255).astype(np.uint8)
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR) # RGB -> BGR
                
                # Redimensiona de volta para o tamanho do crop original
                h_orig, w_orig = face_img.shape[:2]
                output_resized = cv2.resize(output, (w_orig, h_orig))

                # Máscara anatômica via BiseNet (face_parser). Falha -> círculo.
                mask = None
                if self.parser is not None:
                    try:
                        mask = self.parser.parse(face_img)
                        if not mask.any():
                            mask = None  # parser não achou rosto, usa fallback
                    except Exception as e:
                        print(f"[FaceEnhancer] Parser falhou, fallback p/ círculo: {e}")
                        mask = None
                if mask is None:
                    mask = self._circle_mask(h_orig, w_orig)

                mask = np.expand_dims(mask, axis=2)

                # Combinar
                enhanced_frame[y1_p:y2_p, x1_p:x2_p] = (output_resized * mask + face_img * (1 - mask)).astype(np.uint8)
                
            except Exception as e:
                print(f"[FaceEnhancer] Erro ao processar rosto: {e}")
                continue
                
        return enhanced_frame
