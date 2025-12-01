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

        print(f"[FaceSwapper] Inicializando FaceAnalysis com providers: {self.providers} e det_size: {self.det_size}")
        
        # Aplicativo de detecção de rostos
        self.app = insightface.app.FaceAnalysis(name='buffalo_l', providers=self.providers)
        self.app.prepare(ctx_id=0, det_size=self.det_size)

        # Modelo de troca de rostos
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em {model_path}")
            
        if not os.path.exists('trt_cache'):
            os.makedirs('trt_cache')
            
        self.swapper = insightface.model_zoo.get_model(model_path, providers=self.providers)
        
        # Inicializa Enhancer (GFPGAN)
        try:
            self.enhancer = FaceEnhancer()
            self.enhancement_enabled = False # Desativado por padrão
        except Exception as e:
            print(f"[FaceSwapper] Aviso: Não foi possível carregar FaceEnhancer: {e}")
            self.enhancer = None
            self.enhancement_enabled = False

        self.source_face = None

        # Estado para detecção periódica
        self.frame_count = 0
        self.last_faces = []
        
        # Pool de Threads para processamento paralelo
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            self.max_workers = min(cpu_count, 5)
        else:
            self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        print(f"[FaceSwapper] Inicializado com {self.max_workers} threads de trabalho.")

    def set_source_image(self, source_img_path):
        img = cv2.imread(source_img_path)
        if img is None:
            raise ValueError(f"Não foi possível ler a imagem de origem: {source_img_path}")

        faces = self.app.get(img)
        if len(faces) == 0:
            raise ValueError("Nenhum rosto detectado na imagem de origem")

        # Usa o maior rosto
        self.source_face = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0])*(x.bbox[3] - x.bbox[1]))[-1]
        print("[FaceSwapper] Rosto de origem definido")

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
        
        # Aplica melhoria se ativado e disponível
        if self.enhancer and self.enhancement_enabled:
            try:
                res = self.enhancer.enhance(res, faces)
            except Exception as e:
                print(f"[FaceSwapper] Erro no enhancer: {e}")
                
        return res

    def process_frame_async(self, frame, detect_interval=5):
        if self.source_face is None:
            # Retorna um future completo com o quadro original
            future = concurrent.futures.Future()
            future.set_result(frame)
            return future

        # Detecta rostos periodicamente (Síncrono para manter o estado simples)
        if self.frame_count % detect_interval == 0:
            try:
                faces = self._detect_faces_downscale(frame, scale=0.5)
                self.last_faces = faces
            except Exception as e:
                print(f"Erro de detecção: {e}")
                pass
        self.frame_count += 1

        # Envia tarefa de troca para o pool de threads
        # Copia quadro e rostos para garantir segurança de thread
        frame_copy = frame.copy()
        faces_copy = list(self.last_faces) 
        
        future = self.executor.submit(self._swap_worker, frame_copy, faces_copy, self.source_face)
        return future

    def toggle_enhancer(self):
        if self.enhancer:
            self.enhancement_enabled = not self.enhancement_enabled
            return self.enhancement_enabled
        return False