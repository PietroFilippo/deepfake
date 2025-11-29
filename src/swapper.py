import cv2
import insightface
import numpy as np
import os
import time

# Adiciona explicitamente diretórios DLL para Windows (Python 3.8+)
if os.name == 'nt':
    try:
        # Adiciona onnxruntime/capi
        ort_capi = os.path.join(os.path.dirname(os.path.dirname(insightface.__file__)), 'onnxruntime', 'capi')
        if os.path.exists(ort_capi):
            os.add_dll_directory(ort_capi)
        
        # Adiciona o caminho original da lib do TensorRT
        # Detecta automaticamente ou usa variável de ambiente TENSORRT_DIR
        trt_lib = None
        if 'TENSORRT_DIR' in os.environ:
            trt_lib = os.path.join(os.environ['TENSORRT_DIR'], 'lib')
        else:
            # Tenta localizar TensorRT (10.4.0.26) em caminhos comuns (Windows, pode ser modificado)
            common_paths = [
                r"C:\Program Files\TensorRT-10.4.0.26\TensorRT-10.4.0.26\lib",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\lib",
                r"C:\TensorRT\lib",
            ]
            for path in common_paths:
                if os.path.exists(path):
                    trt_lib = path
                    break
        
        if trt_lib and os.path.exists(trt_lib):
            os.add_dll_directory(trt_lib)

        # Adiciona torch/lib (para zlibwapi)
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.exists(torch_lib):
            os.add_dll_directory(torch_lib)
            
        print("Diretórios DLL adicionados ao caminho de busca.")
    except Exception as e:
        print(f"Aviso: Falha ao adicionar diretórios DLL: {e}")

class FaceSwapper:
    def __init__(self, model_path, providers=None, det_size=(320, 320)):
        if providers is None:
            # Prioriza TensorRT, depois CUDA, depois CPU
            providers = [
                ('TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': 'trt_cache',
                    'trt_fp16_enable': True,
                }),
                ('CUDAExecutionProvider', {
                    'cudnn_conv_algo_search': 'HEURISTIC',
                    'arena_extend_strategy': 'kSameAsRequested',
                }),
                'CPUExecutionProvider'
            ]
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
        self.source_face = None

        # Estado para detecção periódica
        self.frame_count = 0
        self.last_faces = []
        
        # Pool de Threads para processamento paralelo
        # Usa até 5 workers para balancear FPS (paralelismo) vs Latência (tamanho do buffer)
        # Mais workers = FPS maior mas mais atraso. 5 é um ponto bom (~200ms de atraso).
        import concurrent.futures
        cpu_count = os.cpu_count() or 4
        self.max_workers = min(cpu_count, 5)
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
        faces_copy = list(self.last_faces) # Cópia rasa da lista é suficiente pois os elementos são tratados como imutáveis aqui
        
        future = self.executor.submit(self._swap_worker, frame_copy, faces_copy, self.source_face)
        return future