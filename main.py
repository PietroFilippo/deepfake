import cv2
import argparse
import time
import os
import sys
import numpy as np
from src.camera import WebcamStream
from src.swapper import FaceSwapper

def main():
    parser = argparse.ArgumentParser(description="Deepfake em Tempo Real")
    parser.add_argument("--source", help="Caminho para imagem de origem", required=True)
    parser.add_argument("--model", help="Caminho para modelo inswapper", default="models/inswapper_128_fp16.onnx")
    parser.add_argument("--max-workers", type=int, default=None, help="Número máximo de threads (workers). Menos = menos latência, Mais = mais FPS.")
    parser.add_argument("--detect-interval", type=int, default=5, help="Intervalo de quadros para detecção de rosto. Maior = mais FPS.")
    parser.add_argument("--camera-fps", type=int, default=30, help="FPS desejado para a webcam.")
    parser.add_argument("--virtual-cam", action="store_true", help="Ativa saída para câmera virtual (OBS Virtual Camera).")
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"Erro: Imagem de origem não encontrada em {args.source}")
        sys.exit(1)

    print("Inicializando.")
    try:
        swapper = FaceSwapper(args.model, max_workers=args.max_workers)
        swapper.set_source_image(args.source)
    except Exception as e:
        print(f"Erro ao inicializar swapper: {e}")
        sys.exit(1)

    print(f"Iniciando webcam com {args.camera_fps} FPS solicitados.")
    webcam = WebcamStream(fps=args.camera_fps).start()

    # Inicializa câmera virtual se solicitado
    vcam = None
    if args.virtual_cam:
        try:
            import pyvirtualcam
            # Tenta inicializar com resolução padrão
            # é usado o valor que foi definido na WebcamStream.
            vcam = pyvirtualcam.Camera(width=1920, height=1080, fps=args.camera_fps, fmt=pyvirtualcam.PixelFormat.BGR)
            print(f"[VirtualCam] Câmera virtual iniciada: {vcam.device}")
        except ImportError:
            print("Erro: 'pyvirtualcam' não instalado. Execute: pip install pyvirtualcam")
            sys.exit(1)
        except Exception as e:
            print(f"Erro ao iniciar câmera virtual: {e}")
            print("Certifique-se que o OBS Studio está instalado (ou outro driver de câmera virtual).")
            vcam = None

    print("Pressione 'q' para sair.")
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    from collections import deque
    # Buffer para armazenar futures pendentes. 
    # Tamanho = workers + 1 minimiza latência enquanto mantém workers ocupados.
    # Acessa swapper.max_workers que foi adicionado ao swapper.py
    buffer_size = getattr(swapper, 'max_workers', 4) + 1
    pending_futures = deque(maxlen=buffer_size)
    print(f"[Main] Tamanho do buffer de quadros: {buffer_size}")
    
    while True:
        frame = webcam.read()
        if frame is None:
            continue
            
        # 1. Envia quadro para o pool de workers
        try:
            future = swapper.process_frame_async(frame, detect_interval=args.detect_interval)
            pending_futures.append(future)
        except Exception as e:
            print(f"Erro de envio: {e}")
            continue

        # 2. Recupera resultado se o buffer estiver cheio (ou apenas para manter o fluxo)
        # Aguarda o quadro mais antigo ficar pronto.
        if len(pending_futures) >= buffer_size - 1:
            try:
                # Remove o future mais antigo e aguarda por ele
                output = pending_futures.popleft().result()
            except Exception as e:
                print(f"Erro de processamento: {e}")
                output = frame # Fallback
        else:
            # Buffer enchendo
            # Espera um pouco para encher o pipeline ou mostra quadro bruto para evitar congelamento inicial
            # Continua enchendo o buffer pelos primeiros quadros
            continue

        # Envia para câmera virtual
        if vcam:
            try:
                # Garante que o tamanho do frame corresponde ao esperado pela câmera virtual
                if output.shape[1] != vcam.width or output.shape[0] != vcam.height:
                    output_vcam = cv2.resize(output, (vcam.width, vcam.height))
                    vcam.send(output_vcam)
                else:
                    vcam.send(output)
                    
                vcam.sleep_until_next_frame()
            except Exception as e:
                print(f"Erro na câmera virtual: {e}")

        # Cálculo de FPS
        fps_frame_count += 1
        if time.time() - fps_start_time > 1.0:
            fps = fps_frame_count / (time.time() - fps_start_time)
            fps_frame_count = 0
            fps_start_time = time.time()
        
        cv2.putText(output, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Deepfake Real-time", output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    webcam.stop()
    if vcam:
        vcam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()