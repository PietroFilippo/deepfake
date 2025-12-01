import cv2
import argparse
import time
import os
import sys
import glob
import numpy as np
from src.camera import WebcamStream
from src.swapper import FaceSwapper

def main():
    parser = argparse.ArgumentParser(description="Deepfake em Tempo Real")
    parser.add_argument("--source", help="Caminho para imagem de origem inicial", required=True)
    parser.add_argument("--model", help="Caminho para modelo inswapper", default="models/inswapper_128_fp16.onnx")
    parser.add_argument("--max-workers", type=int, default=None, help="Número máximo de threads (workers). Menos = menos latência, Mais = mais FPS.")
    parser.add_argument("--detect-interval", type=int, default=5, help="Intervalo de quadros para detecção de rosto. Maior = mais FPS.")
    parser.add_argument("--camera-fps", type=int, default=30, help="FPS desejado para a webcam.")
    parser.add_argument("--virtual-cam", action="store_true", help="Ativa saída para câmera virtual (OBS Virtual Camera).")
    args = parser.parse_args()

    # Procura imagens no diretório
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    images_dir = os.path.dirname(os.path.abspath(args.source))
    if not images_dir: # Lida com caso onde source é apenas um nome de arquivo
        images_dir = "images"
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    
    # Ordena arquivos para ordem consistente
    image_files.sort()
    
    if not image_files:
        print(f"Aviso: Nenhuma imagem encontrada em {images_dir}. A troca dinâmica não funcionará.")
        image_files = [args.source]

    # Encontra índice inicial
    try:
        current_image_index = image_files.index(os.path.abspath(args.source))
    except ValueError:
        # Tenta correspondência de caminho relativo
        try:
             current_image_index = image_files.index(os.path.join(images_dir, os.path.basename(args.source)))
        except ValueError:
             # Apenas adiciona se não encontrado (por exemplo, fora da pasta)
             if os.path.exists(args.source):
                 image_files.append(args.source)
                 current_image_index = len(image_files) - 1
             else:
                 print(f"Erro: Imagem de origem não encontrada em {args.source}")
                 sys.exit(1)

    print(f"Imagens disponíveis: {len(image_files)}")
    print("Inicializando.")
    try:
        swapper = FaceSwapper(args.model, max_workers=args.max_workers)
        swapper.set_source_image(image_files[current_image_index])
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

    print("Controles:")
    print("  'q': Sair")
    print("  'n': Próxima imagem")
    print("  'p': Imagem anterior")
    print("  'x': Ativar/Desativar troca")
    print("  'e': Ativar/Desativar melhoria de rosto (GFPGAN)")
    
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    swap_enabled = True
    
    from collections import deque
    # Buffer para armazenar futures pendentes. 
    # Tamanho = workers + 1 minimiza latência enquanto mantém workers ocupados.
    # Acessa swapper.max_workers que foi adicionado ao swapper.py
    buffer_size = getattr(swapper, 'max_workers', 4) + 1
    pending_futures = deque(maxlen=buffer_size)
    print(f"[Main] Tamanho do buffer de quadros: {buffer_size}")
    
    current_image_name = os.path.basename(image_files[current_image_index])

    while True:
        frame = webcam.read()
        if frame is None:
            continue
            
        if swap_enabled:
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
                    output = frame # Fallback se houver erro
            else:
                # Buffer enchendo
                # Espera um pouco para encher o pipeline ou mostra quadro bruto para evitar congelamento inicial
                # Continua enchendo o buffer pelos primeiros quadros
                continue
        else:
            # Se desativado, limpa o buffer para não mostrar frames antigos ao reativar
            if pending_futures:
                pending_futures.clear()
            output = frame

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

        # UI Overlay
        cv2.putText(output, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(output, f"Img: {current_image_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        status_color = (0, 255, 0) if swap_enabled else (0, 0, 255)
        status_text = "ON" if swap_enabled else "OFF"
        cv2.putText(output, f"Status: {status_text}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Status do enhancer
        enh_enabled = getattr(swapper, 'enhancement_enabled', False)
        enh_color = (0, 255, 0) if enh_enabled else (0, 0, 255)
        enh_text = "ON" if enh_enabled else "OFF"
        cv2.putText(output, f"Enhance: {enh_text}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, enh_color, 2)
        
        cv2.imshow("Deepfake Real-time", output)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_image_index = (current_image_index + 1) % len(image_files)
            new_image = image_files[current_image_index]
            print(f"Trocando para: {new_image}")
            try:
                swapper.set_source_image(new_image)
                current_image_name = os.path.basename(new_image)
                # Limpa buffer para evitar mistura de rostos antigos
                pending_futures.clear() 
            except Exception as e:
                print(f"Erro ao trocar imagem: {e}")
        elif key == ord('p'):
            current_image_index = (current_image_index - 1) % len(image_files)
            new_image = image_files[current_image_index]
            print(f"Trocando para: {new_image}")
            try:
                swapper.set_source_image(new_image)
                current_image_name = os.path.basename(new_image)
                pending_futures.clear()
            except Exception as e:
                print(f"Erro ao trocar imagem: {e}")
        elif key == ord('x'):
            swap_enabled = not swap_enabled
            print(f"Troca de rosto: {'Ativado' if swap_enabled else 'Desativado'}")
        elif key == ord('e'):
            if hasattr(swapper, 'toggle_enhancer'):
                is_enabled = swapper.toggle_enhancer()
                print(f"Enhancer: {'Ativado' if is_enabled else 'Desativado'}")
            else:
                print("Enhancer não disponível.")
            
    webcam.stop()
    if vcam:
        vcam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()