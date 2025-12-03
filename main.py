import cv2
import argparse
import time
import os
import sys
import glob
import numpy as np
import pyaudio
import wave
import threading
from src.camera import WebcamStream, VideoFileStream
from src.swapper import FaceSwapper

# Tenta importar moviepy para processamento de vídeo com áudio
try:
    from moviepy import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError as e:
    MOVIEPY_AVAILABLE = False
    print(f"Aviso: 'moviepy' não pôde ser importado: {e}")
    print("Processamento de vídeo será sem áudio.")
except Exception as e:
    MOVIEPY_AVAILABLE = False
    print(f"Aviso: Erro inesperado ao importar 'moviepy': {e}")
    print("Processamento de vídeo será sem áudio.")

class AudioRecorder:
    def __init__(self):
        self.open = True
        self.rate = 44100
        self.frames_per_buffer = 1024
        self.channels = 1
        self.format = pyaudio.paInt16
        self.audio_filename = "temp_audio.wav"
        self.audio = None
        self.stream = None
        self.audio_frames = []
        
        try:
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(format=self.format,
                                          channels=self.channels,
                                          rate=self.rate,
                                          input=True,
                                          frames_per_buffer=self.frames_per_buffer)
        except Exception as e:
            print(f"Erro ao inicializar áudio: {e}")
            self.open = False

    def record(self):
        if not self.open or not self.stream:
            return
            
        self.stream.start_stream()
        while self.open:
            try:
                data = self.stream.read(self.frames_per_buffer)
                self.audio_frames.append(data)
            except Exception:
                break

    def stop(self):
        self.open = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        
        if self.audio_frames:
            waveFile = wave.open(self.audio_filename, 'wb')
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(self.audio.get_sample_size(self.format))
            waveFile.setframerate(self.rate)
            waveFile.writeframes(b''.join(self.audio_frames))
            waveFile.close()
            return self.audio_filename
        return None

    def start(self):
        if self.open:
            audio_thread = threading.Thread(target=self.record)
            audio_thread.start()

def main():
    parser = argparse.ArgumentParser(description="Deepfake em Tempo Real")
    parser.add_argument("--source", help="Caminho para imagem de origem inicial", required=True)
    parser.add_argument("--model", help="Caminho para modelo inswapper", default="models/inswapper_128_fp16.onnx")
    parser.add_argument("--max-workers", type=int, default=None, help="Número máximo de threads (workers). Menos = menos latência, Mais = mais FPS.")
    parser.add_argument("--detect-interval", type=int, default=5, help="Intervalo de quadros para detecção de rosto. Maior = mais FPS.")
    parser.add_argument("--camera-fps", type=int, default=30, help="FPS desejado para a webcam.")
    parser.add_argument("--virtual-cam", action="store_true", help="Ativa saída para câmera virtual (OBS Virtual Camera).")
    parser.add_argument("--video", help="Caminho para arquivo de vídeo de destino")
    parser.add_argument("--image", help="Caminho para imagem de destino")
    parser.add_argument("--out", help="Caminho para salvar o vídeo gravado/processado")
    parser.add_argument("--enhance", action="store_true", help="Ativa melhoria de rosto (GFPGAN) por padrão")
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
        if args.enhance:
            swapper.enhancement_enabled = True
            print("Enhancer ativado por padrão.")
    except Exception as e:
        print(f"Erro ao inicializar swapper: {e}")
        sys.exit(1)

    # Modo de Imagem Estática
    if args.image:
        print(f"Processando imagem única: {args.image}")
        target_img = cv2.imread(args.image)
        if target_img is None:
            print("Erro: Não foi possível ler a imagem de destino.")
            sys.exit(1)
        
        # Detecta rostos na imagem alvo
        faces = swapper.app.get(target_img)
        res = target_img.copy()
        
        # Realiza a troca
        try:
            res = swapper._swap_worker(target_img, faces, swapper.source_face)
        except Exception as e:
            print(f"Erro na troca: {e}")
            
        cv2.imshow("Deepfake Image", res)
        
        # Lógica de auto-save
        if not os.path.exists("outputs"):
            os.makedirs("outputs")

        if args.out:
            out_path = args.out
        else:
            source_name = os.path.splitext(os.path.basename(image_files[current_image_index]))[0]
            target_name = os.path.splitext(os.path.basename(args.image))[0]
            filename = f"processed_{target_name}_with_{source_name}_{int(time.time())}.jpg"
            out_path = os.path.join("outputs", filename)

        cv2.imwrite(out_path, res)
        print(f"Salvo em: {out_path}")
            
        print("Pressione qualquer tecla para sair.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Modo Vídeo Offline
    if args.video:
        print(f"Processando vídeo: {args.video}")
        if not os.path.exists(args.video):
            print("Erro: Arquivo de vídeo não encontrado.")
            sys.exit(1)
            
        if not os.path.exists("outputs"):
            os.makedirs("outputs")
            
        if args.out:
            filename = args.out
        else:
            video_name = os.path.splitext(os.path.basename(args.video))[0]
            source_name = os.path.splitext(os.path.basename(image_files[current_image_index]))[0]
            filename = f"processed_{video_name}_with_{source_name}_{int(time.time())}.mp4"

        if not os.path.isabs(filename) and not args.out:
             out_path = os.path.join("outputs", filename)
        else:
             out_path = filename

        # Se moviepy estiver disponível, usa ele para preservar áudio
        if MOVIEPY_AVAILABLE:
            print("Usando MoviePy para processamento com áudio.")
            try:
                import tempfile
                import subprocess
                
                clip = VideoFileClip(args.video)
                fps = clip.fps
                width, height = clip.size
                
                # Cria arquivo temporário para vídeo sem áudio
                temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
                
                # Processa frames manualmente usando OpenCV writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
                
                frame_count = 0
                total_frames = int(clip.duration * fps)
                start_time = time.time()
                
                print(f"Processando {total_frames} frames.")
                
                # Itera sobre os frames
                for frame_rgb in clip.iter_frames():
                    # MoviePy usa RGB, OpenCV usa BGR
                    frame_rgb = np.array(frame_rgb)
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    
                    # Detecta e troca faces
                    faces = swapper._detect_faces_downscale(frame_bgr, scale=0.5)
                    try:
                        res_bgr = swapper._swap_worker(frame_bgr, faces, swapper.source_face)
                    except Exception:
                        res_bgr = frame_bgr
                    
                    out_writer.write(res_bgr)
                    
                    frame_count += 1
                    if frame_count % 10 == 0:
                        elapsed = time.time() - start_time
                        fps_proc = frame_count / elapsed if elapsed > 0 else 0
                        progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                        print(f"\rProcessando: {progress:.1f}% | FPS: {fps_proc:.2f} | Frame: {frame_count}/{total_frames}", end="")
                
                print()  # Nova linha
                out_writer.release()
                
                # Combina vídeo processado com áudio original usando ffmpeg
                print("Combinando com áudio original.")
                try:
                    subprocess.run([
                        'ffmpeg', '-y', '-i', temp_video, '-i', args.video,
                        '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
                        out_path
                    ], check=True, capture_output=True)
                    print(f"Concluído. Salvo em: {out_path}")
                except subprocess.CalledProcessError:
                    # Se ffmpeg falhar, usa o vídeo sem áudio
                    print("Aviso: ffmpeg falhou. Salvando sem áudio.")
                    import shutil
                    shutil.move(temp_video, out_path)
                    print(f"Salvo sem áudio em: {out_path}")
                except FileNotFoundError:
                    print("Aviso: ffmpeg não encontrado. Salvando sem áudio.")
                    import shutil
                    shutil.move(temp_video, out_path)
                    print(f"Salvo sem áudio em: {out_path}")
                finally:
                    # Limpa arquivo temporário se ainda existir
                    if os.path.exists(temp_video) and temp_video != out_path:
                        os.remove(temp_video)
                
                clip.close()
                return
            except Exception as e:
                print(f"Erro com MoviePy: {e}")
                print("Tentando fallback para OpenCV (sem áudio).")


        # Fallback para OpenCV (Sem áudio)
        cap = cv2.VideoCapture(args.video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"Salvando em: {out_path} (SEM ÁUDIO)")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            faces = swapper._detect_faces_downscale(frame, scale=0.5)
            try:
                res = swapper._swap_worker(frame, faces, swapper.source_face)
            except Exception as e:
                res = frame
                
            out.write(res)
            
            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps_proc = frame_count / elapsed if elapsed > 0 else 0
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"\rProcessando: {progress:.1f}% | FPS: {fps_proc:.2f} | Frame: {frame_count}/{total_frames}", end="")
                
        print("\nConcluído.")
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return

    # Modo Webcam (Real-time)
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
    print("  'r': Iniciar/Parar gravação")
    print("  'u': Mostrar/Ocultar Interface (UI)")
    
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    swap_enabled = True
    show_ui = True
    
    from collections import deque
    # Buffer para armazenar futures pendentes. 
    # Tamanho = workers + 1 minimiza latência enquanto mantém workers ocupados.
    # Acessa swapper.max_workers que foi adicionado ao swapper.py
    buffer_size = getattr(swapper, 'max_workers', 4) + 1
    pending_futures = deque(maxlen=buffer_size)
    print(f"[Main] Tamanho do buffer de quadros: {buffer_size}")
    
    current_image_name = os.path.basename(image_files[current_image_index])

    # Estado de gravação
    recording = False
    video_writer = None
    audio_recorder = None
    recording_start_time = 0
    recording_frame_count = 0

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
        if show_ui:
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
        
        # Status de gravação
        if recording:
            if show_ui:
                cv2.circle(output, (50, 150), 10, (0, 0, 255), -1)
                cv2.putText(output, "REC", (70, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if video_writer is None:
                # Inicia gravação de áudio
                audio_recorder = AudioRecorder()
                audio_recorder.start()

                # Cria pasta outputs se não existir
                if not os.path.exists("outputs"):
                    os.makedirs("outputs")
                
                # Usa mp4v para melhor compatibilidade com áudio aac
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                filename = args.out if args.out else f"output_{int(time.time())}.mp4"
                # Se não for caminho absoluto, salva em outputs/
                if not os.path.isabs(filename) and not args.out:
                     out_path = os.path.join("outputs", filename)
                else:
                     out_path = filename
                     
                video_writer = cv2.VideoWriter(out_path, fourcc, 30.0, (output.shape[1], output.shape[0]))
                print(f"Gravando em: {out_path}")
                recording_start_time = time.time()
                recording_frame_count = 0
            
            video_writer.write(output)
            recording_frame_count += 1
        elif video_writer:
            video_writer.release()
            video_writer = None            
            # Para gravação de áudio e combina
            if audio_recorder:
                print("Processando áudio.")
                temp_audio = audio_recorder.stop()
                
                if temp_audio and os.path.exists(temp_audio):
                    print("Combinando áudio e vídeo.")
                    # Cria nome para arquivo temporário de vídeo
                    temp_video = out_path.replace(".mp4", "_temp.mp4")
                    if temp_video == out_path:
                        temp_video = out_path + "_temp.mp4"
                    
                    try:
                        if os.path.exists(out_path):
                            # Renomeia vídeo original para temp
                            # Se arquivo temp já existe, remove antes
                            if os.path.exists(temp_video):
                                os.remove(temp_video)
                                
                            os.rename(out_path, temp_video)
                            
                            # Calcula FPS real
                            elapsed_recording = time.time() - recording_start_time
                            actual_fps = recording_frame_count / elapsed_recording if elapsed_recording > 0 else 30.0
                            print(f"FPS real da gravação: {actual_fps:.2f}")

                            # Combina usando ffmpeg com re-encode para ajustar FPS
                            import subprocess
                            subprocess.run([
                                'ffmpeg', '-y', 
                                '-r', f'{actual_fps:.2f}', 
                                '-i', temp_video, 
                                '-i', temp_audio,
                                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
                                '-c:a', 'aac', 
                                '-map', '0:v:0', '-map', '1:a:0',
                                out_path
                            ], check=True, capture_output=True)
                            
                            print(f"Gravação concluída com áudio: {out_path}")
                            
                            # Limpa arquivos temporários
                            if os.path.exists(temp_video):
                                os.remove(temp_video)
                            if os.path.exists(temp_audio):
                                os.remove(temp_audio)
                        else:
                            print("Erro: Arquivo de vídeo não encontrado para merge.")
                    except Exception as e:
                        print(f"Erro ao combinar áudio: {e}")
                        # Tenta restaurar vídeo original se falhar
                        if os.path.exists(temp_video) and not os.path.exists(out_path):
                            os.rename(temp_video, out_path)
                else:
                    print("Áudio não gravado ou erro ao salvar.")
                
                audio_recorder = None
        
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
        elif key == ord('r'):
            recording = not recording
            print(f"Gravação: {'Iniciada' if recording else 'Parada'}")
        elif key == ord('u'):
            show_ui = not show_ui
            print(f"Interface: {'Visível' if show_ui else 'Oculta'}")
            
    webcam.stop()
    if vcam:
        vcam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()