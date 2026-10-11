import cv2
import argparse
import time
import os
import sys
import glob
import numpy as np
import threading
from src.camera import WebcamStream, VideoFileStream
from src.swapper import FaceSwapper

try:
    import pyaudio
    import wave
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: 'pyaudio' not found. Audio recording disabled.")

# Try to import moviepy for video processing with audio
try:
    from moviepy import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError as e:
    MOVIEPY_AVAILABLE = False
    print(f"Warning: 'moviepy' could not be imported: {e}")
    print("Video processing will have no audio.")
except Exception as e:
    MOVIEPY_AVAILABLE = False
    print(f"Warning: Unexpected error while importing 'moviepy': {e}")
    print("Video processing will have no audio.")

class AudioRecorder:
    def __init__(self):
        self.open = True
        self.rate = 44100
        self.frames_per_buffer = 1024
        self.channels = 1
        self.format = pyaudio.paInt16 if AUDIO_AVAILABLE else None
        self.audio_filename = "temp_audio.wav"
        self.audio = None
        self.stream = None
        self.audio_frames = []

        if not AUDIO_AVAILABLE:
            self.open = False
            return

        try:
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(format=self.format,
                                          channels=self.channels,
                                          rate=self.rate,
                                          input=True,
                                          frames_per_buffer=self.frames_per_buffer)
        except Exception as e:
            print(f"Error initializing audio: {e}")
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
    parser = argparse.ArgumentParser(description="Real-time Deepfake")
    parser.add_argument("--source", help="Path to the initial source image", required=True)
    parser.add_argument("--model", help="Path to the inswapper model", default="models/inswapper_128_fp16.onnx")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum number of threads (workers). Fewer = less latency, more = more FPS.")
    parser.add_argument("--detect-interval", type=int, default=5, help="Frame interval for face detection. Higher = more FPS.")
    parser.add_argument("--camera-fps", type=int, default=30, help="Desired webcam FPS.")
    parser.add_argument("--virtual-cam", action="store_true", help="Enable output to a virtual camera (OBS Virtual Camera).")
    parser.add_argument("--video", help="Path to the target video file")
    parser.add_argument("--image", help="Path to the target image")
    parser.add_argument("--gif", help="Path to the target GIF file")
    parser.add_argument("--out", help="Path to save the recorded/processed video")
    parser.add_argument("--enhance", action="store_true", help="Enable face enhancement (GFPGAN) by default")
    args = parser.parse_args()

    # Look for images in the directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    images_dir = os.path.dirname(os.path.abspath(args.source))
    if not images_dir: # Handles the case where source is just a file name
        images_dir = "images"

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))

    # Sort files for a consistent order
    image_files.sort()

    if not image_files:
        print(f"Warning: No images found in {images_dir}. Dynamic switching will not work.")
        image_files = [args.source]

    # Find the starting index
    try:
        current_image_index = image_files.index(os.path.abspath(args.source))
    except ValueError:
        # Try matching the relative path
        try:
             current_image_index = image_files.index(os.path.join(images_dir, os.path.basename(args.source)))
        except ValueError:
             # Just append it if not found (e.g. outside the folder)
             if os.path.exists(args.source):
                 image_files.append(args.source)
                 current_image_index = len(image_files) - 1
             else:
                 print(f"Error: Source image not found at {args.source}")
                 sys.exit(1)

    print(f"Available images: {len(image_files)}")
    print("Initializing.")
    try:
        swapper = FaceSwapper(args.model, max_workers=args.max_workers)
        swapper.set_source_image(image_files[current_image_index])
        if args.enhance:
            swapper.enhancement_enabled = True
            print("Enhancer enabled by default.")
    except Exception as e:
        print(f"Error initializing swapper: {e}")
        sys.exit(1)

    # Static Image Mode
    if args.image:
        print(f"Processing single image: {args.image}")
        target_img = cv2.imread(args.image)
        if target_img is None:
            print("Error: Could not read the target image.")
            sys.exit(1)

        # Detect faces in the target image
        faces = swapper.app.get(target_img)
        res = target_img.copy()

        # Perform the swap
        try:
            res = swapper._swap_worker(target_img, faces, swapper.source_face)
        except Exception as e:
            print(f"Swap error: {e}")

        cv2.imshow("Deepfake Image", res)

        # Auto-save logic
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
        print(f"Saved to: {out_path}")

        print("Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Offline Video Mode
    if args.video:
        print(f"Processing video: {args.video}")
        if not os.path.exists(args.video):
            print("Error: Video file not found.")
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

        # If moviepy is available, use it to preserve audio
        if MOVIEPY_AVAILABLE:
            print("Using MoviePy for processing with audio.")
            try:
                import tempfile
                import subprocess

                clip = VideoFileClip(args.video)
                fps = clip.fps
                width, height = clip.size

                # Create a temporary file for the video without audio
                temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

                # Process frames manually using the OpenCV writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

                frame_count = 0
                total_frames = int(clip.duration * fps)
                start_time = time.time()

                print(f"Processing {total_frames} frames.")

                # Iterate over the frames
                for frame_rgb in clip.iter_frames():
                    # MoviePy uses RGB, OpenCV uses BGR
                    frame_rgb = np.array(frame_rgb)
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                    # Detect and swap faces
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
                        print(f"\rProcessing: {progress:.1f}% | FPS: {fps_proc:.2f} | Frame: {frame_count}/{total_frames}", end="")

                print()  # New line
                out_writer.release()

                # Merge the processed video with the original audio using ffmpeg
                print("Merging with the original audio.")
                try:
                    subprocess.run([
                        'ffmpeg', '-y', '-i', temp_video, '-i', args.video,
                        '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
                        out_path
                    ], check=True, capture_output=True)
                    print(f"Done. Saved to: {out_path}")
                except subprocess.CalledProcessError:
                    # If ffmpeg fails, use the video without audio
                    print("Warning: ffmpeg failed. Saving without audio.")
                    import shutil
                    shutil.move(temp_video, out_path)
                    print(f"Saved without audio to: {out_path}")
                except FileNotFoundError:
                    print("Warning: ffmpeg not found. Saving without audio.")
                    import shutil
                    shutil.move(temp_video, out_path)
                    print(f"Saved without audio to: {out_path}")
                finally:
                    # Clean up the temporary file if it still exists
                    if os.path.exists(temp_video) and temp_video != out_path:
                        os.remove(temp_video)

                clip.close()
                return
            except Exception as e:
                print(f"MoviePy error: {e}")
                print("Falling back to OpenCV (no audio).")


        # Fallback to OpenCV (no audio)
        cap = cv2.VideoCapture(args.video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"Saving to: {out_path} (NO AUDIO)")

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
                print(f"\rProcessing: {progress:.1f}% | FPS: {fps_proc:.2f} | Frame: {frame_count}/{total_frames}", end="")

        print("\nDone.")
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return

    # GIF Mode
    if args.gif:
        print(f"Processing GIF: {args.gif}")
        if not os.path.exists(args.gif):
            print("Error: GIF file not found.")
            sys.exit(1)

        if not os.path.exists("outputs"):
            os.makedirs("outputs")

        if args.out:
            filename = args.out
        else:
            gif_name = os.path.splitext(os.path.basename(args.gif))[0]
            source_name = os.path.splitext(os.path.basename(image_files[current_image_index]))[0]
            filename = f"processed_{gif_name}_with_{source_name}_{int(time.time())}.gif"

        if not os.path.isabs(filename) and not args.out:
             out_path = os.path.join("outputs", filename)
        else:
             out_path = filename

        try:
            import imageio

            if MOVIEPY_AVAILABLE:
                clip = VideoFileClip(args.gif)
                # Try to get the FPS; if missing (common in some GIFs), assume 15
                fps = clip.fps if clip.fps else 15

                frames = []
                # Frame count estimate
                total_frames = int(clip.duration * fps) if clip.duration else "Unknown"
                print(f"Reading GIF.")

                frame_count = 0
                for frame_rgb in clip.iter_frames():
                    # MoviePy returns RGB
                    frame_rgb = np.array(frame_rgb)
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                    # Detect and swap faces
                    faces = swapper._detect_faces_downscale(frame_bgr, scale=0.5)
                    try:
                        res_bgr = swapper._swap_worker(frame_bgr, faces, swapper.source_face)
                    except Exception:
                        res_bgr = frame_bgr

                    # Convert back to RGB for the GIF
                    res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
                    frames.append(res_rgb)

                    frame_count += 1
                    print(f"\rProcessing frame: {frame_count}", end="")

                print("\nSaving GIF.")
                # Save using imageio
                imageio.mimsave(out_path, frames, fps=fps, loop=0)
                print(f"Saved to: {out_path}")
                clip.close()
            else:
                print("Error: MoviePy is required to process GIFs.")
                sys.exit(1)

        except Exception as e:
            print(f"Error processing GIF: {e}")

        return

    # Webcam Mode (Real-time)
    print(f"Starting webcam with {args.camera_fps} FPS requested.")
    webcam = WebcamStream(fps=args.camera_fps).start()

    # Initialize the virtual camera if requested
    vcam = None
    if args.virtual_cam:
        try:
            import pyvirtualcam
            # Try to initialize with the default resolution
            # (uses the value defined in WebcamStream).
            vcam = pyvirtualcam.Camera(width=1920, height=1080, fps=args.camera_fps, fmt=pyvirtualcam.PixelFormat.BGR)
            print(f"[VirtualCam] Virtual camera started: {vcam.device}")
        except ImportError:
            print("Error: 'pyvirtualcam' is not installed. Run: pip install pyvirtualcam")
            sys.exit(1)
        except Exception as e:
            print(f"Error starting virtual camera: {e}")
            print("Make sure OBS Studio is installed (or another virtual camera driver).")
            vcam = None

    print("Controls:")
    print("  'q': Quit")
    print("  'n': Next image")
    print("  'p': Previous image")
    print("  'x': Toggle swap on/off")
    print("  'e': Toggle face enhancement (GFPGAN) on/off")
    print("  'r': Start/Stop recording")
    print("  'u': Show/Hide interface (UI)")

    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    swap_enabled = True
    show_ui = True

    from collections import deque
    # Buffer to hold pending futures.
    # Size = workers + 1 minimizes latency while keeping the workers busy.
    # Reads swapper.max_workers, which was added to swapper.py
    buffer_size = getattr(swapper, 'max_workers', 4) + 1
    pending_futures = deque(maxlen=buffer_size)
    print(f"[Main] Frame buffer size: {buffer_size}")

    current_image_name = os.path.basename(image_files[current_image_index])

    # Recording state
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
            # 1. Submit the frame to the worker pool
            try:
                future = swapper.process_frame_async(frame, detect_interval=args.detect_interval)
                pending_futures.append(future)
            except Exception as e:
                print(f"Submit error: {e}")
                continue

            # 2. Fetch a result if the buffer is full (or just to keep the flow going)
            # Waits for the oldest frame to be ready.
            if len(pending_futures) >= buffer_size - 1:
                try:
                    # Pop the oldest future and wait for it
                    output = pending_futures.popleft().result()
                except Exception as e:
                    print(f"Processing error: {e}")
                    output = frame # Fallback on error
            else:
                # Buffer filling up
                # Wait a bit to fill the pipeline, or show the raw frame to avoid an initial freeze
                # Keep filling the buffer for the first frames
                continue
        else:
            # When disabled, clear the buffer so old frames are not shown when re-enabled
            if pending_futures:
                pending_futures.clear()
            output = frame

        # Send to the virtual camera
        if vcam:
            try:
                # Make sure the frame size matches what the virtual camera expects
                if output.shape[1] != vcam.width or output.shape[0] != vcam.height:
                    output_vcam = cv2.resize(output, (vcam.width, vcam.height))
                    vcam.send(output_vcam)
                else:
                    vcam.send(output)

                vcam.sleep_until_next_frame()
            except Exception as e:
                print(f"Virtual camera error: {e}")

        # FPS calculation
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

            # Enhancer status
            enh_enabled = getattr(swapper, 'enhancement_enabled', False)
            enh_color = (0, 255, 0) if enh_enabled else (0, 0, 255)
            enh_text = "ON" if enh_enabled else "OFF"
            cv2.putText(output, f"Enhance: {enh_text}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, enh_color, 2)

        # Recording status
        if recording:
            if show_ui:
                cv2.circle(output, (50, 150), 10, (0, 0, 255), -1)
                cv2.putText(output, "REC", (70, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if video_writer is None:
                # Start audio recording
                audio_recorder = AudioRecorder()
                audio_recorder.start()

                # Create the outputs folder if it does not exist
                if not os.path.exists("outputs"):
                    os.makedirs("outputs")

                # Use mp4v for better compatibility with aac audio
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                filename = args.out if args.out else f"output_{int(time.time())}.mp4"
                # If it is not an absolute path, save to outputs/
                if not os.path.isabs(filename) and not args.out:
                     out_path = os.path.join("outputs", filename)
                else:
                     out_path = filename

                video_writer = cv2.VideoWriter(out_path, fourcc, 30.0, (output.shape[1], output.shape[0]))
                print(f"Recording to: {out_path}")
                recording_start_time = time.time()
                recording_frame_count = 0

            video_writer.write(output)
            recording_frame_count += 1
        elif video_writer:
            video_writer.release()
            video_writer = None
            # Stop audio recording and merge
            if audio_recorder:
                print("Processing audio.")
                temp_audio = audio_recorder.stop()

                if temp_audio and os.path.exists(temp_audio):
                    print("Merging audio and video.")
                    # Build a name for the temporary video file
                    temp_video = out_path.replace(".mp4", "_temp.mp4")
                    if temp_video == out_path:
                        temp_video = out_path + "_temp.mp4"

                    try:
                        if os.path.exists(out_path):
                            # Rename the original video to temp
                            # If the temp file already exists, remove it first
                            if os.path.exists(temp_video):
                                os.remove(temp_video)

                            os.rename(out_path, temp_video)

                            # Compute the real FPS
                            elapsed_recording = time.time() - recording_start_time
                            actual_fps = recording_frame_count / elapsed_recording if elapsed_recording > 0 else 30.0
                            print(f"Actual recording FPS: {actual_fps:.2f}")

                            # Merge using ffmpeg, re-encoding to fix the FPS
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

                            print(f"Recording finished with audio: {out_path}")

                            # Clean up temporary files
                            if os.path.exists(temp_video):
                                os.remove(temp_video)
                            if os.path.exists(temp_audio):
                                os.remove(temp_audio)
                        else:
                            print("Error: Video file not found for merging.")
                    except Exception as e:
                        print(f"Error merging audio: {e}")
                        # Try to restore the original video on failure
                        if os.path.exists(temp_video) and not os.path.exists(out_path):
                            os.rename(temp_video, out_path)
                else:
                    print("Audio not recorded or error while saving.")

                audio_recorder = None

        cv2.imshow("Deepfake Real-time", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_image_index = (current_image_index + 1) % len(image_files)
            new_image = image_files[current_image_index]
            print(f"Switching to: {new_image}")
            try:
                swapper.set_source_image(new_image)
                current_image_name = os.path.basename(new_image)
                # Clear the buffer to avoid mixing in old faces
                pending_futures.clear()
            except Exception as e:
                print(f"Error switching image: {e}")
        elif key == ord('p'):
            current_image_index = (current_image_index - 1) % len(image_files)
            new_image = image_files[current_image_index]
            print(f"Switching to: {new_image}")
            try:
                swapper.set_source_image(new_image)
                current_image_name = os.path.basename(new_image)
                pending_futures.clear()
            except Exception as e:
                print(f"Error switching image: {e}")
        elif key == ord('x'):
            swap_enabled = not swap_enabled
            print(f"Face swap: {'Enabled' if swap_enabled else 'Disabled'}")
        elif key == ord('e'):
            if hasattr(swapper, 'toggle_enhancer'):
                is_enabled = swapper.toggle_enhancer()
                print(f"Enhancer: {'Enabled' if is_enabled else 'Disabled'}")
            else:
                print("Enhancer not available.")
        elif key == ord('r'):
            recording = not recording
            print(f"Recording: {'Started' if recording else 'Stopped'}")
        elif key == ord('u'):
            show_ui = not show_ui
            print(f"Interface: {'Visible' if show_ui else 'Hidden'}")

    webcam.stop()
    if vcam:
        vcam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
