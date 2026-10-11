import cv2
import argparse
import time
import os
import sys
from collections import deque
from src.camera import WebcamStream
from src.swapper import FaceSwapper, PIXEL_BOOST_SIZES, FACE_MASKS
from src.recorder import Recorder
from src.media import process_image, process_video, process_gif
from src.utils import OUTPUTS_DIR, model_path

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Deepfake")
    parser.add_argument("--source", help="Path to the initial source image", required=True)
    parser.add_argument("--model", default=model_path("inswapper_128_fp16.onnx"),
                        help="Path to the swap model: inswapper_128(_fp16).onnx or hyperswap_1a/1b/1c_256.onnx")

    target = parser.add_mutually_exclusive_group()
    target.add_argument("--video", help="Path to the target video file")
    target.add_argument("--image", help="Path to the target image")
    target.add_argument("--gif", help="Path to the target GIF file")
    parser.add_argument("--out", help="Path to save the recorded/processed video")

    quality = parser.add_argument_group("quality")
    quality.add_argument("--pixel-boost", type=int, choices=PIXEL_BOOST_SIZES, default=None,
                         help="Resolution the face is swapped at. Values above the model's own resolution run "
                              "the model on several interleaved tiles: sharper, but (size/model size)^2 times "
                              "slower. Default: the model's resolution (128 for inswapper, 256 for HyperSwap).")
    quality.add_argument("--face-mask", choices=FACE_MASKS, default="box",
                         help="'parser' keeps hair, hands and glasses of the target in front of the face "
                              "(slower). Default: box.")
    quality.add_argument("--enhance", action="store_true", help="Enable face enhancement by default")
    quality.add_argument("--enhancer-model", default=model_path("GFPGANv1.4.onnx"),
                         help="Face enhancer ONNX model (e.g. models/GFPGANv1.4.onnx or models/GPEN-BFR-512.onnx)")
    quality.add_argument("--enhance-blend", type=float, default=0.8,
                         help="How much of the enhanced face is blended in (0-1). Lower = less 'plastic'. Default: 0.8.")
    quality.add_argument("--det-size", type=int, choices=(320, 480, 640), default=640,
                         help="Face detector input size. 320 is ~2.5 ms faster per frame, 640 gives more "
                              "precise landmarks. Default: 640.")
    quality.add_argument("--no-smoothing", action="store_true",
                         help="Disable landmark smoothing between frames (webcam and video).")

    performance = parser.add_argument_group("webcam / performance")
    performance.add_argument("--max-workers", type=int, default=None, help="Maximum number of threads (workers). Fewer = less latency, more = more FPS.")
    performance.add_argument("--detect-interval", type=int, default=1, help="Detect faces every N frames. 1 = tightest tracking, higher = a few ms less per frame. Default: 1.")
    performance.add_argument("--camera", type=int, default=0, help="Webcam index. Default: 0.")
    performance.add_argument("--width", type=int, default=1920, help="Requested webcam width. Default: 1920.")
    performance.add_argument("--height", type=int, default=1080, help="Requested webcam height. Default: 1080.")
    performance.add_argument("--camera-fps", type=int, default=30, help="Desired webcam FPS.")
    performance.add_argument("--virtual-cam", action="store_true", help="Enable output to a virtual camera (OBS Virtual Camera).")

    args = parser.parse_args()
    if args.detect_interval < 1:
        parser.error("--detect-interval must be at least 1")
    if not 0.0 <= args.enhance_blend <= 1.0:
        parser.error("--enhance-blend must be between 0 and 1")
    return args


def find_source_images(source):
    """Lists the images in the source image's folder (for 'n'/'p' switching) and returns the source's index."""
    source = os.path.abspath(source)
    folder = os.path.dirname(source)
    image_files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                         if f.lower().endswith(IMAGE_EXTENSIONS))
    keys = [os.path.normcase(f) for f in image_files]
    if os.path.normcase(source) in keys:
        return image_files, keys.index(os.path.normcase(source))
    # Source with an unlisted extension: keep it in the list anyway
    image_files.append(source)
    return image_files, len(image_files) - 1


def build_output_path(custom_path, target_path, source_path, ext):
    """Uses --out if given, otherwise outputs/processed_<target>_with_<source>_<timestamp><ext>."""
    if custom_path:
        out_path = custom_path
    else:
        target_name = os.path.splitext(os.path.basename(target_path))[0]
        source_name = os.path.splitext(os.path.basename(source_path))[0]
        out_path = os.path.join(OUTPUTS_DIR, f"processed_{target_name}_with_{source_name}_{int(time.time())}{ext}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    return out_path


def draw_overlay(img, fps, image_name, swap_enabled, swapper, recording):
    def put(text, y, color):
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    on_off = lambda enabled: ("ON", (0, 255, 0)) if enabled else ("OFF", (0, 0, 255))
    put(f"FPS: {fps:.2f}", 30, (0, 255, 0))
    put(f"Img: {image_name}", 60, (255, 255, 0))
    text, color = on_off(swap_enabled)
    put(f"Status: {text}", 90, color)
    text, color = on_off(swapper.enhancement_enabled)
    put(f"Enhance: {text}", 120, color)
    put(f"Boost: {swapper.pixel_boost} | Mask: {swapper.face_mask}", 150, (255, 255, 0))

    # Recording status
    if recording:
        cv2.circle(img, (50, 180), 10, (0, 0, 255), -1)
        cv2.putText(img, "REC", (70, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def run_webcam(args, swapper, image_files, current_image_index):
    print(f"Starting webcam with {args.camera_fps} FPS requested.")
    try:
        webcam = WebcamStream(src=args.camera, width=args.width, height=args.height, fps=args.camera_fps).start()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    frame_w, frame_h = webcam.size
    print(f"[Webcam] Resolution: {frame_w}x{frame_h}")

    # Initialize the virtual camera if requested
    vcam = None
    if args.virtual_cam:
        try:
            import pyvirtualcam
            # Use the resolution the webcam actually delivers
            vcam = pyvirtualcam.Camera(width=frame_w, height=frame_h, fps=args.camera_fps, fmt=pyvirtualcam.PixelFormat.BGR)
            print(f"[VirtualCam] Virtual camera started: {vcam.device}")
        except ImportError:
            print("Error: 'pyvirtualcam' is not installed. Run: pip install pyvirtualcam")
            webcam.stop()
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
    print("  'e': Toggle face enhancement on/off")
    print(f"  'b': Cycle pixel boost ({'/'.join(map(str, swapper.boost_sizes))})")
    print("  'm': Toggle face mask (box/parser)")
    print("  'r': Start/Stop recording")
    print("  'u': Show/Hide interface (UI)")

    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    swap_enabled = True
    show_ui = True

    # Buffer to hold pending futures.
    # Size = workers + 1 minimizes latency while keeping the workers busy.
    buffer_size = swapper.max_workers + 1
    pending_futures = deque()
    print(f"[Main] Frame buffer size: {buffer_size}")

    current_image_name = os.path.basename(image_files[current_image_index])
    recorder = None
    last_frame_id = 0

    try:
        while True:
            # Only process new camera frames (the same frame used to be swapped and counted repeatedly)
            frame_id, frame = webcam.read(last_frame_id)
            if frame_id == last_frame_id:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            last_frame_id = frame_id

            if swap_enabled:
                # 1. Submit the frame to the worker pool
                try:
                    pending_futures.append(swapper.process_frame_async(frame, detect_interval=args.detect_interval))
                except Exception as e:
                    print(f"Submit error: {e}")
                    continue

                # 2. Keep filling the pipeline for the first frames, then wait for the oldest frame
                if len(pending_futures) < buffer_size - 1:
                    continue
                try:
                    output = pending_futures.popleft().result()
                except Exception as e:
                    print(f"Processing error: {e}")
                    output = frame # Fallback on error
            else:
                # When disabled, clear the buffer so old frames are not shown when re-enabled
                pending_futures.clear()
                output = frame

            # Send to the virtual camera
            if vcam:
                try:
                    if output.shape[1] != vcam.width or output.shape[0] != vcam.height:
                        vcam.send(cv2.resize(output, (vcam.width, vcam.height)))
                    else:
                        vcam.send(output)
                except Exception as e:
                    print(f"Virtual camera error: {e}")

            # Recordings never include the UI overlay
            if recorder:
                recorder.write(output)

            # FPS calculation
            fps_frame_count += 1
            if time.time() - fps_start_time > 1.0:
                fps = fps_frame_count / (time.time() - fps_start_time)
                fps_frame_count = 0
                fps_start_time = time.time()

            # UI Overlay (drawn on a copy so the camera frame itself is never modified)
            display = output
            if show_ui:
                display = output.copy()
                draw_overlay(display, fps, current_image_name, swap_enabled, swapper, recorder is not None)

            cv2.imshow("Deepfake Real-time", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in (ord('n'), ord('p')):
                step = 1 if key == ord('n') else -1
                current_image_index = (current_image_index + step) % len(image_files)
                new_image = image_files[current_image_index]
                print(f"Switching to: {new_image}")
                try:
                    swapper.set_source_image(new_image)
                    current_image_name = os.path.basename(new_image)
                    # Clear the buffer to avoid mixing in old faces
                    pending_futures.clear()
                except Exception as e:
                    print(f"Error switching image: {e}")
            elif key == ord('x'):
                swap_enabled = not swap_enabled
                print(f"Face swap: {'Enabled' if swap_enabled else 'Disabled'}")
            elif key == ord('e'):
                if swapper.enhancer:
                    is_enabled = swapper.toggle_enhancer()
                    print(f"Enhancer: {'Enabled' if is_enabled else 'Disabled'}")
                else:
                    print("Enhancer not available.")
            elif key == ord('b'):
                sizes = swapper.boost_sizes
                swapper.pixel_boost = sizes[(sizes.index(swapper.pixel_boost) + 1) % len(sizes)]
                print(f"Pixel boost: {swapper.pixel_boost}")
            elif key == ord('m'):
                if swapper.parser is None:
                    print("Face parser not available.")
                else:
                    swapper.face_mask = 'parser' if swapper.face_mask == 'box' else 'box'
                    print(f"Face mask: {swapper.face_mask}")
            elif key == ord('r'):
                if recorder:
                    print("Recording: Stopped")
                    # Detached first, so a failing stop() is not retried on exit
                    stopping, recorder = recorder, None
                    try:
                        stopping.stop()
                    except Exception as e:
                        print(f"Error finishing recording: {e}")
                else:
                    print("Recording: Started")
                    out_path = args.out or os.path.join(OUTPUTS_DIR, f"output_{int(time.time())}.mp4")
                    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
                    try:
                        recorder = Recorder(out_path, (output.shape[1], output.shape[0]))
                    except Exception as e:
                        print(f"Error starting recording: {e}")
            elif key == ord('u'):
                show_ui = not show_ui
                print(f"Interface: {'Visible' if show_ui else 'Hidden'}")
    finally:
        # Finish an ongoing recording instead of losing it (and hanging on its audio thread).
        # Each step runs even if an earlier one fails.
        try:
            if recorder:
                recorder.stop()
        finally:
            try:
                webcam.stop()
                if vcam:
                    vcam.close()
            finally:
                cv2.destroyAllWindows()


def main():
    args = parse_args()

    if not os.path.isfile(args.source):
        print(f"Error: Source image not found at {args.source}")
        sys.exit(1)
    image_files, current_image_index = find_source_images(args.source)
    source_path = image_files[current_image_index]

    offline = bool(args.image or args.video or args.gif)
    det_size = args.det_size

    print(f"Available images: {len(image_files)}")
    print("Initializing.")
    try:
        swapper = FaceSwapper(args.model, det_size=(det_size, det_size), max_workers=args.max_workers,
                              pixel_boost=args.pixel_boost, face_mask=args.face_mask,
                              enhancer_model=args.enhancer_model, enhance_blend=args.enhance_blend,
                              smoothing=not args.no_smoothing)
        swapper.set_source_image(source_path)
        if args.enhance:
            if swapper.enhancer:
                swapper.enhancement_enabled = True
                print("Enhancer enabled by default.")
            else:
                print("Warning: --enhance ignored, the enhancer could not be loaded.")
    except Exception as e:
        print(f"Error initializing swapper: {e}")
        sys.exit(1)

    if not offline:
        run_webcam(args, swapper, image_files, current_image_index)
        return

    target_path = args.image or args.video or args.gif
    if not os.path.isfile(target_path):
        print(f"Error: Target file not found: {target_path}")
        sys.exit(1)
    if args.out and os.path.normcase(os.path.abspath(args.out)) == os.path.normcase(os.path.abspath(target_path)):
        print("Error: --out must be different from the input file.")
        sys.exit(1)

    try:
        if args.image:
            # Static Image Mode
            print(f"Processing single image: {args.image}")
            ext = os.path.splitext(args.image)[1].lower()
            out_path = build_output_path(args.out, args.image, source_path, ext if ext in IMAGE_EXTENSIONS else ".jpg")
            process_image(swapper, args.image, out_path)
        elif args.video:
            # Offline Video Mode
            print(f"Processing video: {args.video}")
            process_video(swapper, args.video, build_output_path(args.out, args.video, source_path, ".mp4"))
        else:
            # GIF Mode
            print(f"Processing GIF: {args.gif}")
            process_gif(swapper, args.gif, build_output_path(args.out, args.gif, source_path, ".gif"))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
