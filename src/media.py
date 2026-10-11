"""
Offline processing of images, videos and GIFs.
"""
import shutil
import subprocess
import tempfile
import time

import cv2
import numpy as np
from PIL import Image, ImageSequence

from .utils import read_image, write_image


def _print_progress(frame_count, total_frames, start_time):
    elapsed = time.time() - start_time
    fps_proc = frame_count / elapsed if elapsed > 0 else 0
    if total_frames > 0:
        progress = min(frame_count / total_frames, 1.0) * 100
        print(f"\rProcessing: {progress:.1f}% | FPS: {fps_proc:.2f} | Frame: {frame_count}/{total_frames}", end="")
    else:
        print(f"\rProcessing: FPS: {fps_proc:.2f} | Frame: {frame_count}", end="")


def process_image(swapper, image_path, out_path, show=True):
    target_img = read_image(image_path)
    if target_img is None:
        raise ValueError(f"Could not read the target image: {image_path}")

    res = swapper.process_frame(target_img)
    write_image(out_path, res)
    print(f"Saved to: {out_path}")

    if show:
        cv2.namedWindow("Deepfake Image", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Deepfake Image", res)
        print("Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(swapper, video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open the video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Encode with ffmpeg through a pipe: H.264 at high quality, with the source audio (if any)
    # copied over. Falls back to OpenCV's mp4v writer (no audio) when ffmpeg is not installed.
    ffmpeg = None
    writer = None
    ffmpeg_log = tempfile.TemporaryFile()
    if shutil.which('ffmpeg'):
        cmd = ['ffmpeg', '-y', '-loglevel', 'error',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', f'{fps:.6f}', '-i', '-',
               '-i', video_path,
               '-map', '0:v:0', '-map', '1:a:0?',
               # yuv420p (needed for compatibility) requires even dimensions
               '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
               '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
               '-c:a', 'aac', '-b:a', '192k',
               out_path]
        ffmpeg = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=ffmpeg_log)
        print(f"Processing {total_frames} frames.")
    else:
        print("Warning: ffmpeg not found. Saving without audio.")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create the output video: {out_path}")

    frame_count = 0
    start_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            res = swapper.process_frame(frame)
            if ffmpeg:
                ffmpeg.stdin.write(res.tobytes())
            else:
                writer.write(res)

            frame_count += 1
            if frame_count % 10 == 0:
                _print_progress(frame_count, total_frames, start_time)
    except BrokenPipeError:
        pass  # ffmpeg exited early, its error is reported below
    finally:
        print()  # New line
        cap.release()
        if writer:
            writer.release()
        if ffmpeg:
            try:
                ffmpeg.stdin.close()
            except BrokenPipeError:
                pass
            ffmpeg.wait()
            ffmpeg_log.seek(0)
            error = ffmpeg_log.read().decode(errors='replace').strip()
            if ffmpeg.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {error}")
        ffmpeg_log.close()

    print(f"Done. Saved to: {out_path}")


def process_gif(swapper, gif_path, out_path):
    frames = []
    durations = []
    with Image.open(gif_path) as gif:
        total_frames = getattr(gif, 'n_frames', 0)
        start_time = time.time()
        for frame in ImageSequence.Iterator(gif):
            # Keep each frame's own delay (GIFs often mix frame durations)
            durations.append(frame.info.get('duration', 100))
            frame_bgr = cv2.cvtColor(np.array(frame.convert('RGB')), cv2.COLOR_RGB2BGR)

            res_bgr = swapper.process_frame(frame_bgr)
            frames.append(Image.fromarray(cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)))
            _print_progress(len(frames), total_frames, start_time)

    if not frames:
        raise ValueError(f"No frames found in {gif_path}")

    print("\nSaving GIF.")
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f"Saved to: {out_path}")
