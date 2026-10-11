"""
Offline processing of images, videos and GIFs.
"""
import math
import shutil
import subprocess
import tempfile
import time
from fractions import Fraction

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


def _probe_frame_rate(video_path, average_fps):
    """
    The stream's nominal frame rate from ffprobe (e.g. 30/1 or 30000/1001). OpenCV only reports the
    average rate, which is wrong for variable frame rate videos (common from phones).
    Falls back to the average when ffprobe is missing or reports something implausible.
    """
    try:
        result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                 'stream=r_frame_rate', '-of', 'csv=p=0', video_path],
                                capture_output=True, text=True, check=True)
        rate = Fraction(result.stdout.strip().splitlines()[0])
        if average_fps * 0.95 <= rate <= 120:
            return rate
    except (OSError, subprocess.CalledProcessError, ValueError, ZeroDivisionError, IndexError):
        pass
    return Fraction(average_fps).limit_denominator(1001)


def process_video(swapper, video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open the video: {video_path}")

    fps = _probe_frame_rate(video_path, cap.get(cv2.CAP_PROP_FPS) or 30.0)
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
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
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
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), float(fps), (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create the output video: {out_path}")

    # Decoded frames are placed on a constant frame rate timeline by their timestamps: each frame
    # fills the output slots until the next frame's timestamp (repeated or dropped as needed), so
    # variable frame rate videos stay in sync with their audio. Dropped frames are never processed.
    frame_interval_ms = 1000.0 / float(fps)
    written = 0

    def emit(frame, until_slot):
        nonlocal written
        if until_slot <= written:
            return
        res = swapper.process_frame(frame)
        for _ in range(until_slot - written):
            if ffmpeg:
                ffmpeg.stdin.write(res.tobytes())
            else:
                writer.write(res)
        written = until_slot

    first_ms = None
    pending = None  # last decoded frame, written once the next frame's timestamp is known
    pending_ms = previous_ms = 0.0
    frame_count = 0
    start_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            if first_ms is None:
                first_ms = max(timestamp, 0.0) if math.isfinite(timestamp) else 0.0
                timestamp = first_ms
            elif not math.isfinite(timestamp) or timestamp - first_ms <= pending_ms:
                timestamp = first_ms + pending_ms + frame_interval_ms
            timestamp -= first_ms

            if pending is not None:
                # Half a slot of tolerance absorbs rounded/jittery timestamps
                emit(pending, math.ceil(timestamp / frame_interval_ms - 0.5))
            pending, previous_ms, pending_ms = frame, pending_ms, timestamp

            if frame_count % 10 == 0:
                _print_progress(frame_count, total_frames, start_time)

        if pending is not None:
            # The last frame lasts as long as the one before it
            last_duration = pending_ms - previous_ms if frame_count > 1 else frame_interval_ms
            emit(pending, max(written + 1, math.ceil((pending_ms + last_duration) / frame_interval_ms - 0.5)))
    except BrokenPipeError:
        pass  # ffmpeg exited early, its error is reported below
    finally:
        print()  # New line
        cap.release()
        if writer:
            writer.release()
        error = None
        if ffmpeg:
            try:
                ffmpeg.stdin.close()
            except BrokenPipeError:
                pass
            ffmpeg.wait()
            if ffmpeg.returncode != 0:
                ffmpeg_log.seek(0)
                error = ffmpeg_log.read().decode(errors='replace').strip()
        ffmpeg_log.close()
    if error is not None:
        raise RuntimeError(f"ffmpeg failed: {error}")

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
