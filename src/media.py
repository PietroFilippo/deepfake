"""
Offline processing of images, videos and GIFs.
"""
import json
import math
import shutil
import subprocess
import tempfile
import time
from fractions import Fraction

import cv2
import numpy as np
from PIL import Image, ImageSequence

from .tracking import FaceTracker
from .utils import read_image, write_image


def _print_progress(frame_count, total_frames, start_time, progress=None):
    if progress:
        progress(frame_count, total_frames)
    elapsed = time.time() - start_time
    fps_proc = frame_count / elapsed if elapsed > 0 else 0
    if total_frames > 0:
        progress = min(frame_count / total_frames, 1.0) * 100
        print(f"\rProcessing: {progress:.1f}% | FPS: {fps_proc:.2f} | Frame: {frame_count}/{total_frames}", end="")
    else:
        print(f"\rProcessing: FPS: {fps_proc:.2f} | Frame: {frame_count}", end="")


def process_image(swapper, image_path, out_path, show=True, pixel_boost=None):
    target_img = read_image(image_path)
    if target_img is None:
        raise ValueError(f"Could not read the target image: {image_path}")

    res = swapper.process_frame(target_img, pixel_boost=pixel_boost)
    write_image(out_path, res)
    print(f"Saved to: {out_path}")

    if show:
        cv2.namedWindow("Deepfake Image", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Deepfake Image", res)
        print("Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _probe_video_stream(video_path, average_fps):
    """
    Returns (frame_rate, duration_ms) of the first video stream using ffprobe.

    frame_rate is the stream's nominal rate (e.g. 30/1 or 30000/1001). OpenCV only reports the
    average rate, which is wrong for variable frame rate videos (common from phones); the average
    is used when ffprobe is missing or reports something implausible.
    duration_ms is the stream's length (None if unknown).
    """
    rate = Fraction(average_fps).limit_denominator(1001)
    duration_ms = None
    try:
        result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                 'stream=r_frame_rate,duration', '-of', 'json', video_path],
                                capture_output=True, text=True, check=True)
        stream = json.loads(result.stdout)['streams'][0]
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError, IndexError):
        return rate, duration_ms
    try:
        probed = Fraction(stream['r_frame_rate'])
        if average_fps * 0.95 <= probed <= 120:
            rate = probed
    except (KeyError, ValueError, ZeroDivisionError):
        pass
    try:
        duration_ms = float(stream['duration']) * 1000.0
    except (KeyError, ValueError):
        pass
    return rate, duration_ms


def process_video(swapper, video_path, out_path, pixel_boost=None, progress=None, cancel=None):
    """
    Swaps the faces in a video. `progress(done, total)` is called as frames are processed and
    `cancel()` is polled to stop early (both optional, used by the GUI).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open the video: {video_path}")

    fps, duration_ms = _probe_video_stream(video_path, cap.get(cv2.CAP_PROP_FPS) or 30.0)
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

    # Landmark smoothing across frames (timestamps in seconds)
    tracker = FaceTracker() if swapper.smoothing else None

    def emit(frame, frame_ms, until_slot):
        nonlocal written
        if until_slot <= written:
            return
        res = swapper.process_frame(frame, tracker=tracker, timestamp=frame_ms / 1000.0, pixel_boost=pixel_boost)
        for _ in range(until_slot - written):
            if ffmpeg:
                ffmpeg.stdin.write(res.tobytes())
            else:
                writer.write(res)
        written = until_slot

    first_ms = None
    pending = None  # last decoded frame, written once the next frame's timestamp is known
    pending_ms = 0.0
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
                emit(pending, pending_ms, math.ceil(timestamp / frame_interval_ms - 0.5))
            pending, pending_ms = frame, timestamp

            if frame_count % 10 == 0:
                _print_progress(frame_count, total_frames, start_time, progress)
            if cancel and cancel():
                pending = None
                break

        if pending is not None:
            # The last frame lasts until the end of the stream (one frame if the length is unknown)
            end_ms = (duration_ms - first_ms) if duration_ms else pending_ms + frame_interval_ms
            emit(pending, pending_ms, max(written + 1, math.ceil(end_ms / frame_interval_ms - 0.5)))
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


def process_gif(swapper, gif_path, out_path, pixel_boost=None, progress=None, cancel=None):
    frames = []
    durations = []
    tracker = FaceTracker() if swapper.smoothing else None
    with Image.open(gif_path) as gif:
        total_frames = getattr(gif, 'n_frames', 0)
        start_time = time.time()
        for frame in ImageSequence.Iterator(gif):
            if cancel and cancel():
                break
            # Keep each frame's own delay (GIFs often mix frame durations)
            timestamp = sum(durations) / 1000.0
            durations.append(frame.info.get('duration', 100))
            frame_bgr = cv2.cvtColor(np.array(frame.convert('RGB')), cv2.COLOR_RGB2BGR)

            res_bgr = swapper.process_frame(frame_bgr, tracker=tracker, timestamp=timestamp, pixel_boost=pixel_boost)
            frames.append(Image.fromarray(cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)))
            _print_progress(len(frames), total_frames, start_time, progress)

    if not frames:
        raise ValueError(f"No frames found in {gif_path}")

    print("\nSaving GIF.")
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f"Saved to: {out_path}")
