"""
Live webcam pipeline for the GUI: capture -> swap (worker pool) -> virtual camera / recording.
Runs on its own thread; the window only reads the latest processed frame.
"""
import queue
import threading
import time
from collections import deque

import cv2

from ..camera import WebcamStream
from ..recorder import Recorder


class LivePipeline:
    def __init__(self, swapper, on_status=print):
        self.swapper = swapper
        self.on_status = on_status
        self.detect_interval = 1
        self.swap_enabled = True

        self._commands = queue.Queue()
        self._lock = threading.Lock()
        self._latest = (0, None)
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="LivePipeline")

        self.webcam = None
        self.vcam = None
        self.vcam_wanted = False
        self.vcam_fps = 30
        self.recorder = None
        self._recording_path = None
        self.fps = 0.0

    # ------------------------------------------------------------------ called from the GUI thread

    def start(self):
        self._thread.start()

    def stop(self):
        """Stops the loop and releases the camera, virtual camera and an ongoing recording."""
        self._stopped = True
        # No timeout: an ongoing recording is encoded before the thread ends
        if self._thread.is_alive():
            self._thread.join()

    def latest(self):
        """(frame_number, frame) of the most recent output frame."""
        with self._lock:
            return self._latest

    def open_camera(self, index, width, height, fps):
        self._commands.put(lambda: self._open_camera(index, width, height, fps))

    def set_virtual_cam(self, enabled, fps=30):
        def apply():
            self.vcam_wanted, self.vcam_fps = enabled, fps
            if not enabled:
                self._close_vcam()
        self._commands.put(apply)

    def start_recording(self, out_path):
        def apply():
            self._recording_path = out_path  # the recorder is created with the next frame's size
        self._commands.put(apply)

    def stop_recording(self):
        self._commands.put(self._stop_recording)

    @property
    def recording(self):
        return self.recorder is not None or self._recording_path is not None

    @property
    def camera_open(self):
        return self.webcam is not None

    # ------------------------------------------------------------------ pipeline thread

    def _open_camera(self, index, width, height, fps):
        if self.webcam:
            self.webcam.stop()
            self.webcam = None
        self.on_status(f"Opening camera {index} ({width}x{height} @ {fps} FPS)...")
        try:
            self.webcam = WebcamStream(src=index, width=width, height=height, fps=fps).start()
        except RuntimeError as e:
            self.on_status(f"Error: {e}. Choose another camera in the settings (Tab).")
            return
        w, h = self.webcam.size
        self.on_status(f"Camera {index}: {w}x{h}")
        self._close_vcam()  # recreated with the new size

    def _close_vcam(self):
        if self.vcam:
            self.vcam.close()
            self.vcam = None

    def _send_vcam(self, frame):
        if not self.vcam_wanted:
            return
        h, w = frame.shape[:2]
        if self.vcam is None or (self.vcam.width, self.vcam.height) != (w, h):
            self._close_vcam()
            try:
                import pyvirtualcam
                self.vcam = pyvirtualcam.Camera(width=w, height=h, fps=self.vcam_fps, fmt=pyvirtualcam.PixelFormat.BGR)
                self.on_status(f"Virtual camera started: {self.vcam.device}")
            except Exception as e:
                self.vcam_wanted = False
                self.on_status(f"Error starting virtual camera: {e} (is OBS Studio installed?)")
                return
        try:
            self.vcam.send(frame)
        except Exception as e:
            self.on_status(f"Virtual camera error: {e}")

    def _stop_recording(self):
        self._recording_path = None
        if self.recorder:
            recorder, self.recorder = self.recorder, None
            self.on_status("Saving recording...")

            def finish():
                try:
                    recorder.stop()
                    self.on_status(f"Recording saved: {recorder.out_path}")
                except Exception as e:
                    self.on_status(f"Error finishing recording: {e}")
            # Encoding takes a few seconds; don't freeze the live view meanwhile
            threading.Thread(target=finish, daemon=False).start()

    def _record(self, frame):
        if self._recording_path:
            path, self._recording_path = self._recording_path, None
            try:
                self.recorder = Recorder(path, (frame.shape[1], frame.shape[0]))
                self.on_status(f"Recording to {path}")
            except Exception as e:
                self.on_status(f"Error starting recording: {e}")
        if self.recorder:
            self.recorder.write(frame)

    def _run(self):
        pending = deque()
        last_frame_id = 0
        frame_number = 0
        fps_count, fps_start = 0, time.time()
        try:
            while not self._stopped:
                while not self._commands.empty():
                    try:
                        self._commands.get_nowait()()
                    except Exception as e:
                        self.on_status(f"Error: {e}")

                if self.webcam is None:
                    time.sleep(0.05)
                    continue

                frame_id, frame = self.webcam.read(last_frame_id, timeout=0.5)
                if frame_id == last_frame_id:
                    continue
                last_frame_id = frame_id

                swapper = self.swapper
                if self.swap_enabled and swapper.source_latent is not None:
                    try:
                        pending.append(swapper.process_frame_async(frame, detect_interval=self.detect_interval))
                    except Exception as e:
                        self.on_status(f"Submit error: {e}")
                        continue
                    # Keep the workers busy: wait for the oldest frame once the pipeline is full
                    if len(pending) < max(1, swapper.max_workers):
                        continue
                    try:
                        output = pending.popleft().result()
                    except Exception as e:
                        self.on_status(f"Processing error: {e}")
                        output = frame
                else:
                    pending.clear()
                    output = frame

                self._send_vcam(output)
                self._record(output)

                fps_count += 1
                now = time.time()
                if now - fps_start >= 1.0:
                    self.fps = fps_count / (now - fps_start)
                    fps_count, fps_start = 0, now

                frame_number += 1
                with self._lock:
                    self._latest = (frame_number, output)
        finally:
            self._recording_path = None
            if self.recorder:
                try:
                    self.recorder.stop()
                except Exception as e:
                    print(f"Error finishing recording: {e}")
                self.recorder = None
            try:
                if self.webcam:
                    self.webcam.stop()
            finally:
                self._close_vcam()
