import cv2
import threading
import time


class WebcamStream:
    def __init__(self, src=0, width=1920, height=1080, fps=30):
        # Use DirectShow (CAP_DSHOW) on Windows to avoid MSMF errors and speed up initialization
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.stream.isOpened():
            raise RuntimeError(f"Could not open webcam {src}")

        # Request MJPG before the resolution: most USB webcams only reach 1080p at full
        # frame rate with MJPG (raw YUY2 is often limited to ~5 FPS). Ignored if unsupported.
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, fps)

        # Read the first frame after configuring, so it already has the final resolution
        grabbed, frame = self.stream.read()
        if not grabbed:
            self.stream.release()
            raise RuntimeError(f"Could not read from webcam {src}")

        self.frame = frame
        self.frame_id = 1
        self.stopped = False
        self.condition = threading.Condition()
        self.thread = threading.Thread(target=self.update, daemon=True)

    @property
    def size(self):
        """(width, height) of the frames actually delivered by the camera."""
        h, w = self.frame.shape[:2]
        return w, h

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                time.sleep(0.01)
                continue
            with self.condition:
                self.frame = frame
                self.frame_id += 1
                self.condition.notify_all()

    def read(self, last_id=0, timeout=1.0):
        """
        Returns (frame_id, frame), waiting up to `timeout` seconds for a frame newer than `last_id`.
        On timeout the latest (already seen) frame is returned with the same id.
        """
        with self.condition:
            self.condition.wait_for(lambda: self.frame_id != last_id or self.stopped, timeout)
            return self.frame_id, self.frame

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.stream.release()
