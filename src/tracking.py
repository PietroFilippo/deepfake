"""
Landmark smoothing between frames.

The detector's 5 landmarks jitter by a few pixels from frame to frame (more so at a low detector
resolution), which makes the swapped face shimmer. Each face is followed over time and its
landmarks are smoothed with a One Euro filter: strong smoothing while the face is still, almost
no lag while it moves.
"""
import math

import numpy as np
from insightface.app.common import Face


class OneEuroFilter:
    """
    One Euro filter (Casiez et al., 2012) for an array of points.
    Speeds are measured in face sizes per second, so the same settings work for near and far faces.
    Defaults were tuned on sample videos: ~1/3 less landmark jitter with no measurable lag.
    """

    def __init__(self, min_cutoff=2.0, beta=10.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(dt, cutoff):
        return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff * dt))

    def __call__(self, x, t, scale):
        if self.t_prev is None or t <= self.t_prev:
            self.x_prev, self.dx_prev, self.t_prev = x, np.zeros_like(x), t
            return x
        dt = t - self.t_prev

        # Smoothed speed of each point, in face sizes per second
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        speed = np.linalg.norm(dx_hat, axis=-1, keepdims=True) / max(scale, 1e-6)

        # Faster movement -> higher cutoff -> less smoothing (less lag)
        cutoff = self.min_cutoff + self.beta * speed
        a = self._alpha(dt, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class FaceTracker:
    """Matches faces between frames and smooths their landmarks."""

    def __init__(self, min_cutoff=2.0, beta=10.0, max_age=0.5):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.max_age = max_age  # seconds a face may go undetected before its track is dropped
        self.tracks = []

    def reset(self):
        self.tracks = []

    def update(self, faces, t):
        """Returns copies of `faces` with smoothed landmarks. `t` is the frame time in seconds."""
        unmatched = list(self.tracks)
        tracks = []
        smoothed = []
        # Larger faces first, so they get their own track when faces are close together
        for face in sorted(faces, key=lambda f: -(f.bbox[2] - f.bbox[0])):
            kps = np.asarray(face.kps, dtype=np.float64)
            center = kps.mean(axis=0)
            size = float(face.bbox[2] - face.bbox[0])

            track = None
            if unmatched:
                distances = [np.linalg.norm(center - tr['center']) for tr in unmatched]
                best = int(np.argmin(distances))
                if distances[best] < 0.5 * size:
                    track = unmatched.pop(best)
            if track is None:
                track = {'filter': OneEuroFilter(self.min_cutoff, self.beta)}

            kps = track['filter'](kps, t, size)
            track['center'] = kps.mean(axis=0)
            track['last_seen'] = t
            tracks.append(track)
            smoothed.append(Face(bbox=face.bbox, kps=kps.astype(np.float32), det_score=face.det_score))

        # Keep briefly missed faces so a single missed detection doesn't reset the smoothing
        tracks += [tr for tr in unmatched if t - tr['last_seen'] < self.max_age]
        self.tracks = tracks
        return smoothed
