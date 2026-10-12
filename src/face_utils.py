"""
Face alignment, masking and paste-back helpers shared by the swapper and the enhancer.
"""
import cv2
import numpy as np

# 5-point landmark templates (left eye, right eye, nose tip, left and right mouth corners),
# normalized to [0, 1] of the crop size.

# FFHQ alignment used to train GFPGAN / GPEN / CodeFormer (facexlib's 512 face template / 512).
FFHQ_TEMPLATE = np.array([
    [0.37691676, 0.46864664],
    [0.62285697, 0.46912813],
    [0.50123859, 0.61331904],
    [0.39308822, 0.72541100],
    [0.61150205, 0.72490465],
], dtype=np.float32)

_box_mask_cache = {}


def estimate_ffhq_matrix(kps, size):
    """Affine matrix that maps the face (5 landmarks) to an FFHQ-aligned crop of `size` x `size`."""
    matrix, _ = cv2.estimateAffinePartial2D(
        np.asarray(kps, dtype=np.float32), FFHQ_TEMPLATE * size, method=cv2.LMEDS)
    return matrix


def box_mask(size, border=0.05, blur=0.025):
    """
    Feathered square mask (float32, `size` x `size`) for blending an aligned crop back.
    `border` and `blur` are fractions of the crop size. The result is cached per size.
    """
    key = (size, border, blur)
    mask = _box_mask_cache.get(key)
    if mask is None:
        mask = np.zeros((size, size), dtype=np.float32)
        b = max(1, int(size * border))
        mask[b:-b, b:-b] = 1.0
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, size * blur))
        # Make sure the mask fades to exactly 0 at the crop's edge (no visible seam)
        mask[[0, -1], :] = 0.0
        mask[:, [0, -1]] = 0.0
        mask.flags.writeable = False
        _box_mask_cache[key] = mask
    return mask


def _ramp(n, start, end):
    """0 before `start`, 1 after `end` (fractions of n), smooth in between."""
    t = np.clip((np.arange(n, dtype=np.float32) / n - start) / (end - start), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def forehead_cut_mask(size):
    """
    Swap mask that leaves out everything above the eyebrows (crop aligned with the ArcFace template,
    where the eyebrows start ~30% from the top), so the camera's own forehead and hair stay visible.
    """
    key = ('forehead_cut', size)
    mask = _box_mask_cache.get(key)
    if mask is None:
        mask = np.repeat(_ramp(size, 0.20, 0.27)[:, None], size, axis=1)
        mask.flags.writeable = False
        _box_mask_cache[key] = mask
    return mask


def fringe_zone(size):
    """Where a fringe can be: from the top of the crop down to just above the eyes, across the forehead."""
    key = ('fringe_zone', size)
    mask = _box_mask_cache.get(key)
    if mask is None:
        rows = _ramp(size, 0.0, 0.03) * (1.0 - _ramp(size, 0.33, 0.38))
        cols = _ramp(size, 0.10, 0.20) * (1.0 - _ramp(size, 0.80, 0.90))
        mask = rows[:, None] * cols[None, :]
        mask.flags.writeable = False
        _box_mask_cache[key] = mask
    return mask


def paste_back(frame, crop, crop_mask, matrix):
    """
    Warps `crop` (cut out of `frame` with the affine `matrix`) back onto `frame`, in place,
    blending with `crop_mask` (float32 in [0, 1], same size as `crop`).

    Only the face's bounding box is touched, which is much cheaper than warping
    full-size frames like insightface's built-in paste_back does.
    """
    frame_h, frame_w = frame.shape[:2]
    crop_h, crop_w = crop.shape[:2]
    inverse = cv2.invertAffineTransform(matrix)

    # Bounding box of the crop's corners in frame coordinates, plus 1px for the interpolation
    corners = np.array([[0, 0], [crop_w, 0], [0, crop_h], [crop_w, crop_h]], dtype=np.float32)
    points = corners @ inverse[:, :2].T + inverse[:, 2]
    x1, y1 = np.floor(points.min(axis=0)).astype(int) - 1
    x2, y2 = np.ceil(points.max(axis=0)).astype(int) + 1
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame_w), min(y2, frame_h)
    if x2 <= x1 or y2 <= y1:
        return frame

    # Same transform, but relative to the region of interest
    inverse[:, 2] -= (x1, y1)
    roi_size = (x2 - x1, y2 - y1)
    warped = cv2.warpAffine(crop, inverse, roi_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpAffine(crop_mask, inverse, roi_size, flags=cv2.INTER_LINEAR, borderValue=0.0)

    # cv2.blendLinear is ~20x faster than the equivalent numpy float math on large faces
    frame[y1:y2, x1:x2] = cv2.blendLinear(warped, frame[y1:y2, x1:x2], mask, 1.0 - mask)
    return frame
