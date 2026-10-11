"""FaceParser smoke test. Uses the first image in images/."""
import sys
import os
import cv2
import numpy as np

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.face_parser import FaceParser

def main():
    img_dir = os.path.join(ROOT, "images")
    candidates = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    if not candidates:
        print("No images found in images/")
        sys.exit(1)
    img_path = os.path.join(img_dir, candidates[0])
    print(f"Testing with: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        print("Failed to read image.")
        sys.exit(1)

    parser = FaceParser(model_path=os.path.join(ROOT, "models", "face_parser.onnx"))
    mask = parser.parse(img)
    print(f"Mask shape: {mask.shape} dtype: {mask.dtype} min: {mask.min():.3f} max: {mask.max():.3f} mean: {mask.mean():.3f}")
    assert mask.shape == img.shape[:2], "Mask shape mismatch"
    assert mask.dtype == np.float32
    assert 0.0 <= mask.min() and mask.max() <= 1.0
    assert mask.mean() > 0.0, "Empty mask (no face detected?)"

    # Save visualization
    out_dir = os.path.join(ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    vis_mask = (mask * 255).astype(np.uint8)
    vis_mask_3ch = cv2.cvtColor(vis_mask, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(img, 0.6, vis_mask_3ch, 0.4, 0)
    out_path = os.path.join(out_dir, "parser_smoke.jpg")
    cv2.imwrite(out_path, np.hstack([img, vis_mask_3ch, overlay]))
    print(f"Visualization saved to: {out_path}")
    print("OK")

if __name__ == "__main__":
    main()
