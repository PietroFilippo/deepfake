"""
Builds dist\\Deepfake\\Deepfake.exe with PyInstaller (run through scripts\\build_exe.bat).
"""
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(REPO, "dist", "Deepfake")


def main():
    # Without the CUDA/TensorRT folders on PATH, PyInstaller can't pick up mismatched copies of the
    # GPU DLLs from a CUDA toolkit; the spec adds the right ones explicitly.
    env = dict(os.environ)
    gpu_dirs = re.compile(r"cuda|tensorrt|cudnn|nsight|nvidia gpu computing", re.IGNORECASE)
    env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep) if not gpu_dirs.search(p))
    for name in [k for k in env if k.upper().startswith("CUDA_PATH")]:
        del env[name]

    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
           "--distpath", os.path.join(REPO, "dist"), "--workpath", os.path.join(REPO, "build"),
           os.path.join(REPO, "packaging", "deepfake.spec")]
    result = subprocess.run(cmd, env=env, cwd=REPO)
    if result.returncode != 0:
        return result.returncode

    # Folders the app uses next to the executable
    for folder in ("models", "images", "outputs"):
        os.makedirs(os.path.join(DIST, folder), exist_ok=True)
    with open(os.path.join(DIST, "models", "PUT_YOUR_MODELS_HERE.txt"), "w", encoding="utf-8") as f:
        f.write("Put your .onnx models here (or choose another folder when the app starts):\n"
                "inswapper_128_fp16.onnx / hyperswap_1a_256.onnx, GFPGANv1.4.onnx, GPEN-BFR-512.onnx, "
                "face_parser.onnx\n")

    size = sum(os.path.getsize(os.path.join(root, name))
               for root, _, names in os.walk(DIST) for name in names)
    print(f"\nBuilt {os.path.join(DIST, 'Deepfake.exe')} ({size / 1024 ** 3:.1f} GB folder).")
    print("Check it with: Deepfake.exe --self-test  (result in deepfake.log next to the exe)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
