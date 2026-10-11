"""
Builds dist\\Deepfake\\Deepfake.exe with PyInstaller (run through scripts\\build_exe.bat).
"""
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(REPO, "dist", "Deepfake")
# The app keeps these next to the executable; PyInstaller replaces the whole folder, so they are
# set aside during the build and put back afterwards (recordings, settings, engine cache, ...)
USER_FILES = ("settings.json", "deepfake.log", "trt_cache", "models", "images", "outputs")


def run_pyinstaller():
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
    return subprocess.run(cmd, env=env, cwd=REPO).returncode


def main():
    kept_dir = os.path.join(REPO, "dist", "_user_files_during_build")
    kept = []
    if os.path.isdir(DIST):
        os.makedirs(kept_dir, exist_ok=True)
        for name in USER_FILES:
            if os.path.exists(os.path.join(DIST, name)):
                shutil.move(os.path.join(DIST, name), os.path.join(kept_dir, name))
                kept.append(name)
    try:
        returncode = run_pyinstaller()
    finally:
        os.makedirs(DIST, exist_ok=True)
        for name in kept:
            shutil.move(os.path.join(kept_dir, name), os.path.join(DIST, name))
        if os.path.isdir(kept_dir) and not os.listdir(kept_dir):
            os.rmdir(kept_dir)
    if returncode != 0:
        return returncode

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
    if kept:
        print(f"Kept from the previous build: {', '.join(kept)}")
    print("Check it with: Deepfake.exe --self-test  (result in deepfake.log next to the exe)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
