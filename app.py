"""
Deepfake desktop app: live webcam face swap with a settings overlay (press Tab).
For the command line (and offline processing from scripts) use main.py.

    app.py               start the app
    app.py --self-test   load the models and check the GPU setup without a window or camera
                         (the result is printed, or written to deepfake.log in the packaged app)
"""
import os
import sys


def self_test():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    import numpy as np
    from src.gui.settings import (ENHANCER_MODEL_KEYS, PARSER_MODEL_NAME, SWAP_MODEL_KEYS, Settings,
                                  list_models)
    from src.swapper import FaceSwapper
    from src.utils import ffmpeg_executable

    settings = Settings.load()
    swap_models = list_models(settings.models_dir, SWAP_MODEL_KEYS)
    if not swap_models:
        print(f"Self-test FAILED: no swap model in {settings.models_dir}")
        return 1
    name = settings.swap_model if settings.swap_model in swap_models else swap_models[0]
    enhancers = list_models(settings.models_dir, ENHANCER_MODEL_KEYS)
    parser = settings.model_file(PARSER_MODEL_NAME)

    swapper = FaceSwapper(settings.model_file(name), det_size=(settings.det_size, settings.det_size),
                          enhancer_model=settings.model_file(enhancers[0]) if enhancers else '',
                          parser_model=parser if os.path.exists(parser) else None)
    swapper.detect_faces(np.zeros((720, 1280, 3), np.uint8))
    latent = np.random.randn(1, 512).astype(np.float32)
    swapper.model.run(np.zeros((swapper.model.size, swapper.model.size, 3), np.uint8),
                      latent / np.linalg.norm(latent))

    # Build the window once (not shown) to check that Qt and its plugins load
    from PySide6.QtWidgets import QApplication
    from src.gui.main_window import MainWindow
    app = QApplication.instance() or QApplication([])
    MainWindow(settings).deleteLater()

    provider = swapper.model.session.get_providers()[0]
    print(f"Swap model: {name} on {provider}")
    print(f"Enhancer: {swapper.enhancer is not None} | Face parser: {swapper.parser is not None}")
    print(f"ffmpeg: {ffmpeg_executable() or 'missing (videos and recordings will have no audio)'}")
    gpu = provider in ('TensorrtExecutionProvider', 'CUDAExecutionProvider')
    print("Self-test OK" if gpu else "Self-test WARNING: running on the CPU (check the NVIDIA driver)")
    return 0 if gpu else 2


def main():
    # Builder process started by src/engines.py: builds one TensorRT engine and exits
    if len(sys.argv) == 5 and sys.argv[1] == '--build-engine':
        from src.engines import build_engine_main
        return build_engine_main(sys.argv[2], sys.argv[3], sys.argv[4] == '1')

    from src.utils import PROJECT_ROOT

    # pythonw.exe and the packaged app have no console: keep a log file for troubleshooting
    if sys.stdout is None or sys.stderr is None:
        log = open(os.path.join(PROJECT_ROOT, 'deepfake.log'), 'w', encoding='utf-8', buffering=1)
        sys.stdout = sys.stderr = log

    if '--self-test' in sys.argv[1:]:
        return self_test()

    from src.gui.main_window import run
    return run()


if __name__ == "__main__":
    sys.exit(main())
