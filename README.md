# Real-Time Deepfake for Webcam, Videos, Images, and GIFs

Face swapping on a live webcam feed or offline files, with a PySide6 desktop app and a command-line interface. GPU inference uses TensorRT when possible, then CUDA, then CPU.

## Features

- **Desktop app**: live preview, source-face thumbnails, and a settings overlay toggled with Tab/F1.
- **Swap models**: inswapper_128 and HyperSwap 1a/1b/1c at 256x256, with model switching and pixel boost up to 512x512.
- **Face quality**: optional GFPGAN/GPEN enhancement, adjustable blend, box/parser masks, and fringe control.
- **Smoother tracking**: One Euro landmark smoothing between frames to reduce jitter.
- **Threaded webcam pipeline**: adjustable workers and detection interval; mirror, upside-down flip, and virtual camera output.
- **Offline processing**: images, videos with source audio, and GIFs with per-frame delays. The GUI has progress and cancellation for videos/GIFs.
- **Recording**: processed webcam output with microphone audio when available; the HUD is never recorded.
- **Cached TensorRT engines**: built in a separate process once, then loaded through small EP-context models for quick startup and switching.

## Requirements

This is a Windows project (the webcam backend uses DirectShow).

- **Python 3.11, 64-bit**, for running from source or building the executable.
- **NVIDIA GPU and driver** for practical real-time performance. CPU fallback exists, but the FPS below are GPU measurements.
- **CUDA 12.4** and **TensorRT 10.4.0** are the author's existing setup ([CUDA download](https://developer.nvidia.com/cuda-downloads), [TensorRT download](https://developer.nvidia.com/tensorrt)). TensorRT download may require an NVIDIA login. Match runtime versions to your ONNX Runtime build; see the [official compatibility table](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements).
- **PyTorch's CUDA wheel** supplies CUDA/cuDNN DLLs used by the source app and executable builder.
- **FFmpeg** is found on PATH first, otherwise through `imageio-ffmpeg`, which is in `requirements.txt`. Installing [FFmpeg](https://www.ffmpeg.org/download.html) on PATH also supplies **ffprobe** for video frame-rate/duration probing; the bundled fallback does not include ffprobe.
- A **webcam** for live mode, a **microphone** for recording audio, and a virtual camera driver (such as OBS Studio's) for virtual camera output.

The local reference environment has `onnxruntime-gpu 1.23.2` and `insightface 0.7.3`. `requirements.txt` is unpinned: it includes `PySide6-Essentials` (the Qt widgets used by the app) and `imageio-ffmpeg`, but does not install PyTorch or TensorRT. A fresh install can resolve different versions.

## Installation

Run these commands in **Windows Command Prompt**, from the repository root unless a step says otherwise.

### 1. Get the project and create a virtual environment

Clone your copy of the repository, then:

```bat
cd deepfake
py -3.11 -m venv venv
venv\Scripts\activate.bat
```

### 2. Install Python dependencies

```bat
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

Or, with the virtual environment active, run the helper (it does both steps):

```bat
scripts\install_dependencies.bat
```

### 3. Set up TensorRT

Extract TensorRT. Set `TENSORRT_DIR` to the folder that actually contains `lib\` and `python\`, not the outer extraction folder. For the author's nested extraction:

```bat
set "TENSORRT_DIR=C:\Program Files\TensorRT-10.4.0.26\TensorRT-10.4.0.26"
python -m pip install "%TENSORRT_DIR%\python\tensorrt-10.4.0-cp311-none-win_amd64.whl"
```

Use the Python 3.11 Windows wheel from your matching TensorRT distribution. The full `tensorrt` wheel supplies the import expected by `tools/check_environment.py`; installing only `tensorrt_lean` does not satisfy that check. ONNX Runtime uses the TensorRT DLLs in `lib\`.

For double-click launches, make the variable persistent and start a new terminal/app afterwards:

```bat
setx TENSORRT_DIR "C:\Program Files\TensorRT-10.4.0.26\TensorRT-10.4.0.26"
```

Without `TENSORRT_DIR`, the app searches these exact library directories:

- `C:\Program Files\TensorRT-10.4.0.26\TensorRT-10.4.0.26\lib`
- `C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\lib`
- `C:\TensorRT\lib`

Keep CUDA's runtime directories on PATH. The app also adds detected TensorRT, ONNX Runtime, and PyTorch DLL directories to its Windows DLL search path.

### 4. Add models and source faces

Models are downloaded separately. Put them in `models\`, or select another models folder in the desktop app.

| Model | Required? | Purpose |
|---|---|---|
| `inswapper_128_fp16.onnx`, `inswapper_128.onnx`, or `hyperswap_1a_256.onnx` / `1b` / `1c` | One swap model required | Applies the source identity. CLI defaults to `models/inswapper_128_fp16.onnx`; use `--model` for another file. |
| `det_10g.onnx` and `w600k_r50.onnx` from InsightFace's `buffalo_l` pack | Required, fetched by InsightFace if the pack is absent | Detection and source identity embedding. Stored in `~/.insightface/models/buffalo_l/`, outside this repo. Only these two models are loaded. |
| `GFPGANv1.4.onnx` | Optional | Default enhancer; enhancement starts off. |
| `GPEN-BFR-512.onnx` | Optional | Alternative compatible FFHQ-aligned enhancer. |
| `face_parser.onnx` | Optional | BiSeNet anatomical masks for swapping/enhancement; missing or unloadable parser falls back to box masks. |

For downloads, see [HyperSwap](#hyperswap-256x256-swap-model). If you have the original inswapper, `python tools/convert_fp16_v2.py` converts `models/inswapper_128.onnx` to `models/inswapper_128_fp16.onnx`.

The GUI checks the selected models folder for `face_parser.onnx`; if absent, the loader also tries the default `models/face_parser.onnx` beside the source app/executable. If neither loads, it uses box masks. The CLI always uses that default parser path.

Put source photos in `images\`, or choose another faces folder/image in the app. The largest detected source face supplies the identity; all detected target faces are swapped. First use needs internet if InsightFace must fetch `buffalo_l`. For an offline PC, populate the target user's InsightFace cache first; the executable does not bundle that pack.

### 5. Check the setup

```bat
python tools/check_environment.py
python app.py --self-test
```

The environment checker reports Python, TensorRT bindings/DLLs, ONNX Runtime providers, PyTorch CUDA, and selected DLL/PATH checks. It does not test every dependency or actual model inference. The self-test loads configured models, runs detection and a synthetic swap, constructs a hidden Qt window, and reports FFmpeg availability without opening a camera.

Self-test exit codes: `0` = GPU inference, `1` = no swap model, `2` = CPU fallback. Other loading errors produce a traceback/nonzero exit. Optional enhancer/parser failure is reported but does not by itself fail the test.

## Desktop App

Double-click **`Deepfake.bat`**, or run:

```bat
python app.py
```

The launcher uses `venv\Scripts\pythonw.exe`. The app asks for a models folder only if the configured folder has no recognized swap model. It remembers folders, source face, models, and settings in `settings.json` at the repository root (next to the executable in a packaged build).

Press **Tab/F1** to open the overlay:

- **Source face**: click a thumbnail, choose a faces folder, or open another image. Thumbnails follow files added/removed in that folder.
- **Face swap**: on/off, model, separate live/file pixel boost, box/parser mask, and fringe mode.
- **Face enhancer**: on/off, model, and blend from 0 to 1.
- **Face detection**: smoothing, detector size 320/480/640, and detection every 1-10 frames.
- **Camera**: index 0-9, resolution 640x480 / 1280x720 / 1920x1080, and FPS 15/24/30/60. Click **Apply camera settings** to reopen it. Mirror, upside-down flip, virtual camera, and 1-8 workers apply separately.
- **Recording and files**: start/stop recording, process an image/video/GIF, cancel a video/GIF job, and open outputs.
- **Folders and HUD**: change models/outputs folders and show/hide the HUD. After changing the models folder, pick a model to load it.

Defaults: inswapper FP16 if available (otherwise the first recognized swap model), live boost **128**, file boost **512**, box mask, fringe `blend`, enhancer **off** with blend **0.8**, smoothing **on**, detector **640**, detection interval **1**, camera **0**, requested **1920x1080 at 30 FPS**, **4 workers**, HUD **on**. Mirror, upside-down flip, and virtual camera start **off**. HyperSwap's effective live boost is at least 256.

Most settings apply immediately; camera index/resolution/FPS need the Apply button, and model loading takes time. Offline jobs capture their own models and quality settings, so later live changes do not change the file being processed. Cancelling keeps the partial export when frames have been produced. Closing stops engine builds, cancels/finalizes an active file job, and finishes recording before exit.

### Keyboard shortcuts

| Key | Action |
|---|---|
| **Tab / F1** | Toggle settings overlay |
| **Esc** | Close settings; if already closed, leave fullscreen |
| **F11** | Toggle fullscreen |
| **N / P** | Next / previous source face in the faces folder |
| **X** | Toggle swapping |
| **E** | Toggle enhancer, if loaded |
| **B** | Cycle live boost: 128/256/384/512 for inswapper, 256/512 for HyperSwap |
| **M** | Toggle box/parser mask, if parser available |
| **F** | Cycle fringe `blend` / `hide` |
| **R** | Start/stop recording |
| **U** | Show/hide HUD |
| **Q** | Quit |

### First-time engines and model switching

First use can take **1-4 minutes per model** while TensorRT prepares it. This includes the detector/recognizer and any installed parser/enhancer loaded at startup, even if enhancement is off.

`src/engines.py` builds in a separate process because creating a TensorRT `InferenceSession` can hold Python's GIL and freeze the window. The builder saves a small `model_ctx.onnx` plus engine files in `trt_cache/context/`. Subsequent EP-context loads take roughly **0.05-0.6 seconds** in the author's setup; total startup also includes imports and camera initialization.

Initial startup shows loading status. When switching to an uncached model, the window stays responsive but **live swapping pauses and the original camera feed is shown**, leaving the GPU free for the build. Cached switches are quick; already submitted frames keep their previous model/settings snapshot.

If TensorRT cannot run a whole model, that result is remembered and CUDA is used, with CPU fallback if needed. A failed cached-engine load triggers one rebuild attempt. `app.py --build-engine MODEL OUT_DIR FP16` is the internal subprocess entry point, not a normal launch command.

### Recording and virtual camera

Press **R** to start/stop recording. Output is `output_<timestamp>.mp4` in the selected outputs folder. FFmpeg encodes H.264 and adds microphone audio when PyAudio and an input device are available. Timing is adjusted to the measured frame rate when recording finishes; the HUD is excluded.

For calls, install OBS Studio (or another supported virtual camera driver), enable **Virtual camera (OBS)** in settings, and select that device in Discord/Zoom/Teams. Output uses the webcam's actual resolution, which can differ from the request. The CLI equivalent is:

```bat
python main.py --source images/face.jpg --max-workers 4 --virtual-cam
```

## Command Line

Use `main.py` for webcam mode or one offline target at a time:

```bat
python main.py --source images/my_photo.jpg
python main.py --source images/my_photo.jpg --image images/target.jpg
python main.py --source images/my_photo.jpg --video images/my_video.mp4
python main.py --source images/my_photo.jpg --gif images/animation.gif
python main.py --help
```

Without a target flag, it opens the webcam. `--image`, `--video`, and `--gif` are mutually exclusive. The GUI's `settings.json` does not configure the CLI.

- **Images**: saved in the target's supported image format, then displayed until any key is pressed.
- **Videos**: detection on frames used for output, independent of `--detect-interval`. Decoded timestamps are resampled to a constant frame rate by repeating/dropping frames. ffprobe supplies nominal rate/duration when available; otherwise OpenCV's average rate is used. FFmpeg encodes once to H.264 (CRF 18), re-encoding the first source audio stream to AAC at 192 kbps. Without FFmpeg, output uses OpenCV `mp4v` with no audio.
- **GIFs**: original per-frame delays (100 ms fallback when absent), saved as an infinitely looping GIF. Frames are held in memory until saving.

Default exports go to the repository's `outputs/` as `processed_<target>_with_<source>_<timestamp>.<ext>`: `.mp4` for video, `.gif` for GIF, target extension for images. `--out` overrides the path; do not point it at the input. Webcam recordings use `output_<timestamp>.mp4` unless `--out` is given.

The helper activates the repo's `venv` and starts CLI webcam mode with one source argument:

```bat
scripts\run.bat images/my_photo.jpg
```

### Arguments and defaults

Paths shown as `models/...` default to the repository's models folder. User-supplied relative paths resolve from the current directory.

| Argument | Values / default |
|---|---|
| `-h`, `--help` | Show help and exit |
| `--source` | Required source image path |
| `--model` | Swap model; default `models/inswapper_128_fp16.onnx` |
| `--image` / `--video` / `--gif` | One offline target; none by default (webcam) |
| `--out` | File/recording output path; automatic name by default |
| `--pixel-boost` | 128, 256, 384, 512; defaults to native size (inswapper 128, HyperSwap 256). HyperSwap clamps 128 to 256 and 384 to 256. |
| `--face-mask` | `box` (default), `parser`; parser requires `models/face_parser.onnx`, otherwise box blending is used |
| `--fringe` | `blend` (default), `hide` |
| `--enhance` | Enable enhancement at startup; off by default, ignored with a warning if no enhancer loads |
| `--enhancer-model` | Default `models/GFPGANv1.4.onnx`; e.g. `models/GPEN-BFR-512.onnx` |
| `--enhance-blend` | 0-1; default 0.8 |
| `--det-size` | 320, 480, 640; default 640 |
| `--no-smoothing` | Disable smoothing for webcam/video/GIF; on by default |
| `--max-workers` | Default automatic: `min(os.cpu_count() or 4, 5)`. Live swapping only; files run synchronously. |
| `--detect-interval` | At least 1; default 1. Webcam only: detect every N frames. |
| `--camera` | Webcam index; default 0 |
| `--width` / `--height` | Requested webcam dimensions; default 1920 / 1080 |
| `--camera-fps` | Requested webcam FPS; default 30 |
| `--mirror` | Flip webcam left-right; off by default |
| `--flip` | Flip webcam upside down; off by default |
| `--virtual-cam` | Virtual camera output; off by default |

### Webcam controls

Focus the OpenCV preview window. These are lowercase keys:

| Key | Action |
|---|---|
| **q** | Quit and finish recording |
| **n / p** | Next / previous source image in the initial source image's folder |
| **x** | Toggle swapping |
| **e** | Toggle loaded enhancer |
| **b** | Cycle model-compatible pixel boost |
| **m** | Toggle box/parser mask if parser loaded |
| **r** | Start/stop recording |
| **u** | Show/hide HUD |

Fringe is set with `--fringe` at startup; the CLI has no `f` shortcut or desktop settings overlay.

## Improving Quality

inswapper produces a 128x128 face, which looks soft when scaled up to a large face in a 1080p frame. Pixel boost, enhancement, and native 256x256 models offer different speed/detail tradeoffs.

### Pixel boost

`--pixel-boost 256/384/512` (or B/b) splits a high-resolution crop into interleaved model-sized sub-images, swaps each, and weaves them back together. Cost is `(crop size / model size)^2` runs per face. It can keep more detail, but does not turn inswapper into a native 512 model.

Author's RTX 4060 Ti measurements, inswapper FP16:

| Pixel boost | Model runs per face | Swap time per face | Use |
|---|---|---|---|
| 128 | 1 | ~7 ms | Webcam |
| 256 | 4 | ~27 ms | Webcam (one face) / files |
| 384 | 9 | ~60 ms | Files |
| 512 | 16 | ~110 ms | Files |

### Face enhancer and masks

GFPGAN/GPEN restores detail in FFHQ-aligned crops and blends it back. It can look too "plastic", especially in poor light; lower `--enhance-blend` (try 0.5) if needed. Compatible ONNX models must use the expected RGB input/output layout and normalization; a matching filename alone does not guarantee compatibility.

The enhancer uses **TensorRT FP32** to avoid FP16 artifacts, measured at **~40 ms per face** for GFPGAN. It uses the parser mask when available and a feathered box mask otherwise, independently of the swap's selected mask.

The swap's **box** mask is the fast default. **Parser** masking restricts blending to face regions, helping preserve target hair, glasses, and occlusions; segmentation cannot guarantee preservation of every hand or object. It costs extra inference time.

### Fringe (hair over the forehead)

A source fringe can look see-through after swapping. `--fringe` (F in the GUI) has two modes:

- **Model default** (`blend`): leave the model's result alone.
- **Hide** (`hide`): restore the target's forehead, eyebrows, and hair above the eyes, using Poisson blending to match boundary colors.

The author's measured cost is **+3 ms per face at boost 128**, **+7 ms at 256**.

### Landmark smoothing

Smoothing is on by default for webcam, video, and GIF processing. The One Euro filter smooths strongly while still and less while moving. In the author's video tests it removed about a third of the jitter without measurable lag. Use `--no-smoothing` to compare.

For files, try:

```bat
python main.py --source images/my_photo.jpg --video images/my_video.mp4 --pixel-boost 512 --enhance --enhance-blend 0.6
```

### HyperSwap (256x256 swap model)

[HyperSwap](https://github.com/facefusion/facefusion-labs/tree/master/hyperswap) works at 256x256 natively with ArcFace alignment/identity embeddings. The author measured **~6.5 ms per TensorRT run** on an RTX 4060 Ti, compared with **~7 ms** for inswapper. HyperSwap boost 512 uses four runs per face; supported sizes are 256 and 512.

Download a variant into `models/` and compare likeness on your own photos:

- [hyperswap_1a_256.onnx](https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1a_256.onnx)
- [hyperswap_1b_256.onnx](https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1b_256.onnx)
- [hyperswap_1c_256.onnx](https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1c_256.onnx)

```bat
python main.py --source images/my_photo.jpg --model models/hyperswap_1a_256.onnx
python main.py --source images/my_photo.jpg --model models/hyperswap_1a_256.onnx --video images/my_video.mp4 --pixel-boost 512
```

In the author's self-swap tests (six photos swapped onto themselves), HyperSwap kept more detail and original eye color, which inswapper tended to turn brown:

| Model | ArcFace similarity to source | PSNR vs. original | Time per frame (large photos, including detection) |
|---|---|---|---|
| inswapper_128 | 0.942 | 26.2 dB | 26 ms |
| inswapper_128, pixel boost 512 | 0.944 | 26.3 dB | 139 ms |
| hyperswap_1a_256 | 0.983 | 33.3 dB | 38 ms |
| hyperswap_1a_256, pixel boost 512 | 0.982 | 35.1 dB | 59 ms |

These whole-frame timings include more than a model run and use large photos, so they are not webcam FPS estimates. Self-swaps reward keeping the target face; they do not prove better source-identity transfer.

### Performance settings

Existing measurements on the author's **RTX 4060 Ti + TensorRT**:

- Detector only: **~5.5 ms** at 640x640, **~3 ms** at 320x320 (about 2.5 ms faster, with less precise landmarks).
- inswapper FP16: **~7 ms per face** at boost 128, **~27 ms** at 256.
- GFPGAN FP32: **~40 ms per face**.
- Typical live output: **15-30 FPS**, depending on faces/settings. More workers improve throughput but increase queue delay; fewer workers reduce delay.

The author's practical presets:

```bat
rem Low latency: roughly 12-15 FPS in the author's setup
python main.py --source images/c.jpg --max-workers 2

rem Higher throughput: roughly 25-30 FPS, with more delay
python main.py --source images/c.jpg --max-workers 5 --camera-fps 60

rem Balanced, with less frequent detection
python main.py --source images/c.jpg --max-workers 3 --detect-interval 3

rem Settings that worked well for me
python main.py --source images/c.jpg --max-workers 4
```

Requested camera FPS is not processed FPS. Higher detection intervals reuse the last face positions, so fast movement can look worse. These are the author's existing measurements, not new benchmarks.

## Standalone .exe

Build from the configured source environment:

```bat
scripts\build_exe.bat
```

The script installs PyInstaller in `venv` and runs `packaging/build.py` with `packaging/deepfake.spec`. It collects CUDA/cuDNN DLLs from PyTorch and TensorRT DLLs from the detected install, creating **`dist\Deepfake\Deepfake.exe`** plus **`_internal\`**.

- Copy/zip the **whole folder**, not just the executable. Python/GPU runtimes are bundled; the target needs a compatible NVIDIA driver for GPU inference. Hardware and virtual camera drivers are still needed for those features.
- The one-folder build is roughly **4.2 GB before user models/media and engine cache**. A one-file build would unpack that on launch, so this project uses one-folder packaging.
- Models are separate. Put ONNX files in `dist\Deepfake\models\`, or choose another folder. InsightFace's `buffalo_l` pack is still in the target user's `~/.insightface` cache and may need a first-run download.
- Keep the folder writable. `settings.json`, `deepfake.log`, `trt_cache\`, and default `models\`, `images\`, `outputs\` live next to the executable; avoid `Program Files` unless you handle write permissions.
- Engines are GPU/runtime-specific and may rebuild on another PC (roughly 1-4 minutes per model). A rejected engine is rebuilt once, then falls back to CUDA if necessary.
- Rebuilding through the script preserves `settings.json`, `deepfake.log`, `trt_cache\`, `models\`, `images\`, and `outputs\` from the previous distribution. It does not copy source-repo models/settings into the distribution.
- **`Deepfake.exe --self-test`** checks model/GPU/Qt loading without a camera; results go to `deepfake.log`. Bundled `imageio-ffmpeg` supplies FFmpeg when PATH has none; ffprobe still needs a separate install for video probing.

## Utilities and Project Structure

Run utilities from the repository root with the virtual environment active.

| Command | What it does |
|---|---|
| `python tools/check_environment.py` | GPU runtimes, bindings, providers, and selected DLL/PATH checks; see [Check the setup](#5-check-the-setup). |
| `python app.py --self-test` | Configured model loading, synthetic inference, and hidden Qt construction. May build engines on first use. |
| `python tools/benchmark_model.py --model models/hyperswap_1a_256.onnx` | Five warmups, 50 random-input runs; mean inference time and reciprocal FPS, not pipeline FPS. Defaults to inswapper FP16. Uses direct ONNX Runtime loading, not the app's EP-context loader. |
| `python tools/convert_fp16_v2.py` | Converts only `models/inswapper_128.onnx` to `models/inswapper_128_fp16.onnx`, keeping input/output types. FP16 was about 2x faster in the author's setup. |
| `python tools/fix_trt_dlls.py` | Copies detected TensorRT DLLs into ONNX Runtime's `capi` directory. Usually unnecessary with automatic discovery. |
| `python tools/inspect_model.py --model models/example_model.onnx` | Loads on CPU and writes input/output signature to `model_info.txt` in the current directory. |
| `python tools/smoke_test_parser.py` | First supported photo in `images/`, with `models/face_parser.onnx`; checks mask shape/range/nonemptiness and saves `outputs/parser_smoke.jpg`. |

Tracked files and runtime folders:

```text
deepfake/
|-- .gitignore
|-- README.md
|-- app.py                         # Desktop, self-test, internal engine builder
|-- Deepfake.bat                   # Desktop launcher via venv pythonw
|-- main.py                        # CLI webcam/offline modes
|-- requirements.txt
|-- src/
|   |-- camera.py                  # Threaded DirectShow capture
|   |-- engines.py                 # Subprocess builds, EP-context sessions
|   |-- swapper.py                 # SwapModel, pixel boost, SwapJob snapshots
|   |-- enhancer.py                # FFHQ-aligned GFPGAN/GPEN
|   |-- face_parser.py             # BiSeNet anatomical masks
|   |-- face_utils.py              # Alignment, masks, paste-back, forehead blending
|   |-- tracking.py                # One Euro landmark smoothing
|   |-- media.py                   # Image/video/GIF, VFR resampling, audio
|   |-- recorder.py                # Webcam/microphone recording
|   |-- utils.py                   # Paths, DLLs, providers, FFmpeg lookup
|   `-- gui/
|       |-- __init__.py
|       |-- main_window.py         # Preview, overlay, shortcuts, file jobs
|       |-- pipeline.py            # Live processing, virtual camera, recording
|       `-- settings.py            # JSON defaults, validation, file discovery
|-- packaging/
|   |-- build.py                   # Preserves distribution user files
|   |-- deepfake.spec              # PyInstaller one-folder spec
|   `-- rthook_env.py              # Packaged startup environment
|-- scripts/
|   |-- install_dependencies.bat
|   |-- build_exe.bat
|   `-- run.bat
|-- tools/
|   |-- check_environment.py
|   |-- benchmark_model.py
|   |-- convert_fp16_v2.py
|   |-- fix_trt_dlls.py
|   |-- inspect_model.py
|   `-- smoke_test_parser.py
|-- models/.gitkeep                # Downloaded swap/enhancer/parser models
|-- images/.gitkeep                # Source photos, optional target media
|-- outputs/.gitkeep               # Default exports/recordings
|-- venv/                          # Local environment (not tracked)
|-- trt_cache/context/             # Generated contexts, engines, cache metadata
|-- settings.json                  # Generated desktop settings
|-- deepfake.log                   # Generated on windowless launches
|-- model_info.txt                 # Generated by inspect_model.py
|-- build/                         # PyInstaller work files
`-- dist/Deepfake/                 # Packaged app and writable user folders
```

## Troubleshooting

### The launcher closes, or models fail to load

Read **`deepfake.log`** in the repository root after a `Deepfake.bat`/pythonw launch, or next to `Deepfake.exe`. Windowless launches overwrite it each time. `python app.py` and `python app.py --self-test` print to the terminal instead.

Desktop configuration is **`settings.json`** in the same location. Close the app and rename/remove it to reset defaults, then choose folders again. This does not reset CLI options or remove models/engines. If you only have original inswapper or HyperSwap, pass its path with `--model` instead of the CLI's FP16 default.

If InsightFace cannot download `buffalo_l`, check network access or populate `~/.insightface/models/buffalo_l/` with the detector/recognizer. These are separate from the GUI models folder.

### Missing TensorRT DLLs or providers

For `nvinfer_10.dll` errors, verify `%TENSORRT_DIR%\lib\nvinfer_10.dll` exists and the variable points to the inner extracted directory. Restart after `setx`. Check CUDA/cuDNN availability and runtime compatibility. The [installation step](#3-set-up-tensorrt) lists exact auto-detected paths.

```bat
python tools/check_environment.py
python app.py --self-test
```

If the checker cannot import `tensorrt` after installing only the lean wheel, install the full matching wheel. A green summary alone is not proof of GPU inference: inspect the self-test provider (`TensorrtExecutionProvider` or `CUDAExecutionProvider`). For missing CUDA, check the active environment's `onnxruntime-gpu` and matching DLLs. If TensorRT fails but CUDA works, the app can still run on CUDA.

`python tools/fix_trt_dlls.py` is a manual copy workaround. For missing `zlibwapi.dll`, check PyTorch's `lib\` and PATH; reinstall the CUDA wheel if damaged:

```bat
python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Force an engine rebuild

Close the app and remove **`trt_cache/context/`** at the repository root (or next to the executable), then relaunch. This also clears remembered "not fully supported by TensorRT" results. Preparation will take minutes again.

Cache folders depend on model path, size/modification time, precision, and ONNX Runtime version. Changed files/runtime versions get new folders. GPU/driver/TensorRT incompatibility is handled when loading fails; avoid sharing caches between unlike systems.

### Low FPS or too much delay

- Use native boost (128 for inswapper, 256 for HyperSwap), box masks, and turn enhancement off.
- Compare detector 320 with 640, or detection interval 2 with 1. Faster settings can reduce landmark accuracy or tracking.
- Reduce workers for less delay; increase for throughput. Compare the [performance presets](#performance-settings).
- Check the self-test provider and delivered camera resolution/FPS. Requesting 60 FPS cannot make a 30 FPS camera or slow pipeline produce 60 FPS.

### Audio, VFR video, or recording problems

FFmpeg lookup uses PATH, then `imageio-ffmpeg`. For phone/VFR clips, install ffprobe on PATH too; otherwise rate/duration probing falls back to OpenCV. Audio is re-encoded, not copied bit-for-bit.

Microphone failure does not prevent video recording. If FFmpeg is missing or final encoding fails, the raw `recording_*.avi` is kept next to the output (nominal 30 FPS), plus a separate WAV if recorded; the final MP4 may not be produced. Offline video has a separate silent `mp4v` fallback without FFmpeg.

For camera errors, choose another index and apply settings, or close another app using the device. For virtual camera errors, confirm the driver is installed and select it in your calling app.

## Notes

- Model weights are separate and have their own licenses. Check the terms accompanying downloaded inswapper, InsightFace, HyperSwap, and enhancer models, especially before commercial use.
- Settings, media folders, logs, and caches are local to each source/distribution location. CLI arguments and repository-relative defaults are separate from desktop settings.
- Benchmarks are retained from the author's RTX 4060 Ti tests. Model inference, per-face swapping, and whole-frame processing measure different workloads; actual FPS depends on hardware, faces, camera, and quality settings.
