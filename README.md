# Real-Time Deepfake for Webcam, Videos, and Images

Face swapping application that works in real time on a webcam feed, as well as on videos and static images, with GPU acceleration via TensorRT and CUDA.

## Features

- **Desktop app**: Live preview with a settings overlay (Tab) to change faces, models and options at any time
- **Optimized performance**: Uses TensorRT for accelerated GPU inference
- **Asynchronous processing**: Multi-threaded pipeline with a frame buffer
- **Fast detection**: Only the face detector runs on target frames; blending only touches the face region
- **Pixel boost**: Swaps the face at up to 512x512 with the 128px inswapper model for sharper results
- **Face enhancer**: GFPGAN or GPEN on properly aligned faces, with an adjustable blend
- **Landmark smoothing**: Faces are followed between frames to remove jitter (no shimmering face)
- **Multiple provider support**: TensorRT → CUDA → CPU (automatic fallback)
- **Multiple input modes**: Real-time webcam, videos, GIFs, or static images
- **Audio preservation**: Keeps the original audio when processing videos
- **Recording system**: Record the processed output in real time

## Prerequisites

### Hardware
- NVIDIA GPU with CUDA support
- Webcam (optional)
- Microphone for audio recording (optional)

### Software
- **Python 3.11**
- **FFmpeg** (Added to the system PATH) - Essential for audio ([Download](https://www.ffmpeg.org/download.html))
- **CUDA 12.4+** (Added to the system PATH) ([Download](https://developer.nvidia.com/cuda-downloads))
- **TensorRT 10.4.0** (Added to the system PATH) ([Download](https://developer.nvidia.com/tensorrt)) - Requires an NVIDIA login
  - Extract it to one of the default locations:
    - `C:\Program Files\TensorRT-10.4.0.26\` (recommended)
    - `C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\`
    - `C:\TensorRT\`
  - Or set the environment variable: `TENSORRT_DIR=<your_tensorrt_path>`

## Installation

### 1. Clone the repository
```bash
git clone <repository>
cd deepfake
```

### 2. Create the virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install the dependencies
```bash
scripts\install_dependencies.bat
```

Or manually:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 4. Install TensorRT after downloading and extracting the folder
```bash
pip install "C:\Program Files\TensorRT-10.4.0.26\python\tensorrt_lean-10.4.0-cp311-none-win_amd64.whl"
```

### 5. Download the required models separately
Place the `GFPGANv1.4.onnx` and `inswapper_128.onnx` files (or its FP16 version, `inswapper_128_fp16.onnx`, if you already have it) in the `models/` folder.

Optional:
- `hyperswap_1a_256.onnx` (or `1b`/`1c`): higher-resolution swap model, see [HyperSwap](#hyperswap-256x256-swap-model).
- `face_parser.onnx` (BiSeNet face parsing, 512x512): used for the anatomical enhancer mask and `--face-mask parser`.
- `GPEN-BFR-512.onnx`: alternative face enhancer (`--enhancer-model models/GPEN-BFR-512.onnx`).

### 6. Verify the environment
```bash
python tools/check_environment.py
```

You should see: **ENVIRONMENT CONFIGURED CORRECTLY**

## Project Structure

```
deepfake/
├── src/                       # Core modules
│   ├── gui/                   # Desktop app (window, settings overlay, live pipeline)
│   ├── camera.py              # Threaded webcam capture
│   ├── swapper.py             # Face detection, swapping and pixel boost
│   ├── enhancer.py            # Face enhancer (GFPGAN / GPEN)
│   ├── face_parser.py         # BiSeNet face parsing masks
│   ├── face_utils.py          # Alignment, masks and paste-back
│   ├── media.py               # Offline image/video/GIF processing
│   ├── recorder.py            # Webcam recording with microphone audio
│   ├── tracking.py            # Landmark smoothing between frames
│   └── utils.py               # Shared utilities (paths, DLL setup, providers)
├── tools/                     # Python utilities
│   ├── check_environment.py   # Full diagnostics
│   ├── benchmark_model.py     # Inference benchmark
│   ├── convert_fp16_v2.py     # FP16 conversion
│   ├── fix_trt_dlls.py        # Copies TensorRT DLLs
│   └── inspect_model.py       # ONNX model inspection
├── scripts/                   # Automation scripts
│   ├── install_dependencies.bat
│   └── run.bat
├── models/                    # ONNX models (inswapper)
├── images/                    # Source images
├── outputs/                   # Processed videos and images
├── app.py                     # Desktop app
├── Deepfake.bat               # Double-click launcher for the desktop app
├── main.py                    # Command-line application
└── requirements.txt           # Python dependencies
```

## Usage

### Desktop App (Recommended)

Double-click **`Deepfake.bat`** (or run `python app.py`). The first time, the app asks for your models folder (the one with `inswapper_128_fp16.onnx` / `hyperswap_1a_256.onnx`); it remembers it, together with all other settings, in `settings.json`.

Press **Tab** (or **F1**) to open the settings overlay on top of the live video. Everything applies immediately:
- **Source face**: thumbnails of the images in your faces folder (click one), or any other image
- **Face swap**: swap model (switch between inswapper and HyperSwap at any time), pixel boost for the live view and for files, face mask
- **Face enhancer**: on/off, model (GFPGAN / GPEN) and blend
- **Face detection**: landmark smoothing, detector size, detection interval
- **Camera**: camera index, resolution, FPS, virtual camera (OBS), worker threads
- **Recording and files**: record the live output, process an image/video/GIF (with progress and cancel), open the outputs folder
- **Folders**: models, faces and outputs folders

Keys: **Tab/F1** settings · **Esc** close settings · **F11** fullscreen · **N/P** next/previous face · **X** swap on/off · **E** enhancer · **B** pixel boost · **M** face mask · **R** record · **U** HUD · **Q** quit

The first time a model is used, TensorRT builds an engine for it (from a few seconds up to a few minutes); the live view keeps running with the previous model meanwhile.

### Command Line

### Operating Modes

#### 1. Real-Time Webcam (Default)
```bash
python main.py --source images/my_photo.jpg
```

#### 2. Video Processing (Offline)
Processes a video frame by frame and saves it with the audio preserved:
```bash
python main.py --source images/my_photo.jpg --video images/my_video.mp4
```
- Offline processing for maximum quality (every frame is detected)
- Encoded once with H.264 (CRF 18) by FFmpeg, original audio preserved (if the video has any)
- Real-time progress bar
- Automatically saved to `outputs/` with a unique name

#### 3. Static Image Processing
Swaps the face in an image:
```bash
python main.py --source images/my_photo.jpg --image images/target.jpg
```
- Processes a single image
- Automatically saved to `outputs/` (keeps the target's format)
- Displays the result on screen

#### 4. GIF Processing
Swaps the face in an animated GIF file:
```bash
python main.py --source images/my_photo.jpg --gif images/animation.gif
```
- Processes frame by frame
- Keeps each frame's original delay
- Saved as a new GIF in `outputs/`

### Using the helper script
```bash
scripts\run.bat images/my_photo.jpg
```

### Options
```bash
python main.py --source <image> [options]
```

#### Main Arguments
- `--source`: Image of the face to apply (required)
- `--model`: Path to the swap model (default: `models/inswapper_128_fp16.onnx`). Also supports `models/hyperswap_1a_256.onnx` (1b/1c).
- `--video`: Path to the input video (offline processing)
- `--image`: Path to the input image (static processing)
- `--gif`: Path to the input GIF (animated processing)
- `--out`: Custom path for the output file

#### Quality Arguments
- `--pixel-boost`: Resolution the face is swapped at: `128`, `256`, `384` or `512` (default: the model's own resolution). See [Improving Quality](#improving-quality).
- `--face-mask`: `box` (default) or `parser`. `parser` keeps hair, hands and glasses that cover the target face (slower).
- `--enhance`: Enables the face enhancer by default on startup (can be used with videos and images)
- `--enhancer-model`: Enhancer model (default: `models/GFPGANv1.4.onnx`, e.g. `models/GPEN-BFR-512.onnx`)
- `--enhance-blend`: How much of the enhanced face is mixed in, 0-1 (default: 0.8). Lower = less "plastic".
- `--det-size`: Face detector input size: `320`, `480` or `640` (default: 640, ~5.5 ms; 320 is ~2.5 ms faster but its landmarks are less precise).
- `--no-smoothing`: Disables landmark smoothing between frames (webcam and videos).

#### Webcam / Performance Arguments
- `--max-workers`: Number of parallel threads (default: auto).
  - Lower (2-3) = Lower latency (less delay), lower FPS.
  - Higher (5-6) = Higher FPS, higher latency (more delay).
- `--detect-interval`: Detect faces every N frames (default: 1).
  - `1` = the face follows your movements most closely; higher values save a few ms per frame.
- `--camera`: Webcam index (default: 0).
- `--width` / `--height`: Requested webcam resolution (default: 1920x1080).
- `--camera-fps`: Requests a specific FPS from the webcam (default: 30).
- `--virtual-cam`: Enables output to OBS Virtual Camera (useful for Discord, Zoom, etc.).

### Controls (Command-Line Webcam Mode)
- **q**: Quit the application
- **n**: Switch to the next image in the `images/` folder
- **p**: Switch to the previous image in the `images/` folder
- **x**: Toggle face swapping on/off (shows the original webcam feed)
- **e**: Toggle face "enhancement" on/off
- **b**: Cycle pixel boost (128 → 256 → 384 → 512; 256 → 512 with HyperSwap)
- **m**: Toggle the face mask (box/parser)
- **r**: Start/stop recording (saved to `outputs/`)
- **u**: Show/hide the UI overlay (FPS, status, REC indicator)

## Improving Quality

`inswapper_128` produces a 128x128 face, which looks soft once it is scaled up to a face that fills a 1080p frame. InsightFace never released its 256/512 versions publicly (they are only available commercially through Picsi.ai), so there are two ways around it: pixel boost, and a natively higher-resolution model ([HyperSwap](#hyperswap-256x256-swap-model)). In order of cost:

1. **Pixel boost** (`--pixel-boost 256/384/512`, or the `b` key): the face is cropped at a higher resolution and split into interleaved 128x128 sub-images (every 2nd/3rd/4th pixel). Each one is swapped and the results are woven back together, so the swapped face keeps 2-4x more detail with the same model and the same identity quality. It costs (size/128)² model runs per face:

   | Pixel boost | Model runs per face | Swap time per face (RTX 4060 Ti, FP16) | Use |
   |---|---|---|---|
   | 128 | 1 | ~7 ms | Webcam |
   | 256 | 4 | ~27 ms | Webcam (one face) / files |
   | 384 | 9 | ~60 ms | Files |
   | 512 | 16 | ~110 ms | Files |

2. **Face enhancer** (`--enhance`, or the `e` key): restores detail with GFPGAN (or GPEN with `--enhancer-model models/GPEN-BFR-512.onnx`). Use `--enhance-blend 0.5` if the result looks too "plastic".
3. **Parser mask** (`--face-mask parser`, or the `m` key): keeps whatever is in front of the target face (hair, hands, glasses).

Landmark smoothing is on by default: the detector's 5 landmarks jitter by a few pixels from frame to frame, which makes the swapped face shimmer. Faces are followed between frames and smoothed with a One Euro filter (strong smoothing while still, almost no lag while moving), which removed about a third of the jitter on test videos without measurable lag.

Recommended for files:
```bash
python main.py --source images/my_photo.jpg --video images/my_video.mp4 --pixel-boost 512 --enhance --enhance-blend 0.6
```

### HyperSwap (256x256 swap model)

[HyperSwap](https://github.com/facefusion/facefusion-labs/tree/master/hyperswap) is a face swap model released by the FaceFusion team in 2025. It works at 256x256 natively, uses the same face alignment and identity embedding (ArcFace) as inswapper, and runs about as fast on TensorRT as `inswapper_128` (~6.5 ms per run on an RTX 4060 Ti), so it gives twice the resolution at the same cost. Pixel boost also works with it (`--pixel-boost 512` = 4 runs per face).

Download one of the three variants into `models/` (there is no published ranking between them; try them and keep the one whose likeness you prefer):
- [hyperswap_1a_256.onnx](https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1a_256.onnx)
- [hyperswap_1b_256.onnx](https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1b_256.onnx)
- [hyperswap_1c_256.onnx](https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1c_256.onnx)

```bash
python main.py --source images/my_photo.jpg --model models/hyperswap_1a_256.onnx
python main.py --source images/my_photo.jpg --model models/hyperswap_1a_256.onnx --video images/my_video.mp4 --pixel-boost 512
```

In self-swap tests (each photo swapped onto itself, 6 photos) HyperSwap kept more detail and the original eye color, which inswapper tends to turn brown:

| Model | ArcFace similarity to the source | PSNR vs. original | Time per frame (large photos, incl. detection) |
|---|---|---|---|
| inswapper_128 | 0.942 | 26.2 dB | 26 ms |
| inswapper_128, pixel boost 512 | 0.944 | 26.3 dB | 139 ms |
| hyperswap_1a_256 | 0.983 | 33.3 dB | 38 ms |
| hyperswap_1a_256, pixel boost 512 | 0.982 | 35.1 dB | 59 ms |

Self-swaps reward a model that keeps more of the target face, so they don't prove that the source identity transfers better; compare the likeness on your own photos. HyperSwap is released under a ResearchRAIL license (research use; read its terms before any commercial use). `inswapper_128` is non-commercial too.

## Face "Enhancement" (Face Enhancer)

The project supports face enhancement using the **GFPGAN** model to fix blurry faces at higher resolutions. It increases the resolution, but can make faces look more "plastic" (lower `--enhance-blend` to tone it down). Quality is much worse in the dark, but it can look better up close; it depends on your webcam's quality.

- **Model**: `models/GFPGANv1.4.onnx` (downloaded separately, like the other models). Any GFPGAN-style 512x512 ONNX model also works, e.g. `GPEN-BFR-512.onnx`, via `--enhancer-model`.
- **Alignment**: Faces are aligned to the FFHQ template the model was trained on (using the detector's 5 landmarks) and blended back with the face parser mask.
- **Activation**: Press the `e` key while the app is running.
- **Performance**: This enhancer runs in **FP32** (Full Precision) on TensorRT to avoid visual artifacts (~40 ms per face). It uses more resources than the swapper (FP16), so expect an FPS drop when it's enabled.

## Virtual Camera (Discord/Zoom/Teams)

To use the deepfake in video calls:

1.  Make sure you have **OBS Studio** installed (it installs the virtual camera driver).
2.  Run with the `--virtual-cam` flag:
    ```bash
    python main.py --source images/face.jpg --max-workers 4 --virtual-cam
    ```
3.  In Discord/Zoom/Teams, select **OBS Virtual Camera** as your camera.

> **Note:** The webcam is opened at 1920x1080 by default (use `--width`/`--height` to change it). The virtual camera uses whatever resolution the webcam actually delivers.

## Recording and Output

### Recording System
In webcam mode, you can record the processed output:
1. Press **'r'** to start recording
2. A red "REC" indicator will appear on screen (the UI overlay is never recorded)
3. Press **'r'** again to stop (quitting with **'q'** also finishes the recording)
4. Microphone audio is recorded and synced automatically
5. The video is saved to `outputs/output_<timestamp>.mp4` (H.264, with audio)

### Output Files
All processed files are saved to `outputs/` with unique names:
- **Videos**: `processed_<video>_with_<source>_<timestamp>.mp4`
- **Images**: `processed_<image>_with_<source>_<timestamp>.<ext>`
- **GIFs**: `processed_<gif>_with_<source>_<timestamp>.gif`
- **Recordings**: `output_<timestamp>.mp4`

### Audio Preservation
When processing videos with `--video`, the original audio is preserved automatically:
- Requires FFmpeg on the PATH (without it the video is saved without audio)
- Uses the AAC codec for high-quality audio

## Performance Optimization

Find the ideal balance for your hardware:

### 1. Low Latency Mode (Recommended for interaction)
Prioritizes instant response. FPS will be lower (~12-15), but the video "keeps up" with your movements.
```bash
python main.py --source images/c.jpg --max-workers 2
```

### 2. High FPS Mode (Recommended for recording)
Prioritizes smoothness. FPS will be higher (~25-30), but there will be a small delay.
```bash
python main.py --source images/c.jpg --max-workers 5 --camera-fps 60
```

### 3. Balanced Mode
A middle ground between smoothness and responsiveness.
```bash
python main.py --source images/c.jpg --max-workers 3 --detect-interval 3
```

### 4. Settings That Worked Well for Me
```bash
python main.py --source images/c.jpg --max-workers 4
```

## Utility Scripts

### Environment check
```bash
python tools/check_environment.py
```
Checks TensorRT, CUDA, ONNX Runtime, PyTorch, and all dependencies.

### Model benchmark
```bash
python tools/benchmark_model.py --model models/hyperswap_1a_256.onnx
```
Measures the average inference time and estimated FPS of a swap model (default: `models/inswapper_128_fp16.onnx`).

### FP16 conversion
```bash
python tools/convert_fp16_v2.py
```
Converts ONNX models from FP32 to FP16 (~2x faster).

### TensorRT DLL fix
```bash
python tools/fix_trt_dlls.py
```
Copies TensorRT DLLs to ONNX Runtime (rarely needed thanks to auto-detection).

### ONNX model inspection
```bash
python tools/inspect_model.py --model models/example_model.onnx
```
Inspects the ONNX model's signature and saves it to `model_info.txt`. Useful for checking whether a model is compatible with the system and for implementing support for new models.

## Troubleshooting

### Error: `nvinfer_10.dll not found`
**Option 1 - Install to a default location:**
Install TensorRT in one of the default locations:
- `C:\Program Files\TensorRT-10.4.0.26\TensorRT-10.4.0.26\`
- `C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\`
- `C:\TensorRT\`

**Option 2 - Custom location:**
Set the `TENSORRT_DIR` environment variable:
```bash
setx TENSORRT_DIR "C:\your\path\TensorRT-10.4.0.26"
```

**Option 3 - Manual fix:**
1. Run: `python tools/fix_trt_dlls.py`
2. Or add this to your PATH: `C:\your\path\TensorRT\lib`

### Error: `zlibwapi.dll not found`
`zlibwapi.dll` comes with PyTorch. Reinstall it:
```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Low FPS
1. Use the FP16 model: `python tools/convert_fp16_v2.py`
2. Use `--pixel-boost 128` and turn off the enhancer/parser mask
3. Reduce the detection resolution: `--det-size 320`
4. Increase the detection interval: `--detect-interval 2`
5. Check that TensorRT is being used: `python tools/check_environment.py`

### Providers not available
Run `python tools/check_environment.py` for a full diagnosis.
- If `CUDAExecutionProvider` is missing: Reinstall `onnxruntime-gpu`
- If `TensorrtExecutionProvider` is missing: Check your TensorRT installation or set `TENSORRT_DIR`

## Expected Performance

Tested with an RTX 4060 Ti + TensorRT FP16:
- **Detection**: ~5.5ms per frame (detector only, 640x640; ~3ms at 320x320)
- **Swapping**: ~7ms per face (pixel boost 128), ~27ms (256)
- **Enhancer**: ~40ms per face (GFPGAN, FP32)
- **Target FPS**: 15-30 FPS (depends on the number of faces and settings)

## Notes

- The first run is slow (TensorRT compiles engines and creates a cache in `trt_cache/`)
- Subsequent runs are much faster
- The TensorRT cache is GPU-specific
- FP16 models are ~2x faster than FP32
