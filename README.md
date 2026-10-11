# Real-Time Deepfake for Webcam, Videos, and Images

Face swapping application that works in real time on a webcam feed, as well as on videos and static images, with GPU acceleration via TensorRT and CUDA.

## Features

- **Optimized performance**: Uses TensorRT for accelerated GPU inference
- **Asynchronous processing**: Multi-threaded pipeline with a frame buffer
- **Fast detection**: Only the face detector runs on target frames; blending only touches the face region
- **Pixel boost**: Swaps the face at up to 512x512 with the 128px inswapper model for sharper results
- **Face enhancer**: GFPGAN or GPEN on properly aligned faces, with an adjustable blend
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
│   ├── camera.py              # Threaded webcam capture
│   ├── swapper.py             # Face detection, swapping and pixel boost
│   ├── enhancer.py            # Face enhancer (GFPGAN / GPEN)
│   ├── face_parser.py         # BiSeNet face parsing masks
│   ├── face_utils.py          # Alignment, masks and paste-back
│   ├── media.py               # Offline image/video/GIF processing
│   ├── recorder.py            # Webcam recording with microphone audio
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
├── main.py                    # Main application
└── requirements.txt           # Python dependencies
```

## Usage

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
- `--model`: Path to the inswapper model (default: `models/inswapper_128_fp16.onnx`)
- `--video`: Path to the input video (offline processing)
- `--image`: Path to the input image (static processing)
- `--gif`: Path to the input GIF (animated processing)
- `--out`: Custom path for the output file

#### Quality Arguments
- `--pixel-boost`: Resolution the face is swapped at: `128` (default), `256`, `384` or `512`. See [Improving Quality](#improving-quality).
- `--face-mask`: `box` (default) or `parser`. `parser` keeps hair, hands and glasses that cover the target face (slower).
- `--enhance`: Enables the face enhancer by default on startup (can be used with videos and images)
- `--enhancer-model`: Enhancer model (default: `models/GFPGANv1.4.onnx`, e.g. `models/GPEN-BFR-512.onnx`)
- `--enhance-blend`: How much of the enhanced face is mixed in, 0-1 (default: 0.8). Lower = less "plastic".
- `--det-size`: Face detector input size (default: 320 for the webcam, 640 for files).

#### Webcam / Performance Arguments
- `--max-workers`: Number of parallel threads (default: auto).
  - Lower (2-3) = Lower latency (less delay), lower FPS.
  - Higher (5-6) = Higher FPS, higher latency (more delay).
- `--detect-interval`: Frame interval for face detection (default: 2).
  - `1` = the face follows your movements most closely; higher values save a few ms per frame.
- `--camera`: Webcam index (default: 0).
- `--width` / `--height`: Requested webcam resolution (default: 1920x1080).
- `--camera-fps`: Requests a specific FPS from the webcam (default: 30).
- `--virtual-cam`: Enables output to OBS Virtual Camera (useful for Discord, Zoom, etc.).

### Controls (Webcam Mode)
- **q**: Quit the application
- **n**: Switch to the next image in the `images/` folder
- **p**: Switch to the previous image in the `images/` folder
- **x**: Toggle face swapping on/off (shows the original webcam feed)
- **e**: Toggle face "enhancement" on/off
- **b**: Cycle pixel boost (128 → 256 → 384 → 512)
- **m**: Toggle the face mask (box/parser)
- **r**: Start/stop recording (saved to `outputs/`)
- **u**: Show/hide the UI overlay (FPS, status, REC indicator)

## Improving Quality

`inswapper_128` produces a 128x128 face, which looks soft once it is scaled up to a face that fills a 1080p frame. In order of cost:

1. **Pixel boost** (`--pixel-boost 256/384/512`, or the `b` key): the face is cropped at a higher resolution and split into interleaved 128x128 sub-images (every 2nd/3rd/4th pixel). Each one is swapped and the results are woven back together, so the swapped face keeps 2-4x more detail with the same model and the same identity quality. It costs (size/128)² model runs per face:

   | Pixel boost | Model runs per face | Swap time per face (RTX 4060 Ti, FP16) | Use |
   |---|---|---|---|
   | 128 | 1 | ~7 ms | Webcam |
   | 256 | 4 | ~27 ms | Webcam (one face) / files |
   | 384 | 9 | ~60 ms | Files |
   | 512 | 16 | ~110 ms | Files |

2. **Face enhancer** (`--enhance`, or the `e` key): restores detail with GFPGAN (or GPEN with `--enhancer-model models/GPEN-BFR-512.onnx`). Use `--enhance-blend 0.5` if the result looks too "plastic".
3. **Parser mask** (`--face-mask parser`, or the `m` key): keeps whatever is in front of the target face (hair, hands, glasses).

Recommended for files:
```bash
python main.py --source images/my_photo.jpg --video images/my_video.mp4 --pixel-boost 512 --enhance --enhance-blend 0.6
```

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
python tools/benchmark_model.py
```
Measures average inference time and estimated FPS.

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
3. Reduce the detection resolution: `--det-size 256`
4. Increase the detection interval: `--detect-interval 5`
5. Check that TensorRT is being used: `python tools/check_environment.py`

### Providers not available
Run `python tools/check_environment.py` for a full diagnosis.
- If `CUDAExecutionProvider` is missing: Reinstall `onnxruntime-gpu`
- If `TensorrtExecutionProvider` is missing: Check your TensorRT installation or set `TENSORRT_DIR`

## Expected Performance

Tested with an RTX 4060 Ti + TensorRT FP16:
- **Detection**: ~3ms per frame (detector only, 320x320)
- **Swapping**: ~7ms per face (pixel boost 128), ~27ms (256)
- **Enhancer**: ~40ms per face (GFPGAN, FP32)
- **Target FPS**: 15-30 FPS (depends on the number of faces and settings)

## Notes

- The first run is slow (TensorRT compiles engines and creates a cache in `trt_cache/`)
- Subsequent runs are much faster
- The TensorRT cache is GPU-specific
- FP16 models are ~2x faster than FP32
