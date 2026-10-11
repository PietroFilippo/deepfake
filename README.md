# Real-Time Deepfake for Webcam, Videos, and Images

Face swapping application that works in real time on a webcam feed, as well as on videos and static images, with GPU acceleration via TensorRT and CUDA.

## Features

- **Optimized performance**: Uses TensorRT for accelerated GPU inference
- **Asynchronous processing**: Multi-threaded pipeline with a frame buffer
- **Smart detection**: Face detection with periodic downscaling
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

### 6. Verify the environment
```bash
python tools/check_environment.py
```

You should see: **ENVIRONMENT CONFIGURED CORRECTLY**

## Project Structure

```
deepfake/
├── src/                       # Core modules
│   ├── camera.py              # Threaded webcam/video capture
│   ├── swapper.py             # Face detection and swapping
│   └── utils.py               # Shared utilities (DLL setup, providers)
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
- Offline processing for maximum quality
- Original audio preserved (via MoviePy/FFmpeg)
- Real-time progress bar
- Automatically saved to `outputs/` with a unique name

#### 3. Static Image Processing
Swaps the face in an image:
```bash
python main.py --source images/my_photo.jpg --image images/target.jpg
```
- Processes a single image
- Automatically saved to `outputs/`
- Displays the result on screen

#### 4. GIF Processing
Swaps the face in an animated GIF file:
```bash
python main.py --source images/my_photo.jpg --gif images/animation.gif
```
- Processes frame by frame
- Keeps the original speed (FPS)
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
- `--enhance`: Enables GFPGAN by default on startup (can be used with videos and images)

#### Performance Arguments
- `--max-workers`: Number of parallel threads (default: auto).
  - Lower (2-3) = Lower latency (less delay), lower FPS.
  - Higher (5-6) = Higher FPS, higher latency (more delay).
- `--detect-interval`: Frame interval for face detection (default: 5).
  - Increasing it (e.g. 10) reduces CPU usage and can increase GPU FPS.
- `--camera-fps`: Requests a specific FPS from the webcam (default: 30).
- `--virtual-cam`: Enables output to OBS Virtual Camera (useful for Discord, Zoom, etc.).

### Controls (Webcam Mode)
- **q**: Quit the application
- **n**: Switch to the next image in the `images/` folder
- **p**: Switch to the previous image in the `images/` folder
- **x**: Toggle face swapping on/off (shows the original webcam feed)
- **e**: Toggle face "enhancement" on/off (GFPGAN)
- **r**: Start/stop recording (saved to `outputs/`)
- **u**: Show/hide the UI overlay (FPS, status, REC indicator)

## Face "Enhancement" (Face Enhancer)

The project supports face enhancement using the **GFPGAN** model to fix blurry faces at higher resolutions. It increases the resolution, but makes faces look more "plastic". Quality is much worse in the dark, but it can look better up close; it depends on your webcam's quality.

- **Model**: `models/GFPGANv1.4.onnx` (downloaded separately, like the other models)
- **Activation**: Press the `e` key while the app is running.
- **Performance**: This enhancer runs in **FP32** (Full Precision) on TensorRT to avoid visual artifacts. It uses more resources than the swapper (FP16), so expect an FPS drop when it's enabled.

## Virtual Camera (Discord/Zoom/Teams)

To use the deepfake in video calls:

1.  Make sure you have **OBS Studio** installed (it installs the virtual camera driver).
2.  Run with the `--virtual-cam` flag:
    ```bash
    python main.py --source images/face.jpg --max-workers 4 --detect-interval 5 --virtual-cam
    ```
3.  In Discord/Zoom/Teams, select **OBS Virtual Camera** as your camera.

> **Note:** The script supports 1080p (1920x1080). If you want to change the default resolution, edit `src/camera.py` (lines 12 and 13) and `main.py` (line 42).

## Recording and Output

### Recording System
In webcam mode, you can record the processed output:
1. Press **'r'** to start recording
2. A red "REC" indicator will appear on screen
3. Press **'r'** again to stop
4. Microphone audio is recorded and synced automatically
5. The video is saved to `outputs/output_<timestamp>.mp4` (with audio)

### Output Files
All processed files are saved to `outputs/` with unique names:
- **Videos**: `processed_<video>_with_<source>_<timestamp>.mp4`
- **Images**: `processed_<image>_with_<source>_<timestamp>.jpg`
- **Recordings**: `output_<timestamp>.mp4`

### Audio Preservation
When processing videos with `--video`, the original audio is preserved automatically:
- Requires `moviepy` to be installed
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
python main.py --source images/c.jpg --max-workers 3 --detect-interval 8
```

### 4. Settings That Worked Well for Me
```bash
python main.py --source images/c.jpg --max-workers 4 --detect-interval 5
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
2. Reduce the detection resolution: `det_size=(256, 256)` in `src/swapper.py`
3. Increase the detection interval: `detect_interval=10` in `main.py`
4. Check that TensorRT is being used: `python tools/check_environment.py`

### Providers not available
Run `python tools/check_environment.py` for a full diagnosis.
- If `CUDAExecutionProvider` is missing: Reinstall `onnxruntime-gpu`
- If `TensorrtExecutionProvider` is missing: Check your TensorRT installation or set `TENSORRT_DIR`

## Expected Performance

Tested with an RTX 4060 Ti + TensorRT FP16:
- **Detection**: ~15-20ms per frame (at 50% downscale)
- **Swapping**: ~5-10ms per face
- **Target FPS**: 15-30 FPS (depends on the number of faces and settings)

## Notes

- The first run is slow (TensorRT compiles engines and creates a cache in `trt_cache/`)
- Subsequent runs are much faster
- The TensorRT cache is GPU-specific
- FP16 models are ~2x faster than FP32
