"""
Webcam recording: processed video frames plus microphone audio, merged into an MP4 with ffmpeg.
"""
import os
import subprocess
import tempfile
import threading
import time
import wave

import cv2

try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: 'pyaudio' not found. Audio recording disabled.")


def _temp_path(directory, suffix):
    fd, path = tempfile.mkstemp(prefix="recording_", suffix=suffix, dir=directory)
    os.close(fd)
    return path


class AudioRecorder:
    """Records microphone audio to a WAV file in a background thread."""

    def __init__(self, filename, rate=44100, channels=1, frames_per_buffer=1024):
        self.filename = filename
        self.rate = rate
        self.channels = channels
        self.frames_per_buffer = frames_per_buffer
        self.wave_file = None
        self.frames_written = 0
        self.recording = False
        self.thread = None
        self.audio = None
        self.stream = None

        if not AUDIO_AVAILABLE:
            return

        self.format = pyaudio.paInt16
        try:
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(format=self.format,
                                          channels=self.channels,
                                          rate=self.rate,
                                          input=True,
                                          frames_per_buffer=self.frames_per_buffer,
                                          start=False)
        except Exception as e:
            print(f"Error initializing audio: {e}")
            self._close()

    def start(self):
        if not self.stream:
            return
        # Audio is streamed straight to disk instead of being kept in memory
        self.wave_file = wave.open(self.filename, 'wb')
        self.wave_file.setnchannels(self.channels)
        self.wave_file.setsampwidth(pyaudio.get_sample_size(self.format))
        self.wave_file.setframerate(self.rate)
        self.recording = True
        self.stream.start_stream()
        self.thread = threading.Thread(target=self._record, daemon=True)
        self.thread.start()

    def _record(self):
        while self.recording:
            try:
                # Don't raise on input overflow (e.g. when the main loop is busy): that used to
                # end the audio silently and leave the recording shorter than the video
                data = self.stream.read(self.frames_per_buffer, exception_on_overflow=False)
            except Exception as e:
                print(f"Audio recording stopped: {e}")
                break
            self.wave_file.writeframes(data)
            self.frames_written += 1

    def _close(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.audio:
            self.audio.terminate()
            self.audio = None

    def stop(self):
        """Stops recording and writes the WAV file. Returns its path, or None if nothing was recorded."""
        self.recording = False
        # Wait for the reader thread before closing the stream it is reading from
        if self.thread:
            self.thread.join(timeout=2.0)
        self._close()

        if not self.wave_file:
            return None
        self.wave_file.close()
        self.wave_file = None
        if self.frames_written == 0:
            os.remove(self.filename)
            return None
        return self.filename


class Recorder:
    """Records the processed webcam output (and microphone audio) to an MP4 file."""

    def __init__(self, out_path, frame_size, nominal_fps=30.0):
        self.out_path = out_path
        # Unique temporary files next to the output, so nothing else gets overwritten
        out_dir = os.path.dirname(os.path.abspath(out_path))
        self.temp_video = _temp_path(out_dir, ".avi")
        self.temp_audio = _temp_path(out_dir, ".wav")

        # High-quality MJPG intermediate: the video is only compressed for real (H.264) once, by ffmpeg
        self.writer = cv2.VideoWriter(self.temp_video, cv2.VideoWriter_fourcc(*'MJPG'), nominal_fps, frame_size)
        self.writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not create video file {self.temp_video}")

        self.audio_recorder = AudioRecorder(self.temp_audio)
        self.audio_recorder.start()
        self.start_time = time.time()
        self.frame_count = 0
        print(f"Recording to: {out_path}")

    def write(self, frame):
        self.writer.write(frame)
        self.frame_count += 1

    def stop(self):
        self.writer.release()
        elapsed = time.time() - self.start_time
        print("Processing audio.")
        audio_path = self.audio_recorder.stop()
        if not audio_path:
            print("Audio not recorded or error while saving.")
            if os.path.exists(self.temp_audio):
                os.remove(self.temp_audio)

        if self.frame_count == 0:
            os.remove(self.temp_video)
            if audio_path:
                os.remove(audio_path)
            return

        # The frames were written at a nominal 30 FPS; retime them to the real capture rate
        # so the video plays at the right speed and stays in sync with the audio
        actual_fps = self.frame_count / elapsed if elapsed > 0 else 30.0
        print(f"Actual recording FPS: {actual_fps:.2f}")

        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-r', f'{actual_fps:.3f}', '-i', self.temp_video]
        if audio_path:
            cmd += ['-i', audio_path, '-map', '0:v:0', '-map', '1:a:0', '-c:a', 'aac']
        cmd += ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', self.out_path]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            os.remove(self.temp_video)
            if audio_path:
                os.remove(audio_path)
            print(f"Recording finished{' with audio' if audio_path else ''}: {self.out_path}")
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            # Keep the raw recording instead of losing it
            fallback = os.path.splitext(self.out_path)[0] + ".avi"
            os.replace(self.temp_video, fallback)
            if isinstance(e, FileNotFoundError):
                print("Warning: ffmpeg not found.")
            else:
                print(f"Error merging audio: {e.stderr.decode(errors='replace').strip()}")
            print(f"Saved the raw recording (no audio, nominal 30 FPS) to: {fallback}")
            if audio_path:
                print(f"Audio kept at: {audio_path}")
