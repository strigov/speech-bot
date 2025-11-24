# GigaAM & PyAnnote Telegram Transcriber

A Windows-native Telegram Bot for high-quality audio transcription using GigaAM v3 ASR and PyAnnote speaker diarization. Designed for RTX 5070 Ti (16GB VRAM) and long-form audio processing.

## Features

- **High-Accuracy ASR**: Uses GigaAM v3 for state-of-the-art Russian speech recognition.
- **Speaker Diarization**: Identifies and labels different speakers using PyAnnote 3.1.
- **Long Audio Support**: Processes files up to several hours with smart chunking and memory management.
- **Windows Native**: Optimized for Windows 10/11 with CUDA 12.1 support.
- **Queue System**: Manages concurrent requests with a priority queue and fair usage limits.
- **Admin Tools**: Built-in commands for monitoring GPU usage and queue status.

## System Requirements

- **OS**: Windows 10 or 11 (x64)
- **GPU**: NVIDIA GPU with 12GB+ VRAM (RTX 3080/4070/5070 or better recommended)
- **Driver**: Latest NVIDIA Studio or Game Ready Driver
- **Storage**: SSD with at least 50GB free space (for models and temp files)
- **Software**:
  - Python 3.10.x (Strict requirement)
  - FFmpeg (Must be in system PATH)
  - Git (Optional, for cloning)

## Installation

1. **Clone or Download** the repository to a folder (e.g., `C:\Projects\speech-bot`).

2. **Run Setup Script**:
   Double-click `setup_environment.bat`. This will:
   - Create a virtual environment (`venv`)
   - Install PyTorch with CUDA support
   - Install all Python dependencies
   - Create necessary directories (`temp`, `logs`)
   - Create a `.env` file from the template

3. **Configure Environment**:
   Open `.env` in a text editor (Notepad, VS Code) and set your tokens:
   ```ini
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   HF_TOKEN=your_hugging_face_token
   ```
   *Note: You need to accept the user agreement for `pyannote/speaker-diarization-3.1` on Hugging Face to get a valid token.*

## Usage

### Starting the Bot
Double-click `start.bat`.
The console will show the initialization process. Once you see "Bot started!", it is ready to receive messages.

### Bot Commands
- `/start` - Welcome message and status check.
- `/help` - Show usage instructions.
- `/status` - Check your current position in the queue.
- `/cancel` - Cancel your current processing task.

### Admin Commands
- `/admin gpu` - Show current GPU memory usage.
- `/admin queue` - Show queue statistics.

## Configuration

You can tweak settings in `config/models.yaml` and `config/limits.yaml` (if available) or directly in `.env`:

- **MAX_VRAM_GB**: Limit GPU memory usage (default: 14.0 for 16GB cards).
- **MAX_AUDIO_DURATION_MINUTES**: Maximum allowed audio length (default: 180).
- **MAX_USER_CONCURRENT**: Max files per user in queue (default: 3).

## Troubleshooting

**"CUDA not available"**
- Run `setup_cuda.bat` to verify your driver and CUDA installation.
- Ensure you installed the correct PyTorch version (the setup script handles this).

**"OOM (Out of Memory)"**
- Reduce `ASR_BATCH_SIZE` or `DIARIZATION_BATCH_SIZE` in `.env`.
- Lower `MAX_VRAM_GB` to leave more headroom for the system.

**"FFmpeg not found"**
- Install FFmpeg and add `bin` folder to your Windows PATH environment variable.
- Restart the terminal/script after installing.

## License
[License Name]
