# GigaAM & PyAnnote Telegram Transcriber

A Windows-native Telegram Bot for high-quality audio transcription using GigaAM v3 ASR and PyAnnote speaker diarization. Designed for RTX 5070 Ti (16GB VRAM) and long-form audio processing.

## Features

- **High-Accuracy ASR**: Uses GigaAM v3 for state-of-the-art Russian speech recognition.
- **Speaker Diarization**: Identifies and labels different speakers using PyAnnote 3.1.
- **Long Audio Support**: Processes files up to several hours with smart chunking and memory management.
- **Windows Native**: Optimized for Windows 10/11 with CUDA 12.1 support.
- **Queue System**: Manages concurrent requests with a priority queue and fair usage limits.
- **Admin Tools**: Built-in commands for monitoring GPU usage and queue status.

## GPU Support: RTX 5070 Ti (Blackwell Architecture)

**IMPORTANT**: The RTX 5070 Ti uses Blackwell architecture (sm_120) which requires **PyTorch NIGHTLY** builds for GPU acceleration. Stable PyTorch releases do not support this GPU yet (as of January 2025).

### What This Means

- GPU acceleration is available via PyTorch nightly (development builds)
- Nightly builds may be unstable and have occasional bugs
- Stable PyTorch support expected Q2-Q3 2025
- PyAnnote was upgraded from 3.3.1 to 4.0.2+ for compatibility

### If You Experience Issues

**Fallback to CPU mode** (slower but stable):

```batch
cd d:\Projects\speech-bot
venv\Scripts\activate.bat
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Then set in `.env`:
```ini
MODEL_DEVICE=cpu
DIARIZATION_DEVICE=cpu
PROCESSING_TIMEOUT_MINUTES=120
```

CPU mode is 10-50x slower but guaranteed to work until stable PyTorch support arrives.

## System Requirements

- **OS**: Windows 10 or 11 (x64)
- **GPU**: NVIDIA GPU with 12GB+ VRAM (RTX 3080/4070/5070 or better recommended)
- **Driver**: NVIDIA Driver 581.80+ (for RTX 5070 Ti Blackwell support)
- **CUDA**: CUDA Toolkit 12.8 (bundled with PyTorch nightly, or install separately)
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
   *Note: You need to accept the user agreement for `pyannote/speaker-diarization-3.1` on Hugging Face to get a valid token. (Required for diarization feature)*

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

### Access Control

By default, the bot is open to everyone. You can restrict access to specific users and chats:

1. **Admin Users** - Can use admin commands like `/admin`:
   ```ini
   TELEGRAM_ADMIN_USER_IDS=123456789,987654321
   ```

2. **Allowed Chats** - Can send audio files for transcription:
   ```ini
   TELEGRAM_ALLOWED_CHAT_IDS=123456789,-1001234567890,987654321
   ```
   - Use positive IDs for private chats (user IDs)
   - Use negative IDs for group chats (start with `-100`)
   - Leave empty to allow everyone

**How to get Chat/User IDs:**
- For private chats: User IDs can be obtained from bots like @userinfobot
- For groups: Add @RawDataBot to your group - it will show the chat ID

### Large Files Support (> 20 MB)

Telegram Bot API has a 20 MB file size limit. To support larger files:

1. **Get Telegram API credentials** from https://my.telegram.org/apps:
   - Click "API development tools"
   - Fill in the form (app title/short name can be anything)
   - You'll get `api_id` (number) and `api_hash` (string)

2. **Configure in `.env`**:
   ```ini
   TELEGRAM_API_ID=your_api_id_here
   TELEGRAM_API_HASH=your_api_hash_here
   TELEGRAM_USE_CLIENT_API=true
   ```

3. **Enable Privacy Mode OFF** for group chat support:
   - Message @BotFather in Telegram
   - Send `/setprivacy`
   - Select your bot
   - Choose **Disable** (turn privacy mode OFF)

   This allows the bot to see all messages in groups, not just commands or replies.

With Client API enabled, the bot can download files up to 2 GB.

## Troubleshooting

**"sm_120 is not compatible" or "CUDA capability sm_120 not supported"**
- Your RTX 5070 Ti requires PyTorch nightly with CUDA 12.8
- Verify installation: `python -c "import torch; print(torch.__version__, torch.version.cuda)"`
- Should show: `2.10.0.dev20251124+cu128 12.8`
- Reinstall if needed: `pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128`
- Run `python check_cuda.py` for detailed diagnostics

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
