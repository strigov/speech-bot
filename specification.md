# Technical Specification: GigaAM & PyAnnote Telegram Transcriber

## 1. Executive Summary

Develop a Windows-native Telegram Bot for audio transcription using GigaAM v3 ASR and PyAnnote speaker diarization. The bot will serve up to 250 users in an educational/experimental chat, processing audio files up to several hours in length.

**Core Requirements:**
- Native Windows 10/11 execution (no Docker)
- NVIDIA RTX 5070 Ti (16GB VRAM) utilization
- Single-file launch via `start.bat`
- Support for long-form audio (1+ hours)
- Speaker diarization with timestamps

## 2. Technical Stack

### Runtime Environment
- **OS:** Windows 10/11 x64
- **Python:** 3.10.x (strict requirement for library compatibility)
- **GPU:** NVIDIA RTX 5070 Ti with CUDA 12.1+
- **Memory Management:** 14GB VRAM allocation limit (2GB system reserve)

### Core Dependencies
- **Telegram Framework:** `aiogram>=3.10.0` (async)
- **ASR Model:** GigaAM v3 via Hugging Face Transformers
  - Model: `ai-sage/GigaAM-v3`
- **Diarization:** `pyannote.audio==4.0.0`
  - Model: `pyannote/speaker-diarization-4`
  - Requires: Hugging Face access token
- **Audio Processing:**
  - `ffmpeg-python` (wrapper)
  - `pydub` (segmentation)
  - `torchaudio` (tensor operations)
- **ML Framework:** PyTorch 2.8.0+cu128

### System Requirements
- FFmpeg binary installed and in PATH
- Minimum 32GB RAM recommended
- SSD storage for temp files (min 50GB free)

## 3. Architecture Design

### 3.1 Processing Pipeline

```mermaid
graph LR
    A[Telegram Input] --> B[Download Manager]
    B --> C[Queue System]
    C --> D[Audio Preprocessor]
    D --> E[Diarization Engine]
    E --> F[Segmentation Logic]
    F --> G[ASR Processor]
    G --> H[Result Aggregator]
    H --> I[File Generator]
    I --> J[Telegram Output]
```

### 3.2 Queue Management Strategy

- **Single asyncio.Queue** with configurable size limit (default: 50 tasks)
- **Priority system:** FIFO with timeout detection
- **Rejection policy:** Notify user when queue full
- **Timeout:** 30 minutes max per file

### 3.3 Long Audio Handling

For files >30 minutes:
1. **Chunked Diarization:** Process in 10-minute windows with 30-second overlap
2. **Speaker Mapping:** Maintain speaker consistency across chunks
3. **Progress Updates:** Send status every 10% completion
4. **Checkpointing:** Save intermediate results every 10 minutes
5. **Memory Management:** Clear GPU cache after each chunk

## 4. Detailed Implementation Requirements

### 4.1 Model Loading Strategy

- **Singleton Pattern:** Load models once at startup
- **Lazy Loading:** Initialize only when first task arrives
- **Memory Monitoring:** Check VRAM before loading
- **Fallback:** CPU processing if GPU fails (with user notification)

### 4.2 Audio Processing Pipeline

#### Stage 1: Preprocessing
- Convert to WAV 16kHz mono
- Validate audio integrity
- Reject if corrupted or >2GB
- Store in `./temp/audio/{user_id}/{task_id}/`

#### Stage 2: Diarization
```
Input: Full audio file
Process:
  - If duration <= 30 min: Process whole
  - If duration > 30 min:
    - Split into 10-min chunks (30s overlap)
    - Run diarization per chunk
    - Merge speaker labels using embedding similarity
Output: Timeline [(start, end, speaker_id), ...]
```

#### Stage 3: Smart Segmentation
- Maximum segment: 30 seconds
- Minimum segment: 0.5 seconds
- Padding: 0.1s at boundaries
- Overlap handling: 0.2s for splits
- Silence detection for natural breaks

#### Stage 4: Transcription
- Batch processing: Up to 32 segments simultaneously
- Error recovery: Retry failed segments with reduced batch size
- Confidence scoring: Include optional confidence in output

#### Stage 5: Output Generation
```
Format: [HH:MM:SS - HH:MM:SS] SPEAKER_XX: Transcribed text
Example:
[00:00:05 - 00:00:12] SPEAKER_00: Hello, this is the beginning.
[00:00:13 - 00:00:18] SPEAKER_01: I understand your point.
[01:45:30 - 01:45:45] SPEAKER_00: This works for long recordings.
```

### 4.3 Error Handling & Recovery

#### Critical Errors (Stop Processing)
- Model loading failure
- GPU initialization error
- Invalid API tokens

#### Recoverable Errors (Retry Logic)
- OOM errors → Clear cache, reduce batch size
- Network timeouts → Exponential backoff
- Audio conversion failure → Try alternative method
- Partial diarization failure → Process in smaller chunks

#### User Communication
- Queue position updates
- Processing started notification
- Progress updates (every 10% for long files)
- Error messages with actionable information
- Completion with file delivery

### 4.4 Resource Management

#### GPU Memory Control
- Pre-allocation check before processing
- Dynamic batch size adjustment
- Periodic `torch.cuda.empty_cache()`
- Maximum allocation: 14GB (configurable)

#### Disk Management
- Automatic cleanup after task completion
- Maximum temp storage per user: 5GB
- Orphan file cleanup on startup
- Compression for archived results

#### Rate Limiting
- Per-user: Max 3 concurrent files
- Per-user: Max 10 files per hour
- Global: Queue size limit of 50
- File size: Max 500MB (configurable)

## 5. Project Structure

```
project_root/
├── config/
│   ├── settings.py          # Configuration management
│   ├── models.yaml          # Model paths and parameters
│   └── limits.yaml          # Rate limits and thresholds
├── src/
│   ├── bot/
│   │   ├── handlers.py      # Telegram message handlers
│   │   ├── filters.py       # Custom filters (file size, type)
│   │   └── keyboards.py     # UI elements
│   ├── pipeline/
│   │   ├── preprocessor.py  # Audio conversion and validation
│   │   ├── diarizer.py      # PyAnnote integration
│   │   ├── transcriber.py   # GigaAM integration
│   │   └── aggregator.py    # Result formatting
│   ├── utils/
│   │   ├── audio.py         # Audio utilities
│   │   ├── gpu_monitor.py   # VRAM tracking
│   │   └── file_manager.py  # Temp file handling
│   └── worker.py            # Main processing loop
├── temp/                    # Temporary storage
│   ├── audio/              # Input files
│   ├── results/            # Output files
│   └── checkpoints/        # Intermediate saves
├── logs/                   # Application logs
├── tests/                  # Unit and integration tests
├── .env                    # Environment variables
├── main.py                 # Application entry point
├── requirements.txt        # Pinned dependencies
├── setup_cuda.bat         # CUDA installation helper
├── start.bat              # Production launcher
└── start_dev.bat          # Development launcher with hot reload
```

## 6. Configuration Schema

### Environment Variables (.env)
```ini
# Telegram Configuration
TELEGRAM_BOT_TOKEN=xxx
TELEGRAM_API_URL=https://api.telegram.org
USE_LOCAL_BOT_API=false
LOCAL_BOT_API_URL=http://localhost:8081

# Model Configuration
HF_TOKEN=xxx
MODEL_DEVICE=cuda
DIARIZATION_DEVICE=cuda
ASR_BATCH_SIZE=32
DIARIZATION_BATCH_SIZE=32

# Processing Limits
MAX_AUDIO_DURATION_MINUTES=180
MAX_FILE_SIZE_MB=500
MAX_QUEUE_SIZE=50
MAX_USER_CONCURRENT=3
PROCESSING_TIMEOUT_MINUTES=30

# Memory Management
MAX_VRAM_GB=14.0
CHUNK_DURATION_MINUTES=10
SEGMENT_MAX_SECONDS=30

# Paths
TEMP_DIR=./temp
LOG_DIR=./logs
CHECKPOINT_DIR=./temp/checkpoints

# Development
DEBUG=false
HOT_RELOAD=false
LOG_LEVEL=INFO
```

### Model Configuration (models.yaml)
```yaml
gigaam:
  model_id: "ai-sage/GigaAM-v3"
  processor_id: "ai-sage/GigaAM-v3"
  sample_rate: 16000
  chunk_length_s: 30
  stride_length_s: 5
  
pyannote:
  model_id: "pyannote/speaker-diarization-3.1"
  min_speakers: 1
  max_speakers: 20
  embedding_batch_size: 32
  segmentation:
    min_duration: 0.5
    threshold: 0.5
```

## 7. Administrative Features

### Bot Commands
- `/start` - Welcome message with capabilities
- `/help` - Detailed usage instructions
- `/status` - Current queue position (user-specific)
- `/cancel` - Cancel current task
- `/admin` - Admin panel (restricted)
  - `/admin gpu` - GPU memory status
  - `/admin queue` - Queue statistics
  - `/admin clear` - Clear stuck tasks
  - `/admin restart` - Restart worker (keeps bot running)

### Monitoring Endpoints
- Queue length and processing rate
- Average processing time per minute of audio
- GPU utilization percentage
- Disk usage for temp files
- Error rate by type

## 8. Testing Requirements

### Test Scenarios
1. **Short audio** (<1 minute) - Single speaker
2. **Medium audio** (10-15 minutes) - Multiple speakers
3. **Long audio** (60+ minutes) - Complex conversation
4. **Edge cases:**
   - Silent audio
   - Noisy recording
   - Music with speech
   - Multiple languages
   - Corrupt file handling
5. **Load testing:**
   - 10 simultaneous requests
   - Queue overflow behavior
   - Memory pressure scenarios

### Performance Targets
- Short audio (<5 min): Process within 1 minute
- Medium audio (5-30 min): Process within 5 minutes
- Long audio (60 min): Process within 15 minutes
- Queue wait notification: Within 5 seconds
- Memory usage: <14GB VRAM consistently

## 9. Security Considerations

### Input Validation
- File type verification (MIME type + magic bytes)
- File size limits enforcement
- Filename sanitization
- Path traversal prevention

### User Isolation
- Separate temp directories per user
- No cross-user file access
- Cleanup on task completion/failure

### Rate Limiting
- Token bucket algorithm per user
- Global rate limiting for API calls
- Exponential backoff for repeat offenders

## 10. Development Guidelines

### For AI Agent Implementation

1. **Model Integration Priority:**
   - First: Get GigaAM working with test file
   - Second: Add PyAnnote diarization
   - Third: Integrate Telegram bot
   - Fourth: Add queue system
   - Fifth: Implement long audio handling

2. **Critical Implementation Notes:**
   - Do NOT reload models for each request
   - Always use try-except for GPU operations
   - Implement progress callbacks for long operations
   - Test with real 1-hour audio files early
   - Use logging extensively (not print statements)

3. **Windows-Specific Considerations:**
   - Use `pathlib.Path` for all file operations
   - Handle Windows path length limitations
   - Test with Windows Defender active
   - Ensure FFmpeg PATH is correct

4. **Installation Instructions Must Include:**
   ```
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install transformers accelerate
   pip install pyannote.audio
   ```

## 11. Deliverables

1. **Core Application** - Fully functional bot
2. **Setup Script** - Automated environment setup
3. **Documentation:**
   - User guide (Russian)
   - Admin guide
   - Troubleshooting guide
4. **Test Suite** - Basic automated tests
5. **Launcher Scripts:**
   - `start.bat` - Production
   - `start_dev.bat` - Development
   - `setup_environment.bat` - First-time setup
