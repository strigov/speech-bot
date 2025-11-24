# Development Plan: GigaAM & PyAnnote Telegram Transcriber

## Overview
Windows-native Telegram Bot for audio transcription using GigaAM v3 ASR and PyAnnote speaker diarization.

---

## Step 1: Project Structure & Configuration
**Goal:** Set up project foundation with proper structure, configuration files, and environment management.

**Tasks:**
- Create directory structure (`config/`, `src/`, `temp/`, `logs/`, `tests/`)
- Implement `config/settings.py` with Pydantic-based configuration
- Create `config/models.yaml` and `config/limits.yaml`
- Set up `.env.example` template
- Create `requirements.txt` with pinned dependencies
- Create basic `main.py` entry point

---

## Step 2: Core Utilities
**Goal:** Build foundational utility modules.

**Tasks:**
- Implement `src/utils/file_manager.py` - temp file handling, cleanup
- Implement `src/utils/gpu_monitor.py` - VRAM tracking and monitoring
- Implement `src/utils/audio.py` - audio format utilities
- Add logging configuration

---

## Step 3: GigaAM ASR Integration
**Goal:** Get GigaAM working with test audio files.

**Tasks:**
- Implement `src/pipeline/transcriber.py` - GigaAM model loading (singleton)
- Add batch processing support
- Add error recovery with batch size reduction
- Test with sample audio files
- Implement GPU memory management

---

## Step 4: PyAnnote Diarization Integration
**Goal:** Add speaker diarization capability.

**Tasks:**
- Implement `src/pipeline/diarizer.py` - PyAnnote model loading
- Add chunked processing for long audio (10-min windows, 30s overlap)
- Implement speaker embedding similarity for cross-chunk consistency
- Add progress callbacks

---

## Step 5: Audio Preprocessing Pipeline
**Goal:** Build robust audio conversion and validation.

**Tasks:**
- Implement `src/pipeline/preprocessor.py`
- FFmpeg integration for format conversion (16kHz mono WAV)
- Audio integrity validation
- File size and duration checks
- Temp file management per user/task

---

## Step 6: Smart Segmentation & Aggregation
**Goal:** Combine diarization with transcription.

**Tasks:**
- Implement `src/pipeline/aggregator.py`
- Smart segmentation logic (0.5s-30s segments)
- Silence detection for natural breaks
- Result formatting with timestamps and speaker labels
- Output file generation

---

## Step 7: Queue System & Worker
**Goal:** Build async processing infrastructure.

**Tasks:**
- Implement `src/worker.py` - main processing loop
- asyncio.Queue with 50-task limit
- FIFO priority with timeout detection
- Checkpointing for long audio
- Progress update callbacks

---

## Step 8: Telegram Bot Integration
**Goal:** Create the Telegram interface.

**Tasks:**
- Implement `src/bot/handlers.py` - message handlers
- Implement `src/bot/filters.py` - file type/size validation
- Implement `src/bot/keyboards.py` - UI elements
- Bot commands: `/start`, `/help`, `/status`, `/cancel`, `/admin`
- Progress notifications and file delivery

---

## Step 9: Rate Limiting & Security
**Goal:** Add protection and resource limits.

**Tasks:**
- Per-user rate limiting (3 concurrent, 10/hour)
- Global queue size limiting
- File validation (MIME type, magic bytes)
- Path traversal prevention
- User isolation in temp directories

---

## Step 10: Launcher Scripts & Documentation
**Goal:** Create deployment-ready package.

**Tasks:**
- Create `start.bat` - production launcher
- Create `start_dev.bat` - development launcher
- Create `setup_environment.bat` - first-time setup
- Create `setup_cuda.bat` - CUDA installation helper
- Write user documentation

---

## Step 11: Testing & Optimization
**Goal:** Ensure reliability and performance.

**Tasks:**
- Unit tests for core components
- Integration tests with real audio
- Load testing (10 simultaneous requests)
- Memory pressure testing
- Performance optimization to meet targets

---

## Performance Targets
| Audio Length | Target Time |
|--------------|-------------|
| < 5 min      | < 1 min     |
| 5-30 min     | < 5 min     |
| 60 min       | < 15 min    |

---

## Tech Stack Summary
- Python 3.10.x
- aiogram >= 3.10.0
- PyTorch 2.0+ (CUDA 12.1)
- GigaAM v3 (`ai-sage/GigaAM-v3`)
- PyAnnote 3.3.1 (`pyannote/speaker-diarization-3.1`)
- FFmpeg for audio processing
