# Testing Guide

This document describes how to test the speech-bot pipeline.

## Quick Start

### Prerequisites

Make sure your `.env` file has `HF_TOKEN` set:

```bash
# .env file should contain:
HF_TOKEN=hf_xxxxxxxxxxxxx
```

### Option 1: Using the Test Runner Script (Recommended)

```bash
# Run only preprocessing test (no models, fast)
python run_pipeline_test.py --preprocess-only

# Run full pipeline test (reads HF_TOKEN from .env)
python run_pipeline_test.py
```

### Option 2: Using pytest Directly

```bash
# Run all pipeline tests
pytest tests/test_full_pipeline.py -v -s

# Run only preprocessing test
pytest tests/test_full_pipeline.py::TestFullPipeline::test_pipeline_without_models_validation_only -v -s

# Run full pipeline test (uses .env)
pytest tests/test_full_pipeline.py::TestFullPipeline::test_full_pipeline_with_real_models -v -s
```

## Test Types

### 1. Preprocessing Test (Fast, No GPU Required)

Tests audio validation and conversion without loading ML models.

**Requirements:**
- ✅ No HuggingFace token needed
- ✅ Works on CPU
- ✅ Fast (~5 seconds)

**What it tests:**
- File validation (format, size, integrity)
- Audio conversion to 16kHz mono WAV
- File management and cleanup

**Run with:**
```bash
python run_pipeline_test.py --preprocess-only
```

### 2. Full Pipeline Test (Slow, GPU Recommended)

Tests the complete transcription pipeline with real models.

**Requirements:**
- ⚠️ HuggingFace token in `.env` file (get from https://huggingface.co/settings/tokens)
- ⚠️ Accept PyAnnote terms: https://huggingface.co/pyannote/speaker-diarization-community-1
- ⚠️ GPU with CUDA recommended (CPU fallback available but slow)
- ⚠️ ~3GB disk space for models (downloaded on first run)
- ⚠️ ~2-5 minutes for 30s audio

**What it tests:**
1. Audio preprocessing
2. Speaker diarization (PyAnnote)
3. Speech transcription (GigaAM)
4. Result aggregation
5. Output generation (TXT, SRT, JSON)

**Run with:**
```bash
# Make sure .env has HF_TOKEN set
python run_pipeline_test.py
```

## Test Input & Output

### Input File
- **Location:** `tests/chunk_001.mp3`
- **Size:** ~28MB
- **Duration:** ~30 seconds
- **Content:** Audio sample with multiple speakers

### Output Files
Results are saved to `tests/test_result/`:

- `transcript.txt` - Formatted transcript with timestamps and speakers
- `transcript_plain.txt` - Plain text without formatting
- `transcript.srt` - SRT subtitle file
- `transcript.json` - JSON with full metadata

Example output structure:
```
tests/test_result/
├── transcript.txt          # [00:00:01 - 00:00:05] SPEAKER_00: Hello world
├── transcript_plain.txt    # Hello world How are you
├── transcript.srt          # 1\n00:00:01,000 --> 00:00:05,000\n[SPEAKER_00] Hello world
└── transcript.json         # {"metadata": {...}, "segments": [...]}
```

## Configuration

### Required in .env File

```bash
# .env file (required for full pipeline test)
HF_TOKEN=hf_xxxxxxxxxxxxx

# Optional: override device settings
MODEL_DEVICE=cuda  # or 'cpu'
DIARIZATION_DEVICE=cuda  # or 'cpu'
```

### Optional Environment Variables

```bash
# Force CPU mode (if GPU issues)
export CUDA_VISIBLE_DEVICES=""

# Increase logging verbosity
export PYTHONUNBUFFERED=1
```

## Troubleshooting

### Issue: "HF_TOKEN not found"

**Solution:**
1. Create HuggingFace account
2. Accept PyAnnote model terms: https://huggingface.co/pyannote/speaker-diarization-community-1
3. Create token: https://huggingface.co/settings/tokens
4. Add to `.env` file:
   ```bash
   HF_TOKEN=hf_xxxxxxxxxxxxx
   ```

### Issue: CUDA Out of Memory

**Solutions:**
- Use CPU mode: `export CUDA_VISIBLE_DEVICES=""`
- Reduce batch size in `config/models.yaml`:
  ```yaml
  pyannote:
    embedding_batch_size: 16  # reduce from 32
  ```
- Use shorter test audio

### Issue: Models downloading very slowly

**Solutions:**
- Use HuggingFace mirror: `export HF_ENDPOINT=https://hf-mirror.com`
- Pre-download models manually:
  ```python
  from pyannote.audio import Pipeline
  Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token="your_token")
  ```

### Issue: Test fails with "File not found"

**Solution:**
Ensure you're running from project root:
```bash
cd /path/to/speech-bot
python run_pipeline_test.py
```

## Running Other Tests

### Unit Tests
```bash
# All unit tests
pytest tests/ -v -m "not integration and not slow"

# Specific test file
pytest tests/test_pipeline.py -v
```

### Integration Tests
```bash
# All integration tests
pytest tests/ -v -m integration

# Skip slow tests
pytest tests/ -v -m "not slow"
```

### Test Coverage
```bash
# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Continuous Integration

For CI/CD pipelines, use the preprocessing test (fast, no external dependencies):

```yaml
# GitHub Actions example
- name: Run tests
  run: |
    pytest tests/test_full_pipeline.py::TestFullPipeline::test_pipeline_without_models_validation_only -v
```

For full pipeline testing in CI, you'll need to:
1. Create `.env` file with HF_TOKEN (from CI secrets)
2. Use GPU runners or expect slow execution on CPU
3. Cache downloaded models

## Performance Benchmarks

Typical execution times on different hardware:

| Test Type | Hardware | Time |
|-----------|----------|------|
| Preprocessing only | Any CPU | ~5s |
| Full pipeline (30s audio) | RTX 3090 | ~2min |
| Full pipeline (30s audio) | CPU (16 cores) | ~15min |
| Full pipeline (30s audio) | CPU (4 cores) | ~30min |

## Additional Resources

- [Tests Directory README](tests/test_result/README.md) - Detailed test output documentation
- [Pipeline Documentation](specification.md) - Pipeline architecture details
- [Configuration Guide](config/README.md) - Model and processing configuration
