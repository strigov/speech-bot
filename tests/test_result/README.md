# Test Results Directory

This directory contains the output files from the full pipeline integration test.

## Files Generated

When running the full pipeline test, the following files are created:

- **transcript.txt** - Formatted transcript with timestamps and speaker labels
- **transcript_plain.txt** - Plain text transcript without formatting
- **transcript.srt** - Subtitle file in SRT format
- **transcript.json** - JSON file with complete metadata and segments

## Running the Test

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure your `.env` file has HF_TOKEN set (required for PyAnnote models):
```bash
# In .env file:
HF_TOKEN=hf_xxxxxxxxxxxxx
```

### Running the Full Pipeline Test

Run the complete pipeline test with real models:

```bash
# Using the test runner script (recommended)
python run_pipeline_test.py

# Or directly with pytest
python -m pytest tests/test_full_pipeline.py::TestFullPipeline::test_full_pipeline_with_real_models -v -s
```

This will:
1. Load configuration from `.env` file
2. Preprocess `chunk_001.mp3`
3. Run speaker diarization (PyAnnote)
4. Run speech transcription (GigaAM)
5. Aggregate results
6. Save outputs to this directory

**Note:** First run will download models (~2-3GB). Requires GPU with CUDA or will use CPU (slower).

### Running Preprocessing Test Only

Test just the preprocessing step (no models required, no token needed):

```bash
# Using the test runner script
python run_pipeline_test.py --preprocess-only

# Or directly with pytest
python -m pytest tests/test_full_pipeline.py::TestFullPipeline::test_pipeline_without_models_validation_only -v -s
```

### Running All Tests

Run all integration tests:

```bash
python -m pytest tests/test_full_pipeline.py -v -s
```

## Expected Output

The test will print detailed progress information:

```
==============================================================
FULL PIPELINE TEST
==============================================================
Audio file: tests/chunk_001.mp3
Results dir: tests/test_result
==============================================================

Step 1: Initializing components...
✓ Components initialized

Step 2: Preprocessing audio...
✓ Audio preprocessed: 30.5s
  Path: tests/temp/999999/test_pipeline/audio.wav

Step 3: Running speaker diarization...
  Diarization progress: 1/1 - Processing...
✓ Diarization complete:
  Speakers: 2
  Segments: 15
  Processing time: 12.3s

Step 4: Running transcription and aggregation...
  Aggregation progress: 1/15 - Transcribing...
  ...
✓ Transcription complete:
  Segments: 15
  Words: 245
  Speakers: 2
  Processing time: 45.2s

Step 5: Generating output files...
✓ Output files generated:
  Transcript: tests/test_result/transcript.txt
  Plain text: tests/test_result/transcript_plain.txt
  SRT: tests/test_result/transcript.srt
  JSON: tests/test_result/transcript.json

==============================================================
TEST PASSED - Full pipeline completed successfully!
==============================================================
```

## Troubleshooting

### CUDA Out of Memory

If you get OOM errors:
- Reduce batch size in config/models.yaml
- Use CPU instead: modify test to use `device="cpu"`
- Use shorter audio file

### HF_TOKEN Error

Make sure you have:
1. Created a HuggingFace account
2. Accepted PyAnnote model terms at: https://huggingface.co/pyannote/speaker-diarization-community-1
3. Created an access token at: https://huggingface.co/settings/tokens
4. Added it to `.env` file: `HF_TOKEN=hf_xxxxxxxxxxxxx`

### Models Not Downloading

Check:
- Internet connection
- HuggingFace token permissions
- Disk space (~3GB needed)
