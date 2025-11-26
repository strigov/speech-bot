"""
Full pipeline integration test for backend processing.

Tests the complete pipeline: Preprocessing -> Diarization -> Transcription -> Aggregation
Uses chunk_001.mp3 from tests directory and saves results to tests/test_result/.
"""

import asyncio
import os
import shutil
from pathlib import Path

import pytest
from dotenv import load_dotenv

from config.settings import Settings
from src.pipeline.aggregator import OutputGenerator, ResultAggregator
from src.pipeline.diarizer import get_diarizer
from src.pipeline.preprocessor import AudioPreprocessor, create_preprocessor
from src.pipeline.transcriber import get_transcriber
from src.utils.file_manager import FileManager

# Load .env file at module level
load_dotenv()


@pytest.mark.integration
@pytest.mark.slow
class TestFullPipeline:
    """Full pipeline integration tests."""

    @pytest.fixture
    def test_audio_file(self):
        """Path to test audio file."""
        audio_path = Path(__file__).parent / "chunk_001.mp3"
        if not audio_path.exists():
            pytest.skip(f"Test audio file not found: {audio_path}")
        return audio_path

    @pytest.fixture
    def results_dir(self):
        """Test results directory."""
        results_path = Path(__file__).parent / "test_result"
        results_path.mkdir(parents=True, exist_ok=True)
        return results_path

    @pytest.fixture
    def temp_dir(self):
        """Temporary directory for processing."""
        temp_path = Path(__file__).parent / "temp"
        temp_path.mkdir(parents=True, exist_ok=True)
        yield temp_path
        # Cleanup after test
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def settings(self):
        """Load settings from .env file."""
        return Settings()

    @pytest.fixture
    def hf_token(self, settings):
        """HuggingFace token from .env file."""
        if not settings.model.hf_token:
            pytest.skip("HF_TOKEN not set in .env file")
        return settings.model.hf_token

    @pytest.mark.asyncio
    async def test_full_pipeline_with_real_models(
        self, test_audio_file, results_dir, temp_dir, settings, hf_token
    ):
        """
        Test complete pipeline with real models.

        This test:
        1. Preprocesses chunk_001.mp3
        2. Runs speaker diarization
        3. Runs speech transcription
        4. Aggregates results
        5. Saves output to tests/test_result/

        NOTE: This test requires:
        - HF_TOKEN set in .env file
        - CUDA GPU (or CPU fallback)
        - Models will be downloaded on first run
        """
        print(f"\n{'='*60}")
        print("FULL PIPELINE TEST")
        print(f"{'='*60}")
        print(f"Audio file: {test_audio_file}")
        print(f"Results dir: {results_dir}")
        print(f"Temp dir: {temp_dir}")
        print(f"{'='*60}\n")

        # Step 1: Setup components
        print("Step 1: Initializing components...")
        file_manager = FileManager(temp_dir=temp_dir, max_per_user_gb=10.0)
        preprocessor = AudioPreprocessor(
            file_manager=file_manager,
            max_file_size_mb=500.0,
            max_duration_minutes=180.0,
        )

        # Get singletons for models (using settings from .env)
        diarizer = get_diarizer(
            model_id="pyannote/speaker-diarization-community-1",
            hf_token=hf_token,
            device=settings.model.diarization_device,  # From .env
        )

        transcriber = get_transcriber(
            model_id="ai-sage/GigaAM-v3",
            device=settings.model.device,  # From .env
        )

        print("[OK] Components initialized\n")

        # Step 2: Preprocessing
        print("Step 2: Preprocessing audio...")
        user_id = 999999  # Test user ID
        task_id = "test_pipeline"

        preprocess_result = await preprocessor.preprocess(
            input_path=test_audio_file,
            user_id=user_id,
            task_id=task_id,
            validate=True,
        )

        assert preprocess_result.is_valid, f"Preprocessing failed: {preprocess_result.error}"
        assert preprocess_result.processed_path is not None
        assert preprocess_result.processed_path.exists()

        audio_path = preprocess_result.processed_path
        duration = preprocess_result.duration_seconds

        print(f"[OK] Audio preprocessed: {duration:.1f}s")
        print(f"     Path: {audio_path}\n")

        # Step 3: Diarization
        print("Step 3: Running speaker diarization...")

        def diarization_progress(chunk: int, total: int, status: str):
            print(f"  Diarization progress: {chunk}/{total} - {status}")

        diarization_result = await diarizer.diarize(
            audio_path=audio_path,
            progress_callback=diarization_progress,
        )

        assert diarization_result.is_successful, f"Diarization failed: {diarization_result.error}"
        assert len(diarization_result.segments) > 0, "No speaker segments found"

        print(f"[OK] Diarization complete:")
        print(f"     Speakers: {diarization_result.num_speakers}")
        print(f"     Segments: {len(diarization_result.segments)}")
        print(f"     Processing time: {diarization_result.processing_time_seconds:.1f}s\n")

        # Step 4: Transcription + Aggregation
        print("Step 4: Running transcription and aggregation...")

        aggregator = ResultAggregator(transcriber=transcriber)

        def aggregation_progress(current: int, total: int, status: str):
            print(f"  Aggregation progress: {current}/{total} - {status}")

        aggregation_result = await aggregator.aggregate(
            audio_path=audio_path,
            diarization_result=diarization_result,
            progress_callback=aggregation_progress,
        )

        assert aggregation_result.is_successful, f"Aggregation failed: {aggregation_result.error}"
        assert len(aggregation_result.segments) > 0, "No aggregated segments"
        assert aggregation_result.total_words > 0, "No words transcribed"

        print(f"[OK] Transcription complete:")
        print(f"     Segments: {len(aggregation_result.segments)}")
        print(f"     Words: {aggregation_result.total_words}")
        print(f"     Speakers: {aggregation_result.num_speakers}")
        print(f"     Processing time: {aggregation_result.processing_time_seconds:.1f}s\n")

        # Step 5: Generate outputs
        print("Step 5: Generating output files...")

        output_generator = OutputGenerator(results_dir=results_dir)

        transcript_path = output_generator.generate_transcript(aggregation_result)
        plain_path = output_generator.generate_plain_text(aggregation_result)
        srt_path = output_generator.generate_srt(aggregation_result)
        json_path = output_generator.generate_json(aggregation_result)

        assert transcript_path.exists(), "Transcript file not created"
        assert plain_path.exists(), "Plain text file not created"
        assert srt_path.exists(), "SRT file not created"
        assert json_path.exists(), "JSON file not created"

        print(f"[OK] Output files generated:")
        print(f"     Transcript: {transcript_path}")
        print(f"     Plain text: {plain_path}")
        print(f"     SRT: {srt_path}")
        print(f"     JSON: {json_path}\n")

        # Verify file contents
        transcript_content = transcript_path.read_text(encoding="utf-8")
        assert len(transcript_content) > 0, "Transcript is empty"
        assert "SPEAKER_" in transcript_content, "No speaker labels in transcript"

        plain_content = plain_path.read_text(encoding="utf-8")
        assert len(plain_content) > 0, "Plain text is empty"

        print(f"{'='*60}")
        print("TEST PASSED - Full pipeline completed successfully!")
        print(f"{'='*60}\n")

        # Print sample transcript (first 500 chars)
        print("Sample transcript:")
        print("-" * 60)
        print(transcript_content[:500])
        if len(transcript_content) > 500:
            print("...")
        print("-" * 60)

    @pytest.mark.asyncio
    async def test_pipeline_without_models_validation_only(
        self, test_audio_file, temp_dir
    ):
        """
        Test preprocessing step without loading heavy models.

        This test validates:
        - File validation
        - Audio conversion
        - File management

        No HF_TOKEN required.
        """
        print(f"\n{'='*60}")
        print("PREPROCESSING VALIDATION TEST")
        print(f"{'='*60}\n")

        # Setup preprocessor
        file_manager = FileManager(temp_dir=temp_dir, max_per_user_gb=10.0)
        preprocessor = AudioPreprocessor(file_manager=file_manager)

        # Run preprocessing
        user_id = 999999
        task_id = "test_preprocess"

        result = await preprocessor.preprocess(
            input_path=test_audio_file,
            user_id=user_id,
            task_id=task_id,
            validate=True,
        )

        # Assertions
        assert result.is_valid, f"Validation failed: {result.error}"
        assert result.processed_path is not None
        assert result.processed_path.exists()
        assert result.audio_info is not None
        assert result.audio_info.sample_rate == 16000, "Should be converted to 16kHz"
        assert result.audio_info.channels == 1, "Should be converted to mono"
        assert result.duration_seconds > 0

        print(f"[OK] Preprocessing validation passed:")
        print(f"     Duration: {result.duration_seconds:.1f}s")
        print(f"     Sample rate: {result.audio_info.sample_rate}Hz")
        print(f"     Channels: {result.audio_info.channels}")
        print(f"     Codec: {result.audio_info.codec}")
        print(f"     Output: {result.processed_path}\n")

        print(f"{'='*60}")
        print("TEST PASSED - Preprocessing validation successful!")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    """
    Run this test directly with:

    # Make sure HF_TOKEN is set in .env file, then run:
    python -m pytest tests/test_full_pipeline.py -v -s

    Or run just preprocessing test:
    python -m pytest tests/test_full_pipeline.py::TestFullPipeline::test_pipeline_without_models_validation_only -v -s
    """
    pytest.main([__file__, "-v", "-s"])
