import pytest
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path
from src.worker import Worker, create_task, TaskStatus
from src.pipeline.preprocessor import PreprocessingResult, AudioPreprocessor
from src.pipeline.diarizer import DiarizationResult
from src.pipeline.aggregator import AggregationResult
from src.utils.audio import AudioInfo

@pytest.mark.asyncio
async def test_end_to_end_worker_process_task(mocker):
    # Mock dependencies
    mock_queue = MagicMock()
    
    mock_preprocessor = AsyncMock()
    # Create PreprocessingResult with correct parameters
    mock_audio_info = AudioInfo(
        path=Path("processed.wav"),
        duration_seconds=60.0,
        sample_rate=16000,
        channels=1,
        codec="pcm_s16le",
        format_name="wav",
        bit_rate=None,
        file_size_bytes=1000000,
        is_valid=True
    )
    
    preprocessing_result = PreprocessingResult(
        original_path=Path("input.mp3"),
        processed_path=Path("processed.wav"),
        audio_info=mock_audio_info,
        task_id="test123",
        user_id=123,
        is_valid=True
    )
    mock_preprocessor.preprocess.return_value = preprocessing_result
    mock_preprocessor.cleanup_task = MagicMock()
    
    mock_transcriber = MagicMock()
    
    mock_diarizer = AsyncMock()
    mock_diarizer.diarize.return_value = DiarizationResult(
        is_successful=True,
        num_speakers=2,
        segments=[MagicMock()],
        duration_seconds=60.0,
        processing_time_seconds=5.0
    )
    
    mock_file_manager = MagicMock()
    mock_file_manager.get_results_dir.return_value = Path("results")
    
    mock_gpu_monitor = MagicMock()
    mock_gpu_monitor.clear_cache = MagicMock()
    
    # Mock Aggregator (it's instantiated inside _process_task, so we need to patch the class)
    mock_aggregator_cls = mocker.patch("src.worker.ResultAggregator")
    mock_aggregator_instance = mock_aggregator_cls.return_value
    mock_aggregator_instance.aggregate = AsyncMock(return_value=AggregationResult(
        is_successful=True,
        segments=[MagicMock()],
        total_words=100,
        num_speakers=2
    ))
    
    # Mock OutputGenerator
    mock_output_gen_cls = mocker.patch("src.worker.OutputGenerator")
    mock_output_gen = mock_output_gen_cls.return_value
    mock_output_gen.generate_transcript.return_value = Path("out.txt")
    mock_output_gen.generate_plain_text.return_value = Path("out.txt")
    mock_output_gen.generate_srt.return_value = Path("out.srt")
    mock_output_gen.generate_json.return_value = Path("out.json")

    # Initialize worker
    worker = Worker(
        queue=mock_queue,
        preprocessor=mock_preprocessor,
        transcriber=mock_transcriber,
        diarizer=mock_diarizer,
        file_manager=mock_file_manager,
        gpu_monitor=mock_gpu_monitor,
        checkpoint_dir=Path("checkpoints")
    )
    
    # Create task
    task = create_task(
        user_id=123,
        input_path=Path("input.mp3"),
        original_filename="input.mp3"
    )
    
    # Run _process_task
    await worker._process_task(task)
    
    # Assertions
    assert task.progress.status == TaskStatus.COMPLETED
    assert task.result is not None
    assert task.output_files["transcript"] == Path("out.txt")
    
    # Verify calls
    mock_preprocessor.preprocess.assert_called_once()
    mock_diarizer.diarize.assert_called_once()
    mock_aggregator_instance.aggregate.assert_called_once()
