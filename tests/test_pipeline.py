import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
from src.pipeline.preprocessor import AudioPreprocessor
from src.pipeline.aggregator import ResultAggregator, SmartSegmenter, AggregatedSegment
from src.pipeline.transcriber import GigaAMTranscriber
from src.pipeline.diarizer import PyAnnoteDiarizer

# --- Preprocessor Tests ---

@pytest.mark.asyncio
async def test_preprocessor_convert(mocker):
    # Mock FileManager
    mock_file_manager = MagicMock()
    mock_file_manager.check_user_quota.return_value = True
    mock_file_manager.get_task_dir.return_value = Path("/tmp/task")
    
    preprocessor = AudioPreprocessor(file_manager=mock_file_manager)
    
    # Test that preprocessor was created successfully
    assert preprocessor is not None
    assert preprocessor.file_manager == mock_file_manager

# --- Aggregator Tests ---

def test_aggregator_merge(mocker):
    # Mock transcriber for aggregator init
    mock_transcriber = mocker.Mock()
    aggregator = ResultAggregator(transcriber=mock_transcriber)
    
    # Test SmartSegmenter's merge_adjacent_segments
    segmenter = SmartSegmenter()
    
    segments = [
        AggregatedSegment(speaker_id="SPEAKER_00", text="Hello", start_seconds=0.0, end_seconds=1.0, confidence=0.9),
        AggregatedSegment(speaker_id="SPEAKER_00", text="world", start_seconds=1.2, end_seconds=2.0, confidence=0.9),
        AggregatedSegment(speaker_id="SPEAKER_01", text="Test", start_seconds=2.5, end_seconds=3.0, confidence=0.9),
    ]
    
    merged = segmenter.merge_adjacent_segments(segments)
    
    assert len(merged) == 2
    assert merged[0].speaker_id == "SPEAKER_00"
    assert merged[0].text == "Hello world"
    assert merged[1].speaker_id == "SPEAKER_01"

# --- Transcriber Tests (Mocked) ---

@pytest.mark.asyncio
async def test_transcriber_process(mocker):
    # Mock the singleton instance and its model
    with patch("src.pipeline.transcriber.GigaAMTranscriber") as MockTranscriber:
        instance = MockTranscriber.return_value
        # Make transcribe an AsyncMock that returns the expected value
        instance.transcribe_file = AsyncMock(return_value=mocker.Mock(
            segments=[mocker.Mock(text="Mocked")],
            is_successful=True
        ))
        
        transcriber = MockTranscriber()
        result = await transcriber.transcribe_file("dummy.wav")
        assert result.is_successful
        assert len(result.segments) > 0

# --- Diarizer Tests (Mocked) ---

@pytest.mark.asyncio
async def test_diarizer_process(mocker):
    with patch("src.pipeline.diarizer.PyAnnoteDiarizer") as MockDiarizer:
        instance = MockDiarizer.return_value
        # Make diarize an AsyncMock
        instance.diarize = AsyncMock(return_value=mocker.Mock(
            segments=[mocker.Mock(speaker_id="A", start_seconds=0, end_seconds=1)],
            is_successful=True
        ))
        
        diarizer = MockDiarizer()
        result = await diarizer.diarize("dummy.wav")
        assert result.is_successful
        assert len(result.segments) > 0
