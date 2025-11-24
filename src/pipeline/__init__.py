"""Audio processing pipeline module."""

from src.pipeline.aggregator import (
    AggregatedSegment,
    AggregationResult,
    OutputGenerator,
    ResultAggregator,
    SmartSegmenter,
)
from src.pipeline.diarizer import (
    DiarizationResult,
    PyAnnoteDiarizer,
    SpeakerSegment,
    get_diarizer,
)
from src.pipeline.preprocessor import (
    AudioPreprocessor,
    PreprocessingResult,
    create_preprocessor,
)
from src.pipeline.transcriber import (
    GigaAMTranscriber,
    TranscriptionResult,
    TranscriptionSegment,
    get_transcriber,
)

__all__ = [
    # Aggregator
    "AggregatedSegment",
    "AggregationResult",
    "OutputGenerator",
    "ResultAggregator",
    "SmartSegmenter",
    # Diarizer
    "DiarizationResult",
    "PyAnnoteDiarizer",
    "SpeakerSegment",
    "get_diarizer",
    # Preprocessor
    "AudioPreprocessor",
    "PreprocessingResult",
    "create_preprocessor",
    # Transcriber
    "GigaAMTranscriber",
    "TranscriptionResult",
    "TranscriptionSegment",
    "get_transcriber",
]
