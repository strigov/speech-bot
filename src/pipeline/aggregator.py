"""Result aggregation and output formatting for transcription pipeline."""

import asyncio
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import structlog

from src.pipeline.diarizer import DiarizationResult, SpeakerSegment
from src.pipeline.transcriber import (
    GigaAMTranscriber,
    TranscriptionResult,
    TranscriptionSegment,
)

logger = structlog.get_logger(__name__)


@dataclass
class AggregatedSegment:
    """A segment combining speaker and transcription info."""

    speaker_id: str
    text: str
    start_seconds: float
    end_seconds: float
    confidence: Optional[float] = None

    @property
    def duration(self) -> float:
        """Get segment duration."""
        return self.end_seconds - self.start_seconds

    def format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def to_formatted_line(self) -> str:
        """Format segment as output line."""
        start_ts = self.format_timestamp(self.start_seconds)
        end_ts = self.format_timestamp(self.end_seconds)
        return f"[{start_ts} - {end_ts}] {self.speaker_id}: {self.text}"


@dataclass
class AggregationResult:
    """Final aggregated transcription result."""

    segments: List[AggregatedSegment] = field(default_factory=list)
    num_speakers: int = 0
    total_duration_seconds: float = 0.0
    total_words: int = 0
    processing_time_seconds: float = 0.0
    diarization_time_seconds: float = 0.0
    transcription_time_seconds: float = 0.0
    error: Optional[str] = None
    is_successful: bool = True

    def get_formatted_text(self) -> str:
        """Get full formatted transcript."""
        return "\n".join(seg.to_formatted_line() for seg in self.segments)

    def get_plain_text(self) -> str:
        """Get plain text without timestamps."""
        return " ".join(seg.text for seg in self.segments if seg.text)

    def get_speaker_stats(self) -> dict:
        """Get statistics per speaker."""
        stats = {}
        for seg in self.segments:
            if seg.speaker_id not in stats:
                stats[seg.speaker_id] = {
                    "duration_seconds": 0.0,
                    "word_count": 0,
                    "segment_count": 0,
                }
            stats[seg.speaker_id]["duration_seconds"] += seg.duration
            stats[seg.speaker_id]["word_count"] += len(seg.text.split())
            stats[seg.speaker_id]["segment_count"] += 1
        return stats


class SmartSegmenter:
    """
    Smart segmentation logic for combining diarization with ASR.

    Handles:
    - Segment splitting based on duration limits
    - Segment merging for short segments
    - Natural break detection
    """

    def __init__(
        self,
        min_segment_seconds: float = 0.5,
        max_segment_seconds: float = 30.0,
        padding_seconds: float = 0.1,
        merge_threshold_seconds: float = 0.3,
    ):
        """
        Initialize segmenter.

        Args:
            min_segment_seconds: Minimum segment duration
            max_segment_seconds: Maximum segment duration
            padding_seconds: Padding at segment boundaries
            merge_threshold_seconds: Gap threshold for merging segments
        """
        self.min_segment_seconds = min_segment_seconds
        self.max_segment_seconds = max_segment_seconds
        self.padding_seconds = padding_seconds
        self.merge_threshold_seconds = merge_threshold_seconds

    def prepare_segments_for_asr(
        self,
        diarization_segments: List[SpeakerSegment],
    ) -> List[Tuple[float, float, str]]:
        """
        Prepare diarization segments for ASR processing.

        - Filters out segments that are too short
        - Splits segments that are too long
        - Adds padding at boundaries

        Args:
            diarization_segments: Raw diarization segments

        Returns:
            List of (start, end, speaker_id) tuples ready for ASR
        """
        prepared = []

        for seg in diarization_segments:
            duration = seg.end_seconds - seg.start_seconds

            # Skip very short segments
            if duration < self.min_segment_seconds:
                continue

            # Split long segments
            if duration > self.max_segment_seconds:
                current_start = seg.start_seconds
                while current_start < seg.end_seconds:
                    chunk_end = min(
                        current_start + self.max_segment_seconds,
                        seg.end_seconds
                    )

                    # Don't create tiny final chunks
                    remaining = seg.end_seconds - chunk_end
                    if 0 < remaining < self.min_segment_seconds:
                        chunk_end = seg.end_seconds

                    # Add with padding
                    padded_start = max(0, current_start - self.padding_seconds)
                    padded_end = chunk_end + self.padding_seconds

                    prepared.append((padded_start, padded_end, seg.speaker_id))
                    current_start = chunk_end
            else:
                # Add with padding
                padded_start = max(0, seg.start_seconds - self.padding_seconds)
                padded_end = seg.end_seconds + self.padding_seconds
                prepared.append((padded_start, padded_end, seg.speaker_id))

        return prepared

    def merge_adjacent_segments(
        self,
        segments: List[AggregatedSegment],
    ) -> List[AggregatedSegment]:
        """
        Merge adjacent segments from the same speaker.

        Args:
            segments: List of aggregated segments

        Returns:
            Merged segments
        """
        if not segments:
            return []

        merged = []
        current = segments[0]

        for next_seg in segments[1:]:
            # Check if can merge
            gap = next_seg.start_seconds - current.end_seconds
            same_speaker = current.speaker_id == next_seg.speaker_id

            if same_speaker and gap <= self.merge_threshold_seconds:
                # Merge segments
                current = AggregatedSegment(
                    speaker_id=current.speaker_id,
                    text=f"{current.text} {next_seg.text}".strip(),
                    start_seconds=current.start_seconds,
                    end_seconds=next_seg.end_seconds,
                    confidence=(
                        (current.confidence + next_seg.confidence) / 2
                        if current.confidence and next_seg.confidence
                        else None
                    ),
                )
            else:
                merged.append(current)
                current = next_seg

        merged.append(current)
        return merged


class ResultAggregator:
    """
    Aggregates diarization and transcription results.

    Orchestrates the full pipeline from diarized segments to final output.
    """

    def __init__(
        self,
        transcriber: GigaAMTranscriber,
        min_segment_seconds: float = 0.5,
        max_segment_seconds: float = 30.0,
        merge_adjacent: bool = True,
    ):
        """
        Initialize aggregator.

        Args:
            transcriber: GigaAM transcriber instance
            min_segment_seconds: Minimum segment duration
            max_segment_seconds: Maximum segment duration
            merge_adjacent: Whether to merge adjacent same-speaker segments
        """
        self.transcriber = transcriber
        self.segmenter = SmartSegmenter(
            min_segment_seconds=min_segment_seconds,
            max_segment_seconds=max_segment_seconds,
        )
        self.merge_adjacent = merge_adjacent

    async def aggregate(
        self,
        audio_path: Path,
        diarization_result: DiarizationResult,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> AggregationResult:
        """
        Aggregate diarization with transcription.

        Args:
            audio_path: Path to preprocessed audio file
            diarization_result: Result from diarization
            progress_callback: Optional callback (current, total, status)

        Returns:
            AggregationResult with full transcript
        """
        import time
        start_time = time.time()

        if not diarization_result.is_successful:
            return AggregationResult(
                error=f"Diarization failed: {diarization_result.error}",
                is_successful=False,
            )

        if not diarization_result.segments:
            return AggregationResult(
                error="No speaker segments found in audio",
                is_successful=False,
            )

        # Prepare segments for ASR
        asr_segments = self.segmenter.prepare_segments_for_asr(
            diarization_result.segments
        )

        if not asr_segments:
            return AggregationResult(
                error="No valid segments for transcription",
                is_successful=False,
            )

        logger.info(
            "aggregation_started",
            diarization_segments=len(diarization_result.segments),
            asr_segments=len(asr_segments),
        )

        # Create progress wrapper
        def transcription_progress(current: int, total: int):
            if progress_callback:
                progress_callback(current, total, "Transcribing...")

        # Run transcription
        transcription_start = time.time()
        transcription_result = await self.transcriber.transcribe_segments(
            audio_path,
            asr_segments,
            progress_callback=transcription_progress,
        )
        transcription_time = time.time() - transcription_start

        if not transcription_result.is_successful:
            return AggregationResult(
                error=f"Transcription failed: {transcription_result.error}",
                is_successful=False,
                diarization_time_seconds=diarization_result.processing_time_seconds,
            )

        # Build aggregated segments
        aggregated_segments = []
        for trans_seg in transcription_result.segments:
            if not trans_seg.text.strip():
                continue

            aggregated_segments.append(AggregatedSegment(
                speaker_id=trans_seg.speaker_id or "SPEAKER_00",
                text=trans_seg.text.strip(),
                start_seconds=trans_seg.start_seconds,
                end_seconds=trans_seg.end_seconds,
                confidence=trans_seg.confidence,
            ))

        # Sort by start time
        aggregated_segments.sort(key=lambda s: s.start_seconds)

        # Merge adjacent same-speaker segments
        if self.merge_adjacent and aggregated_segments:
            aggregated_segments = self.segmenter.merge_adjacent_segments(
                aggregated_segments
            )

        # Calculate stats
        total_words = sum(
            len(seg.text.split()) for seg in aggregated_segments
        )

        processing_time = time.time() - start_time

        logger.info(
            "aggregation_complete",
            segments=len(aggregated_segments),
            speakers=diarization_result.num_speakers,
            words=total_words,
            processing_time=f"{processing_time:.1f}s",
        )

        return AggregationResult(
            segments=aggregated_segments,
            num_speakers=diarization_result.num_speakers,
            total_duration_seconds=diarization_result.duration_seconds,
            total_words=total_words,
            processing_time_seconds=processing_time,
            diarization_time_seconds=diarization_result.processing_time_seconds,
            transcription_time_seconds=transcription_time,
            is_successful=True,
        )


class OutputGenerator:
    """Generates output files from aggregation results."""

    def __init__(self, results_dir: Path):
        """
        Initialize generator.

        Args:
            results_dir: Directory to write output files
        """
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def generate_transcript(
        self,
        result: AggregationResult,
        filename: str = "transcript.txt",
        include_header: bool = True,
    ) -> Path:
        """
        Generate formatted transcript file.

        Args:
            result: Aggregation result
            filename: Output filename
            include_header: Whether to include stats header

        Returns:
            Path to generated file
        """
        output_path = self.results_dir / filename

        lines = []

        if include_header:
            lines.append("=" * 60)
            lines.append("TRANSCRIPT")
            lines.append("=" * 60)
            lines.append(f"Duration: {self._format_duration(result.total_duration_seconds)}")
            lines.append(f"Speakers: {result.num_speakers}")
            lines.append(f"Words: {result.total_words}")
            lines.append("=" * 60)
            lines.append("")

        # Add transcript
        lines.append(result.get_formatted_text())

        # Add footer with stats
        if include_header:
            lines.append("")
            lines.append("=" * 60)
            lines.append("SPEAKER STATISTICS")
            lines.append("=" * 60)
            for speaker_id, stats in sorted(result.get_speaker_stats().items()):
                duration = self._format_duration(stats["duration_seconds"])
                lines.append(
                    f"{speaker_id}: {stats['word_count']} words, "
                    f"{duration}, {stats['segment_count']} segments"
                )

        output_path.write_text("\n".join(lines), encoding="utf-8")

        logger.info("transcript_generated", path=str(output_path))
        return output_path

    def generate_plain_text(
        self,
        result: AggregationResult,
        filename: str = "transcript_plain.txt",
    ) -> Path:
        """
        Generate plain text transcript without formatting.

        Args:
            result: Aggregation result
            filename: Output filename

        Returns:
            Path to generated file
        """
        output_path = self.results_dir / filename
        output_path.write_text(result.get_plain_text(), encoding="utf-8")

        logger.info("plain_text_generated", path=str(output_path))
        return output_path

    def generate_srt(
        self,
        result: AggregationResult,
        filename: str = "transcript.srt",
    ) -> Path:
        """
        Generate SRT subtitle file.

        Args:
            result: Aggregation result
            filename: Output filename

        Returns:
            Path to generated file
        """
        output_path = self.results_dir / filename

        lines = []
        for idx, seg in enumerate(result.segments, 1):
            # SRT format: HH:MM:SS,mmm
            start_ts = self._format_srt_timestamp(seg.start_seconds)
            end_ts = self._format_srt_timestamp(seg.end_seconds)

            lines.append(str(idx))
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(f"[{seg.speaker_id}] {seg.text}")
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")

        logger.info("srt_generated", path=str(output_path))
        return output_path

    def generate_json(
        self,
        result: AggregationResult,
        filename: str = "transcript.json",
    ) -> Path:
        """
        Generate JSON output with full details.

        Args:
            result: Aggregation result
            filename: Output filename

        Returns:
            Path to generated file
        """
        import json

        output_path = self.results_dir / filename

        data = {
            "metadata": {
                "duration_seconds": result.total_duration_seconds,
                "num_speakers": result.num_speakers,
                "total_words": result.total_words,
                "processing_time_seconds": result.processing_time_seconds,
                "diarization_time_seconds": result.diarization_time_seconds,
                "transcription_time_seconds": result.transcription_time_seconds,
            },
            "speaker_stats": result.get_speaker_stats(),
            "segments": [
                {
                    "speaker_id": seg.speaker_id,
                    "text": seg.text,
                    "start_seconds": seg.start_seconds,
                    "end_seconds": seg.end_seconds,
                    "confidence": seg.confidence,
                }
                for seg in result.segments
            ],
            "full_text": result.get_plain_text(),
        }

        output_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        logger.info("json_generated", path=str(output_path))
        return output_path

    def _format_duration(self, seconds: float) -> str:
        """Format duration for display."""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

    def _format_srt_timestamp(self, seconds: float) -> str:
        """Format timestamp for SRT format (HH:MM:SS,mmm)."""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
