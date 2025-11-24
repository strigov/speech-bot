"""PyAnnote speaker diarization integration."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import structlog

logger = structlog.get_logger(__name__)

# Lazy imports for heavy dependencies
torch = None
Pipeline = None
Audio = None


def _import_dependencies():
    """Lazy import heavy dependencies."""
    global torch, Pipeline, Audio

    if torch is None:
        import torch as _torch
        from pyannote.audio import Pipeline as _Pipeline
        from pyannote.audio import Audio as _Audio

        torch = _torch
        Pipeline = _Pipeline
        Audio = _Audio


@dataclass
class SpeakerSegment:
    """A speaker segment with timing information."""

    speaker_id: str
    start_seconds: float
    end_seconds: float
    confidence: Optional[float] = None

    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end_seconds - self.start_seconds


@dataclass
class DiarizationResult:
    """Result of speaker diarization."""

    segments: List[SpeakerSegment] = field(default_factory=list)
    num_speakers: int = 0
    duration_seconds: float = 0.0
    processing_time_seconds: float = 0.0
    error: Optional[str] = None
    is_successful: bool = True

    def get_speaker_segments(self, speaker_id: str) -> List[SpeakerSegment]:
        """Get all segments for a specific speaker."""
        return [s for s in self.segments if s.speaker_id == speaker_id]

    def get_speakers(self) -> List[str]:
        """Get list of unique speaker IDs."""
        return list(set(s.speaker_id for s in self.segments))


class PyAnnoteDiarizer:
    """
    PyAnnote speaker diarization with singleton pattern.

    Handles model loading, chunked processing for long audio,
    and speaker consistency across chunks.
    """

    _instance: Optional["PyAnnoteDiarizer"] = None
    _lock = asyncio.Lock()

    def __new__(cls, *args, **kwargs):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        model_id: str = "pyannote/speaker-diarization-3.1",
        hf_token: str = "",
        device: str = "cuda",
        batch_size: int = 32,
        min_speakers: int = 1,
        max_speakers: int = 20,
    ):
        """
        Initialize diarizer.

        Args:
            model_id: Hugging Face model ID for PyAnnote
            hf_token: Hugging Face access token (required for PyAnnote)
            device: Device to run model on ('cuda' or 'cpu')
            batch_size: Batch size for embedding extraction
            min_speakers: Minimum expected speakers
            max_speakers: Maximum expected speakers
        """
        if self._initialized:
            return

        self.model_id = model_id
        self.hf_token = hf_token
        self.device = device
        self.batch_size = batch_size
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

        self._pipeline = None
        self._is_loaded = False
        self._initialized = True

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def set_hf_token(self, token: str) -> None:
        """Set Hugging Face token (needed before loading)."""
        self.hf_token = token

    async def load_model(self) -> Tuple[bool, Optional[str]]:
        """
        Load PyAnnote diarization pipeline (lazy loading).

        Returns:
            Tuple of (success, error_message)
        """
        if self._is_loaded:
            return True, None

        async with self._lock:
            # Double-check after acquiring lock
            if self._is_loaded:
                return True, None

            if not self.hf_token:
                return False, "Hugging Face token required for PyAnnote models"

            try:
                logger.info(
                    "loading_pyannote_model",
                    model_id=self.model_id,
                    device=self.device,
                )

                # Import dependencies
                _import_dependencies()

                # Check CUDA availability
                if self.device == "cuda" and not torch.cuda.is_available():
                    logger.warning("cuda_not_available_falling_back_to_cpu")
                    self.device = "cpu"

                # Load pipeline in thread pool
                loop = asyncio.get_event_loop()

                def _load():
                    pipeline = Pipeline.from_pretrained(
                        self.model_id,
                        use_auth_token=self.hf_token,
                    )
                    pipeline = pipeline.to(torch.device(self.device))
                    return pipeline

                self._pipeline = await loop.run_in_executor(None, _load)
                self._is_loaded = True

                # Log GPU memory after loading
                if self.device == "cuda":
                    allocated = torch.cuda.memory_allocated() / (1024**3)
                    logger.info(
                        "pyannote_model_loaded",
                        device=self.device,
                        vram_allocated_gb=f"{allocated:.2f}",
                    )
                else:
                    logger.info("pyannote_model_loaded", device=self.device)

                return True, None

            except Exception as e:
                error_msg = f"Failed to load PyAnnote model: {str(e)}"
                logger.error("model_load_failed", error=str(e))
                return False, error_msg

    async def unload_model(self) -> None:
        """Unload model to free memory."""
        async with self._lock:
            if self._pipeline is not None:
                del self._pipeline
                self._pipeline = None

            self._is_loaded = False

            # Clear CUDA cache
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("pyannote_model_unloaded")

    def _diarize_chunk(
        self,
        audio_path: Path,
        start_seconds: float = 0.0,
        end_seconds: Optional[float] = None,
    ) -> List[SpeakerSegment]:
        """
        Run diarization on a chunk of audio.

        Args:
            audio_path: Path to audio file
            start_seconds: Start time in seconds
            end_seconds: End time in seconds (None for full file)

        Returns:
            List of speaker segments
        """
        # Prepare audio input
        if end_seconds is not None:
            # Process specific chunk
            from pyannote.core import Segment
            audio_input = {
                "audio": str(audio_path),
                "onset": start_seconds,
                "offset": end_seconds,
            }
        else:
            audio_input = str(audio_path)

        # Run diarization
        diarization = self._pipeline(
            audio_input,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )

        # Convert to segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Adjust timing for chunk offset
            seg_start = turn.start + start_seconds
            seg_end = turn.end + start_seconds

            segments.append(SpeakerSegment(
                speaker_id=speaker,
                start_seconds=seg_start,
                end_seconds=seg_end,
            ))

        return segments

    def _merge_speaker_labels(
        self,
        all_chunks: List[List[SpeakerSegment]],
        overlap_seconds: float = 30.0,
    ) -> List[SpeakerSegment]:
        """
        Merge speaker segments from multiple chunks with consistent labeling.

        Uses overlap regions to match speakers across chunks.

        Args:
            all_chunks: List of segment lists from each chunk
            overlap_seconds: Duration of overlap between chunks

        Returns:
            Merged list of segments with consistent speaker IDs
        """
        if not all_chunks:
            return []

        if len(all_chunks) == 1:
            return all_chunks[0]

        # First chunk establishes speaker mapping
        merged_segments = list(all_chunks[0])
        global_speaker_map: Dict[str, str] = {
            seg.speaker_id: seg.speaker_id for seg in merged_segments
        }
        next_speaker_id = 0

        # Find highest speaker number from first chunk
        for seg in merged_segments:
            if seg.speaker_id.startswith("SPEAKER_"):
                try:
                    num = int(seg.speaker_id.split("_")[1])
                    next_speaker_id = max(next_speaker_id, num + 1)
                except (IndexError, ValueError):
                    pass

        # Process subsequent chunks
        for chunk_idx in range(1, len(all_chunks)):
            chunk_segments = all_chunks[chunk_idx]
            if not chunk_segments:
                continue

            # Find chunk boundaries
            chunk_start = min(s.start_seconds for s in chunk_segments)
            overlap_end = chunk_start + overlap_seconds

            # Get segments in overlap region from both sides
            prev_overlap_segments = [
                s for s in merged_segments
                if s.end_seconds > chunk_start and s.start_seconds < overlap_end
            ]
            curr_overlap_segments = [
                s for s in chunk_segments
                if s.start_seconds < overlap_end
            ]

            # Build speaker mapping for this chunk based on overlap
            chunk_speaker_map: Dict[str, str] = {}

            for curr_seg in curr_overlap_segments:
                curr_speaker = curr_seg.speaker_id
                if curr_speaker in chunk_speaker_map:
                    continue

                # Find overlapping speaker from previous chunk
                best_match = None
                best_overlap = 0.0

                for prev_seg in prev_overlap_segments:
                    # Calculate overlap duration
                    overlap_start = max(curr_seg.start_seconds, prev_seg.start_seconds)
                    overlap_end_time = min(curr_seg.end_seconds, prev_seg.end_seconds)
                    overlap_duration = max(0, overlap_end_time - overlap_start)

                    if overlap_duration > best_overlap:
                        best_overlap = overlap_duration
                        best_match = prev_seg.speaker_id

                if best_match and best_overlap > 0.5:
                    # Map to existing speaker
                    chunk_speaker_map[curr_speaker] = global_speaker_map.get(
                        best_match, best_match
                    )
                else:
                    # New speaker
                    new_id = f"SPEAKER_{next_speaker_id:02d}"
                    chunk_speaker_map[curr_speaker] = new_id
                    next_speaker_id += 1

            # Map any remaining speakers not in overlap
            for seg in chunk_segments:
                if seg.speaker_id not in chunk_speaker_map:
                    new_id = f"SPEAKER_{next_speaker_id:02d}"
                    chunk_speaker_map[seg.speaker_id] = new_id
                    next_speaker_id += 1

            # Update global mapping
            global_speaker_map.update(chunk_speaker_map)

            # Add non-overlapping segments with remapped speakers
            for seg in chunk_segments:
                # Skip segments fully in overlap (already covered)
                if seg.end_seconds <= overlap_end:
                    continue

                mapped_speaker = chunk_speaker_map.get(seg.speaker_id, seg.speaker_id)

                # For segments starting in overlap, adjust start time
                if seg.start_seconds < overlap_end:
                    # Trim to avoid duplication
                    merged_segments.append(SpeakerSegment(
                        speaker_id=mapped_speaker,
                        start_seconds=overlap_end,
                        end_seconds=seg.end_seconds,
                    ))
                else:
                    merged_segments.append(SpeakerSegment(
                        speaker_id=mapped_speaker,
                        start_seconds=seg.start_seconds,
                        end_seconds=seg.end_seconds,
                    ))

        # Sort by start time
        merged_segments.sort(key=lambda s: s.start_seconds)

        return merged_segments

    async def diarize(
        self,
        audio_path: Path,
        chunk_duration_seconds: float = 600.0,  # 10 minutes
        overlap_seconds: float = 30.0,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> DiarizationResult:
        """
        Run speaker diarization on an audio file.

        For long files, processes in chunks with overlap for speaker consistency.

        Args:
            audio_path: Path to audio file (should be 16kHz mono WAV)
            chunk_duration_seconds: Duration of each chunk for long audio
            overlap_seconds: Overlap between chunks
            progress_callback: Optional callback (chunk_num, total_chunks, status)

        Returns:
            DiarizationResult with speaker segments
        """
        import time
        start_time = time.time()

        # Ensure model is loaded
        success, error = await self.load_model()
        if not success:
            return DiarizationResult(error=error, is_successful=False)

        # Get audio duration
        try:
            _import_dependencies()
            audio = Audio(mono="downmix", sample_rate=16000)
            waveform, sample_rate = audio(audio_path)
            total_duration = waveform.shape[1] / sample_rate
        except Exception as e:
            return DiarizationResult(
                error=f"Failed to load audio: {str(e)}",
                is_successful=False,
            )

        loop = asyncio.get_event_loop()

        # Determine if chunking is needed
        if total_duration <= chunk_duration_seconds:
            # Process whole file
            if progress_callback:
                progress_callback(1, 1, "Processing...")

            try:
                segments = await loop.run_in_executor(
                    None,
                    self._diarize_chunk,
                    audio_path,
                    0.0,
                    None,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return DiarizationResult(
                        error="Out of GPU memory. Try shorter audio or reduce batch size.",
                        is_successful=False,
                    )
                raise
        else:
            # Process in chunks
            chunks = []
            current_start = 0.0

            while current_start < total_duration:
                chunk_end = min(current_start + chunk_duration_seconds, total_duration)
                chunks.append((current_start, chunk_end))
                current_start = chunk_end - overlap_seconds

            logger.info(
                "diarizing_in_chunks",
                total_chunks=len(chunks),
                duration=f"{total_duration:.1f}s",
            )

            all_chunk_segments = []

            for idx, (chunk_start, chunk_end) in enumerate(chunks):
                if progress_callback:
                    progress_callback(idx + 1, len(chunks), f"Chunk {idx + 1}/{len(chunks)}")

                try:
                    chunk_segments = await loop.run_in_executor(
                        None,
                        self._diarize_chunk,
                        audio_path,
                        chunk_start,
                        chunk_end,
                    )
                    all_chunk_segments.append(chunk_segments)

                    # Clear cache between chunks
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.error("oom_during_diarization", chunk=idx + 1)
                        return DiarizationResult(
                            error=f"Out of memory at chunk {idx + 1}",
                            is_successful=False,
                        )
                    raise

                # Yield control
                await asyncio.sleep(0)

            # Merge chunks with speaker consistency
            segments = self._merge_speaker_labels(all_chunk_segments, overlap_seconds)

        # Count unique speakers
        unique_speakers = set(s.speaker_id for s in segments)

        processing_time = time.time() - start_time

        logger.info(
            "diarization_complete",
            segments=len(segments),
            speakers=len(unique_speakers),
            duration_seconds=f"{total_duration:.1f}",
            processing_time=f"{processing_time:.1f}",
        )

        return DiarizationResult(
            segments=segments,
            num_speakers=len(unique_speakers),
            duration_seconds=total_duration,
            processing_time_seconds=processing_time,
            is_successful=True,
        )

    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("diarizer_cache_cleared")

    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage."""
        if torch is None or not torch.cuda.is_available():
            return {"available": False}

        return {
            "available": True,
            "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        }


# Convenience function to get singleton instance
def get_diarizer(
    model_id: str = "pyannote/speaker-diarization-3.1",
    hf_token: str = "",
    device: str = "cuda",
) -> PyAnnoteDiarizer:
    """Get or create the singleton diarizer instance."""
    return PyAnnoteDiarizer(
        model_id=model_id,
        hf_token=hf_token,
        device=device,
    )
