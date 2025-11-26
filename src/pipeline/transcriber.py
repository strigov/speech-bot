"""GigaAM ASR integration for speech-to-text transcription."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import structlog

logger = structlog.get_logger(__name__)

# Lazy imports for heavy dependencies
torch = None
torchaudio = None
AutoModel = None


def _import_dependencies():
    """Lazy import heavy dependencies."""
    global torch, torchaudio, AutoModel

    if torch is None:
        import torch as _torch
        # Patch torchaudio for compatibility with nightly builds
        try:
            from src.utils import torchaudio_patch
        except ImportError:
            pass

        import torchaudio as _torchaudio
        from transformers import AutoModel as _AutoModel

        torch = _torch
        torchaudio = _torchaudio
        AutoModel = _AutoModel


@dataclass
class TranscriptionSegment:
    """A transcribed segment with timing information."""

    text: str
    start_seconds: float
    end_seconds: float
    confidence: Optional[float] = None
    speaker_id: Optional[str] = None


@dataclass
class TranscriptionResult:
    """Result of transcription for an audio file."""

    segments: List[TranscriptionSegment] = field(default_factory=list)
    full_text: str = ""
    duration_seconds: float = 0.0
    processing_time_seconds: float = 0.0
    error: Optional[str] = None
    is_successful: bool = True


class GigaAMTranscriber:
    """
    GigaAM v3 ASR transcriber with singleton pattern.

    Handles model loading, batch processing, and error recovery.
    """

    _instance: Optional["GigaAMTranscriber"] = None
    _lock = asyncio.Lock()

    def __new__(cls, *args, **kwargs):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        model_id: str = "ai-sage/GigaAM-v3",
        device: str = "cuda",
        batch_size: int = 32,
        max_vram_gb: float = 14.0,
        longform_threshold_seconds: float = 25.0,
    ):
        """
        Initialize transcriber.

        Args:
            model_id: Hugging Face model ID for GigaAM
            device: Device to run model on ('cuda' or 'cpu')
            batch_size: Default batch size for processing
            max_vram_gb: Maximum VRAM to use
            longform_threshold_seconds: Duration above which to use transcribe_longform
        """
        if self._initialized:
            return

        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.max_vram_gb = max_vram_gb
        self.sample_rate = 16000
        # GigaAM's .transcribe() rejects audio longer than ~25s
        self.longform_threshold_seconds = longform_threshold_seconds

        self._model = None
        self._is_loaded = False
        self._current_batch_size = batch_size
        self._initialized = True

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    async def load_model(self) -> Tuple[bool, Optional[str]]:
        """
        Load GigaAM model (lazy loading).

        Returns:
            Tuple of (success, error_message)
        """
        if self._is_loaded:
            return True, None

        async with self._lock:
            # Double-check after acquiring lock
            if self._is_loaded:
                return True, None

            try:
                logger.info("loading_gigaam_model", model_id=self.model_id, device=self.device)

                # Import dependencies
                _import_dependencies()

                # Check CUDA availability if requested
                if self.device == "cuda" and not torch.cuda.is_available():
                    logger.warning("cuda_not_available_falling_back_to_cpu")
                    self.device = "cpu"

                # Load model in thread pool
                loop = asyncio.get_event_loop()

                def _load():
                    # GigaAM-v3 requires AutoModel with trust_remote_code
                    model = AutoModel.from_pretrained(
                        self.model_id,
                        trust_remote_code=True,
                        device_map=self.device if self.device == "cuda" else None,
                    )
                    if self.device != "cuda":
                        model = model.to(self.device)
                    model.eval()
                    return model

                self._model = await loop.run_in_executor(None, _load)
                self._is_loaded = True

                # Log GPU memory after loading
                if self.device == "cuda":
                    allocated = torch.cuda.memory_allocated() / (1024**3)
                    logger.info(
                        "gigaam_model_loaded",
                        device=self.device,
                        vram_allocated_gb=f"{allocated:.2f}",
                    )
                else:
                    logger.info("gigaam_model_loaded", device=self.device)

                return True, None

            except Exception as e:
                error_msg = f"Failed to load GigaAM model: {str(e)}"
                logger.error("model_load_failed", error=str(e))
                return False, error_msg

    async def unload_model(self) -> None:
        """Unload model to free memory."""
        async with self._lock:
            if self._model is not None:
                del self._model
                self._model = None

            self._is_loaded = False

            # Clear CUDA cache
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("gigaam_model_unloaded")

    def _load_audio(self, file_path: Path) -> Tuple[Optional["torch.Tensor"], Optional[str]]:
        """
        Load audio file and resample to target sample rate.

        Returns:
            Tuple of (waveform_tensor, error_message)
        """
        try:
            waveform, sample_rate = torchaudio.load(str(file_path))

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if necessary
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.sample_rate
                )
                waveform = resampler(waveform)

            return waveform.squeeze(0), None

        except Exception as e:
            return None, f"Failed to load audio: {str(e)}"

    def _extract_segment(
        self,
        waveform: "torch.Tensor",
        start_seconds: float,
        end_seconds: float,
    ) -> "torch.Tensor":
        """Extract audio segment from waveform."""
        start_sample = int(start_seconds * self.sample_rate)
        end_sample = int(end_seconds * self.sample_rate)

        # Ensure bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(waveform), end_sample)

        return waveform[start_sample:end_sample]

    def _transcribe_batch(
        self,
        segments: List["torch.Tensor"],
        temp_dir: Path,
    ) -> List[Tuple[str, Optional[float]]]:
        """
        Transcribe a batch of audio segments using GigaAM's .transcribe()
        or .transcribe_longform() when segments exceed the short-form limit.

        Returns:
            List of (text, confidence) tuples
        """
        if not segments:
            return []

        # Ensure dependencies are loaded
        if torch is None:
            _import_dependencies()

        results = []

        with torch.inference_mode():
            for i, segment_audio in enumerate(segments):
                # Save segment to temporary file for GigaAM's transcribe method
                segment_path = temp_dir / f"segment_{i}.wav"
                torchaudio.save(
                    str(segment_path),
                    segment_audio.unsqueeze(0),
                    self.sample_rate,
                )

                try:
                    # Calculate segment duration
                    duration_seconds = len(segment_audio) / self.sample_rate

                    # Use transcribe_longform for segments beyond the model limit (~25s)
                    if (
                        duration_seconds >= self.longform_threshold_seconds
                        and hasattr(self._model, "transcribe_longform")
                    ):
                        longform_output = self._model.transcribe_longform(str(segment_path))

                        if isinstance(longform_output, list):
                            parts = []
                            for item in longform_output:
                                if isinstance(item, dict):
                                    parts.append(str(item.get("transcription", "")).strip())
                                else:
                                    parts.append(str(item).strip())
                            text = " ".join(p for p in parts if p)
                        else:
                            text = str(longform_output)
                    else:
                        transcription = self._model.transcribe(str(segment_path))

                        # GigaAM returns string directly
                        if isinstance(transcription, str):
                            text = transcription
                        else:
                            # In case it returns a dict or other structure
                            text = str(transcription)

                    # No confidence scores available from GigaAM
                    results.append((text, None))

                finally:
                    # Clean up temp file
                    if segment_path.exists():
                        segment_path.unlink()

        return results

    async def transcribe_segments(
        self,
        audio_path: Path,
        segments: List[Tuple[float, float, Optional[str]]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> TranscriptionResult:
        """
        Transcribe multiple segments from an audio file.

        Args:
            audio_path: Path to audio file
            segments: List of (start_seconds, end_seconds, speaker_id) tuples
            progress_callback: Optional callback for progress updates (processed, total)

        Returns:
            TranscriptionResult with all transcribed segments
        """
        import time
        import tempfile
        start_time = time.time()

        # Ensure model is loaded
        success, error = await self.load_model()
        if not success:
            return TranscriptionResult(error=error, is_successful=False)

        # Load audio file
        loop = asyncio.get_event_loop()
        waveform, error = await loop.run_in_executor(
            None, self._load_audio, audio_path
        )
        if error:
            return TranscriptionResult(error=error, is_successful=False)

        total_duration = len(waveform) / self.sample_rate
        results: List[TranscriptionSegment] = []
        current_batch_size = self._current_batch_size

        # Create temporary directory for segment files
        temp_dir = Path(tempfile.mkdtemp(prefix="gigaam_"))

        try:
            # Process in batches
            total_segments = len(segments)
            processed = 0

            for batch_start in range(0, total_segments, current_batch_size):
                batch_end = min(batch_start + current_batch_size, total_segments)
                batch_segments = segments[batch_start:batch_end]

                # Extract audio for each segment
                audio_segments = []
                segment_info = []

                for start, end, speaker_id in batch_segments:
                    seg_audio = self._extract_segment(waveform, start, end)
                    if len(seg_audio) > 0:
                        audio_segments.append(seg_audio)
                        segment_info.append((start, end, speaker_id))

                if not audio_segments:
                    continue

                # Try transcription with retry logic
                retry_count = 0
                max_retries = 3

                while retry_count < max_retries:
                    try:
                        transcriptions = await loop.run_in_executor(
                            None,
                            self._transcribe_batch,
                            audio_segments,
                            temp_dir,
                        )

                        # Create result segments
                        for (text, confidence), (start, end, speaker_id) in zip(
                            transcriptions, segment_info
                        ):
                            results.append(TranscriptionSegment(
                                text=text.strip(),
                                start_seconds=start,
                                end_seconds=end,
                                confidence=confidence,
                                speaker_id=speaker_id,
                            ))

                        break  # Success, exit retry loop

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            # OOM error - reduce batch size and retry
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            current_batch_size = max(1, current_batch_size // 2)
                            self._current_batch_size = current_batch_size

                            logger.warning(
                                "oom_reducing_batch_size",
                                new_batch_size=current_batch_size,
                                retry=retry_count + 1,
                            )
                            retry_count += 1

                            if retry_count >= max_retries:
                                return TranscriptionResult(
                                    error=f"OOM error after {max_retries} retries",
                                    is_successful=False,
                                )
                        else:
                            raise

                processed = batch_end
                if progress_callback:
                    progress_callback(processed, total_segments)

                # Yield control to event loop
                await asyncio.sleep(0)

        finally:
            # Clean up temp directory
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

        # Build full text
        full_text = " ".join(seg.text for seg in results if seg.text)

        processing_time = time.time() - start_time

        logger.info(
            "transcription_complete",
            segments=len(results),
            duration_seconds=f"{total_duration:.1f}",
            processing_time=f"{processing_time:.1f}",
        )

        return TranscriptionResult(
            segments=results,
            full_text=full_text,
            duration_seconds=total_duration,
            processing_time_seconds=processing_time,
            is_successful=True,
        )

    async def transcribe_file(
        self,
        audio_path: Path,
        # Keep segments below GigaAM short-form threshold (~25s)
        max_segment_seconds: float = 24.0,
        min_segment_seconds: float = 0.5,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> TranscriptionResult:
        """
        Transcribe an entire audio file without diarization.

        Automatically splits into segments based on max duration.

        Args:
            audio_path: Path to audio file
            max_segment_seconds: Maximum segment duration
            min_segment_seconds: Minimum segment duration
            progress_callback: Optional callback for progress updates

        Returns:
            TranscriptionResult with transcribed segments
        """
        # Ensure model is loaded
        success, error = await self.load_model()
        if not success:
            return TranscriptionResult(error=error, is_successful=False)

        # Get audio duration
        loop = asyncio.get_event_loop()
        waveform, error = await loop.run_in_executor(
            None, self._load_audio, audio_path
        )
        if error:
            return TranscriptionResult(error=error, is_successful=False)

        total_duration = len(waveform) / self.sample_rate

        # Create segments
        segments = []
        current_start = 0.0

        while current_start < total_duration:
            segment_end = min(current_start + max_segment_seconds, total_duration)

            # Skip tiny final segments
            if segment_end - current_start < min_segment_seconds:
                break

            segments.append((current_start, segment_end, None))
            current_start = segment_end

        # Transcribe segments
        return await self.transcribe_segments(
            audio_path,
            segments,
            progress_callback=progress_callback,
        )

    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("transcriber_cache_cleared")

    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage."""
        if torch is None or not torch.cuda.is_available():
            return {"available": False}

        return {
            "available": True,
            "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
            "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
        }


# Convenience function to get singleton instance
def get_transcriber(
    model_id: str = "ai-sage/GigaAM-v3",
    device: str = "cuda",
    batch_size: int = 32,
) -> GigaAMTranscriber:
    """Get or create the singleton transcriber instance."""
    return GigaAMTranscriber(
        model_id=model_id,
        device=device,
        batch_size=batch_size,
    )
