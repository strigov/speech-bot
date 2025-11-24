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
Wav2Vec2ForCTC = None
Wav2Vec2Processor = None


def _import_dependencies():
    """Lazy import heavy dependencies."""
    global torch, torchaudio, Wav2Vec2ForCTC, Wav2Vec2Processor

    if torch is None:
        import torch as _torch
        import torchaudio as _torchaudio
        from transformers import Wav2Vec2ForCTC as _Wav2Vec2ForCTC
        from transformers import Wav2Vec2Processor as _Wav2Vec2Processor

        torch = _torch
        torchaudio = _torchaudio
        Wav2Vec2ForCTC = _Wav2Vec2ForCTC
        Wav2Vec2Processor = _Wav2Vec2Processor


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
    ):
        """
        Initialize transcriber.

        Args:
            model_id: Hugging Face model ID for GigaAM
            device: Device to run model on ('cuda' or 'cpu')
            batch_size: Default batch size for processing
            max_vram_gb: Maximum VRAM to use
        """
        if self._initialized:
            return

        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.max_vram_gb = max_vram_gb
        self.sample_rate = 16000

        self._model = None
        self._processor = None
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

                # Load processor and model in thread pool
                loop = asyncio.get_event_loop()

                def _load():
                    processor = Wav2Vec2Processor.from_pretrained(self.model_id)
                    model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
                    model = model.to(self.device)
                    model.eval()
                    return processor, model

                self._processor, self._model = await loop.run_in_executor(None, _load)
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

            if self._processor is not None:
                del self._processor
                self._processor = None

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
    ) -> List[Tuple[str, Optional[float]]]:
        """
        Transcribe a batch of audio segments.

        Returns:
            List of (text, confidence) tuples
        """
        if not segments:
            return []

        # Ensure dependencies are loaded
        if torch is None:
            _import_dependencies()

        with torch.inference_mode():
            # Pad segments to same length
            max_length = max(len(s) for s in segments)
            padded = []
            attention_masks = []

            for seg in segments:
                padding = max_length - len(seg)
                if padding > 0:
                    padded_seg = torch.nn.functional.pad(seg, (0, padding))
                else:
                    padded_seg = seg
                padded.append(padded_seg)

                # Create attention mask
                mask = torch.ones(max_length)
                if padding > 0:
                    mask[-padding:] = 0
                attention_masks.append(mask)

            # Stack into batch
            batch_waveforms = torch.stack(padded)
            batch_attention = torch.stack(attention_masks)

            # Process through model
            inputs = self._processor(
                batch_waveforms.numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            )

            input_values = inputs.input_values.to(self.device)
            if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
                attention_mask = inputs.attention_mask.to(self.device)
            else:
                attention_mask = batch_attention.to(self.device)

            # Get logits
            outputs = self._model(input_values, attention_mask=attention_mask)
            logits = outputs.logits

            # Decode predictions
            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = self._processor.batch_decode(predicted_ids)

            # Calculate confidence scores (optional)
            confidences = []
            probs = torch.nn.functional.softmax(logits, dim=-1)
            for i in range(len(segments)):
                # Average of max probabilities for each timestep
                max_probs = probs[i].max(dim=-1).values
                # Exclude padding
                valid_length = int(len(segments[i]) / self.sample_rate * 50)  # ~50 frames per second
                valid_probs = max_probs[:valid_length] if valid_length > 0 else max_probs
                confidence = valid_probs.mean().item()
                confidences.append(confidence)

            return list(zip(transcriptions, confidences))

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
        max_segment_seconds: float = 30.0,
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
