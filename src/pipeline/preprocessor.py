"""Audio preprocessing pipeline for conversion and validation."""

import asyncio
import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import structlog

from src.utils.audio import (
    AudioInfo,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_MIME_TYPES,
    TARGET_SAMPLE_RATE,
    convert_to_wav,
    get_audio_info,
    is_supported_format,
)
from src.utils.file_manager import FileManager

logger = structlog.get_logger(__name__)

# Magic bytes for common audio/video formats
MAGIC_BYTES = {
    b"ID3": "mp3",  # MP3 with ID3 tag
    b"\xff\xfb": "mp3",  # MP3 frame sync
    b"\xff\xfa": "mp3",  # MP3 frame sync
    b"\xff\xf3": "mp3",  # MP3 frame sync
    b"\xff\xf2": "mp3",  # MP3 frame sync
    b"RIFF": "wav",  # WAV
    b"OggS": "ogg",  # Ogg container (Vorbis, Opus)
    b"fLaC": "flac",  # FLAC
    b"\x1aE\xdf\xa3": "webm",  # WebM/Matroska
    b"\x00\x00\x00": "mp4",  # MP4/M4A (partial, needs more checks)
}


@dataclass
class PreprocessingResult:
    """Result of audio preprocessing."""

    original_path: Path
    processed_path: Optional[Path]
    audio_info: Optional[AudioInfo]
    task_id: str
    user_id: int
    is_valid: bool
    error: Optional[str] = None
    warning: Optional[str] = None

    @property
    def duration_seconds(self) -> float:
        """Get audio duration."""
        return self.audio_info.duration_seconds if self.audio_info else 0.0


class AudioPreprocessor:
    """
    Preprocesses audio files for ASR pipeline.

    Handles:
    - File validation (format, size, integrity)
    - Conversion to 16kHz mono WAV
    - Temporary file management
    - Security checks (path traversal, magic bytes)
    """

    def __init__(
        self,
        file_manager: FileManager,
        max_file_size_mb: float = 500.0,
        max_duration_minutes: float = 180.0,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
    ):
        """
        Initialize preprocessor.

        Args:
            file_manager: FileManager instance for temp files
            max_file_size_mb: Maximum allowed file size in MB
            max_duration_minutes: Maximum audio duration in minutes
            target_sample_rate: Target sample rate for output (default 16kHz)
        """
        self.file_manager = file_manager
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self.max_duration_seconds = max_duration_minutes * 60
        self.target_sample_rate = target_sample_rate

    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        return str(uuid.uuid4())[:8]

    def _validate_magic_bytes(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate file by checking magic bytes.

        Returns:
            Tuple of (is_valid, detected_format)
        """
        try:
            with open(file_path, "rb") as f:
                header = f.read(12)

            if len(header) < 4:
                return False, None

            # Check known magic bytes
            for magic, format_name in MAGIC_BYTES.items():
                if header.startswith(magic):
                    return True, format_name

            # Special check for MP4/M4A (ftyp box)
            if len(header) >= 8 and header[4:8] == b"ftyp":
                return True, "mp4"

            # Check for WAV RIFF header
            if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
                return True, "wav"

            logger.warning(
                "unknown_magic_bytes",
                header_hex=header[:8].hex(),
                path=str(file_path),
            )
            return False, None

        except Exception as e:
            logger.error("magic_bytes_check_failed", error=str(e))
            return False, None

    def _validate_file_size(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate file size.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            size = file_path.stat().st_size

            if size == 0:
                return False, "File is empty"

            if size > self.max_file_size_bytes:
                max_mb = self.max_file_size_bytes / (1024 * 1024)
                actual_mb = size / (1024 * 1024)
                return False, f"File too large: {actual_mb:.1f}MB (max: {max_mb:.0f}MB)"

            return True, None

        except Exception as e:
            return False, f"Cannot read file size: {str(e)}"

    def _validate_extension(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate file extension.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not is_supported_format(file_path):
            ext = file_path.suffix.lower()
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            return False, f"Unsupported format: {ext}. Supported: {supported}"

        return True, None

    async def _validate_audio_integrity(
        self,
        file_path: Path,
    ) -> Tuple[bool, Optional[AudioInfo], Optional[str]]:
        """
        Validate audio file integrity using ffprobe.

        Returns:
            Tuple of (is_valid, audio_info, error_message)
        """
        audio_info = await get_audio_info(file_path)

        if not audio_info.is_valid:
            return False, None, audio_info.error or "Invalid audio file"

        if audio_info.duration_seconds <= 0:
            return False, None, "Audio has no duration"

        if audio_info.duration_seconds > self.max_duration_seconds:
            max_min = self.max_duration_seconds / 60
            actual_min = audio_info.duration_seconds / 60
            return False, audio_info, f"Audio too long: {actual_min:.1f}min (max: {max_min:.0f}min)"

        if audio_info.sample_rate <= 0:
            return False, None, "Invalid sample rate"

        return True, audio_info, None

    async def validate_file(
        self,
        file_path: Path,
        check_magic_bytes: bool = True,
    ) -> Tuple[bool, Optional[AudioInfo], Optional[str]]:
        """
        Perform full validation on an audio file.

        Args:
            file_path: Path to audio file
            check_magic_bytes: Whether to verify magic bytes

        Returns:
            Tuple of (is_valid, audio_info, error_message)
        """
        # Check file exists
        if not file_path.exists():
            logger.error(
                "file_not_found_during_validation",
                requested_path=str(file_path),
                path_is_absolute=file_path.is_absolute(),
                path_exists=file_path.exists(),
            )
            return False, None, "File not found"

        # Validate extension
        valid, error = self._validate_extension(file_path)
        if not valid:
            return False, None, error

        # Validate file size
        valid, error = self._validate_file_size(file_path)
        if not valid:
            return False, None, error

        # Validate magic bytes
        if check_magic_bytes:
            valid, detected_format = self._validate_magic_bytes(file_path)
            if not valid:
                return False, None, "File format verification failed (invalid header)"

        # Validate audio integrity
        valid, audio_info, error = await self._validate_audio_integrity(file_path)
        if not valid:
            return False, audio_info, error

        return True, audio_info, None

    async def preprocess(
        self,
        input_path: Path,
        user_id: int,
        task_id: Optional[str] = None,
        validate: bool = True,
    ) -> PreprocessingResult:
        """
        Preprocess an audio file for ASR.

        Steps:
        1. Validate file (format, size, integrity)
        2. Create task directory
        3. Convert to 16kHz mono WAV
        4. Return processed file path

        Args:
            input_path: Path to input audio file
            user_id: User ID for file isolation
            task_id: Optional task ID (generated if not provided)
            validate: Whether to perform full validation

        Returns:
            PreprocessingResult with processed file path
        """
        task_id = task_id or self._generate_task_id()

        logger.info(
            "preprocessing_started",
            user_id=user_id,
            task_id=task_id,
            input_file=input_path.name,
            input_path=str(input_path),
            input_path_exists=input_path.exists(),
            input_path_is_absolute=input_path.is_absolute(),
        )

        # Validate if requested
        audio_info = None
        if validate:
            valid, audio_info, error = await self.validate_file(input_path)
            if not valid:
                logger.warning(
                    "validation_failed",
                    user_id=user_id,
                    task_id=task_id,
                    error=error,
                )
                return PreprocessingResult(
                    original_path=input_path,
                    processed_path=None,
                    audio_info=audio_info,
                    task_id=task_id,
                    user_id=user_id,
                    is_valid=False,
                    error=error,
                )

        # Check user quota
        file_size = input_path.stat().st_size
        if not self.file_manager.check_user_quota(user_id, file_size * 2):  # *2 for converted file
            return PreprocessingResult(
                original_path=input_path,
                processed_path=None,
                audio_info=audio_info,
                task_id=task_id,
                user_id=user_id,
                is_valid=False,
                error="Storage quota exceeded. Please wait for previous files to be processed.",
            )

        # Create task directory
        task_dir = self.file_manager.get_task_dir(user_id, task_id)

        # Determine output path
        output_path = task_dir / "audio.wav"

        # Check if conversion is needed
        needs_conversion = True
        if audio_info:
            # Skip conversion if already in correct format
            if (
                audio_info.codec == "pcm_s16le"
                and audio_info.sample_rate == self.target_sample_rate
                and audio_info.channels == 1
                and input_path.suffix.lower() == ".wav"
            ):
                needs_conversion = False
                logger.debug("skipping_conversion", reason="already_correct_format")

        if needs_conversion:
            # Convert to target format
            success, error = await convert_to_wav(
                input_path,
                output_path,
                sample_rate=self.target_sample_rate,
                channels=1,
            )

            if not success:
                logger.error(
                    "conversion_failed",
                    user_id=user_id,
                    task_id=task_id,
                    error=error,
                )
                return PreprocessingResult(
                    original_path=input_path,
                    processed_path=None,
                    audio_info=audio_info,
                    task_id=task_id,
                    user_id=user_id,
                    is_valid=False,
                    error=f"Conversion failed: {error}",
                )
        else:
            # Copy file if no conversion needed
            import shutil
            shutil.copy2(input_path, output_path)

        # Get info for converted file
        converted_info = await get_audio_info(output_path)
        if not converted_info.is_valid:
            return PreprocessingResult(
                original_path=input_path,
                processed_path=None,
                audio_info=audio_info,
                task_id=task_id,
                user_id=user_id,
                is_valid=False,
                error="Converted file is invalid",
            )

        logger.info(
            "preprocessing_complete",
            user_id=user_id,
            task_id=task_id,
            duration_seconds=f"{converted_info.duration_seconds:.1f}",
            output_size_mb=f"{converted_info.file_size_bytes / (1024*1024):.1f}",
        )

        return PreprocessingResult(
            original_path=input_path,
            processed_path=output_path,
            audio_info=converted_info,
            task_id=task_id,
            user_id=user_id,
            is_valid=True,
        )

    async def preprocess_from_bytes(
        self,
        data: bytes,
        filename: str,
        user_id: int,
        task_id: Optional[str] = None,
    ) -> PreprocessingResult:
        """
        Preprocess audio from bytes (e.g., downloaded from Telegram).

        Args:
            data: Raw file bytes
            filename: Original filename
            user_id: User ID
            task_id: Optional task ID

        Returns:
            PreprocessingResult
        """
        task_id = task_id or self._generate_task_id()

        # Get safe path for input file
        safe_path = self.file_manager.get_safe_path(user_id, filename)
        task_dir = self.file_manager.get_task_dir(user_id, task_id)
        input_path = task_dir / f"input{safe_path.suffix}"

        # Write data to file
        try:
            input_path.write_bytes(data)
        except Exception as e:
            return PreprocessingResult(
                original_path=input_path,
                processed_path=None,
                audio_info=None,
                task_id=task_id,
                user_id=user_id,
                is_valid=False,
                error=f"Failed to save file: {str(e)}",
            )

        # Process the saved file
        return await self.preprocess(
            input_path,
            user_id,
            task_id=task_id,
            validate=True,
        )

    def cleanup_task(self, user_id: int, task_id: str) -> None:
        """Clean up files for a completed/failed task."""
        self.file_manager.cleanup_task_files(user_id, task_id)

    def get_task_dir(self, user_id: int, task_id: str) -> Path:
        """Get the directory for a task."""
        return self.file_manager.get_task_dir(user_id, task_id)

    def get_results_dir(self, user_id: int) -> Path:
        """Get the results directory for a user."""
        return self.file_manager.get_results_dir(user_id)


def create_preprocessor(
    temp_dir: Path,
    max_file_size_mb: float = 500.0,
    max_duration_minutes: float = 180.0,
    max_per_user_gb: float = 5.0,
) -> AudioPreprocessor:
    """
    Create a preprocessor with a new FileManager.

    Args:
        temp_dir: Base temporary directory
        max_file_size_mb: Maximum file size in MB
        max_duration_minutes: Maximum audio duration in minutes
        max_per_user_gb: Maximum storage per user in GB

    Returns:
        Configured AudioPreprocessor
    """
    file_manager = FileManager(
        temp_dir=temp_dir,
        max_per_user_gb=max_per_user_gb,
    )

    return AudioPreprocessor(
        file_manager=file_manager,
        max_file_size_mb=max_file_size_mb,
        max_duration_minutes=max_duration_minutes,
    )
