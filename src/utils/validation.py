"""File validation utilities."""

from pathlib import Path
from typing import Optional, Tuple


class FileValidator:
    """Validates files based on magic bytes and structure."""

    # Magic bytes signatures
    SIGNATURES = {
        "mp3": [
            (0, b"\x49\x44\x33"),  # ID3
            (0, b"\xFF\xFB"),      # MPEG-1 Layer 3
            (0, b"\xFF\xFA"),      # MPEG-1 Layer 3 with CRC
            (0, b"\xFF\xF3"),      # MPEG-1 Layer 3
            (0, b"\xFF\xF2"),      # MPEG-1 Layer 3
        ],
        "wav": [
            (0, b"\x52\x49\x46\x46"),  # RIFF
        ],
        "ogg": [
            (0, b"\x4F\x67\x67\x53"),  # OggS
        ],
        "flac": [
            (0, b"\x66\x4C\x61\x43"),  # fLaC
        ],
        "m4a": [
            (4, b"\x66\x74\x79\x70"),  # ftyp
        ],
        "mp4": [
            (4, b"\x66\x74\x79\x70"),  # ftyp
        ],
        "webm": [
            (0, b"\x1A\x45\xDF\xA3"),  # EBML
        ],
        "wma": [
            (0, b"\x30\x26\xB2\x75\x8E\x66\xCF\x11"),
        ],
        "opus": [
            (0, b"\x4F\x67\x67\x53"),  # OggS (Opus is usually in Ogg container)
        ]
    }

    @staticmethod
    def is_mp3_frame_header(header: bytes) -> bool:
        """Check MPEG audio frame sync headers, including MPEG 2.5 variants."""
        if len(header) < 4:
            return False

        if header[0] != 0xFF or (header[1] & 0xE0) != 0xE0:
            return False

        version_id = (header[1] >> 3) & 0x03
        layer = (header[1] >> 1) & 0x03
        bitrate_index = (header[2] >> 4) & 0x0F
        sample_rate_index = (header[2] >> 2) & 0x03

        return (
            version_id != 0x01
            and layer != 0x00
            and bitrate_index != 0x0F
            and sample_rate_index != 0x03
        )

    @classmethod
    def validate_file(cls, file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate file using magic bytes.

        Args:
            file_path: Path to file

        Returns:
            Tuple (is_valid, error_message)
        """
        if not file_path.exists():
            return False, "File does not exist"

        try:
            with open(file_path, "rb") as f:
                header = f.read(32)  # Read first 32 bytes

            if cls.is_mp3_frame_header(header):
                return True, None

            # Check against signatures
            for fmt, sigs in cls.SIGNATURES.items():
                for offset, sig in sigs:
                    if len(header) >= offset + len(sig):
                        if header[offset:offset+len(sig)] == sig:
                            return True, None
            
            # If no match found, log it but maybe be lenient if extension matches?
            # For security, we should be strict.
            return False, "Invalid file format (magic bytes mismatch)"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    @classmethod
    def is_safe_path(cls, base_dir: Path, path: Path) -> bool:
        """
        Check if path is safe (no traversal).
        
        Args:
            base_dir: Base directory
            path: Path to check
            
        Returns:
            True if path is within base_dir
        """
        try:
            # Resolve resolves symlinks and absolute path
            base = base_dir.resolve()
            target = path.resolve()
            return base in target.parents or base == target
        except Exception:
            return False
