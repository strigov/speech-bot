
import os
import sys
from typing import NamedTuple

# Force torchaudio to prefer legacy backends (soundfile/sox) instead of torchcodec.
# This needs to be set before torchaudio is imported.
os.environ.setdefault("TORCHAUDIO_USE_BACKEND_DISPATCHER", "0")

import torchaudio
import soundfile as sf

_original_load = getattr(torchaudio, "load", None)
_original_info = getattr(torchaudio, "info", None)

# Try to force the soundfile backend when available
if hasattr(torchaudio, "set_audio_backend"):
    try:
        torchaudio.set_audio_backend("soundfile")
    except Exception:
        pass

def _should_use_soundfile(error: Exception) -> bool:
    """Detect torchcodec-related failures and decide when to fall back."""
    message = str(error).lower()
    return "torchcodec" in message or "load_with_torchcodec" in message


# Define AudioMetaData if it doesn't exist
if not hasattr(torchaudio, "AudioMetaData"):
    class AudioMetaData(NamedTuple):
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int
        encoding: str

    torchaudio.AudioMetaData = AudioMetaData
    # Also inject into backend.common if it exists, as some libs might look there
    if hasattr(torchaudio, "backend") and hasattr(torchaudio.backend, "common"):
        torchaudio.backend.common.AudioMetaData = AudioMetaData


def _info_with_soundfile(filepath: str, format: str = None, buffer_size: int = 4096) -> torchaudio.AudioMetaData:
    try:
        sinfo = sf.info(filepath)
        return torchaudio.AudioMetaData(
            sample_rate=sinfo.samplerate,
            num_frames=sinfo.frames,
            num_channels=sinfo.channels,
            bits_per_sample=0,  # Soundfile doesn't always give bits per sample easily
            encoding=sinfo.subtype,
        )
    except Exception as e:
        # Fallback or re-raise
        raise RuntimeError(f"Failed to get audio info via soundfile: {e}")


def info(filepath: str, format: str = None, buffer_size: int = 4096) -> torchaudio.AudioMetaData:
    if _original_info is not None:
        try:
            return _original_info(filepath, format=format, buffer_size=buffer_size)
        except Exception as e:
            if not _should_use_soundfile(e):
                raise
    return _info_with_soundfile(filepath, format=format, buffer_size=buffer_size)


torchaudio.info = info

import torch


def _load_with_soundfile(
    filepath: str,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    format: str = None,
    buffer_size: int = 4096,
) -> tuple[torch.Tensor, int]:
    try:
        # soundfile reads as (frames, channels)
        data, sr = sf.read(
            filepath,
            start=frame_offset,
            stop=None if num_frames == -1 else frame_offset + num_frames,
            dtype="float32",
            always_2d=True,
        )

        # Convert to tensor
        tensor = torch.from_numpy(data)

        # Transpose to (channels, frames) if needed (torchaudio default is channels first)
        if channels_first:
            tensor = tensor.t()

        return tensor, sr
    except Exception as e:
        raise RuntimeError(f"Failed to load audio via soundfile: {e}")


def load(
    filepath: str,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    format: str = None,
    buffer_size: int = 4096,
) -> tuple[torch.Tensor, int]:
    if _original_load is not None:
        try:
            return _original_load(
                filepath,
                frame_offset=frame_offset,
                num_frames=num_frames,
                normalize=normalize,
                channels_first=channels_first,
                format=format,
                buffer_size=buffer_size,
            )
        except Exception as e:
            if not _should_use_soundfile(e):
                raise
    return _load_with_soundfile(
        filepath,
        frame_offset=frame_offset,
        num_frames=num_frames,
        normalize=normalize,
        channels_first=channels_first,
        format=format,
        buffer_size=buffer_size,
    )


torchaudio.load = load

# Patch backend functions
if not hasattr(torchaudio, "list_audio_backends"):
    def list_audio_backends():
        return ["soundfile"]
    torchaudio.list_audio_backends = list_audio_backends

if not hasattr(torchaudio, "get_audio_backend"):
    def get_audio_backend():
        return "soundfile"
    torchaudio.get_audio_backend = get_audio_backend

if not hasattr(torchaudio, "set_audio_backend"):
    def set_audio_backend(backend):
        pass # No-op
    torchaudio.set_audio_backend = set_audio_backend

print("Torchaudio patched successfully.")
