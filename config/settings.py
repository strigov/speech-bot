"""Configuration management using Pydantic settings."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TelegramSettings(BaseSettings):
    """Telegram bot configuration."""

    model_config = SettingsConfigDict(env_prefix="TELEGRAM_")

    bot_token: str = Field(..., description="Telegram bot token")
    api_url: str = Field(
        default="https://api.telegram.org",
        description="Telegram API URL"
    )
    use_local_bot_api: bool = Field(
        default=False,
        description="Use local bot API server"
    )
    local_bot_api_url: str = Field(
        default="http://localhost:8081",
        description="Local bot API URL"
    )


class ModelSettings(BaseSettings):
    """ML model configuration."""

    model_config = SettingsConfigDict(env_prefix="MODEL_")

    hf_token: str = Field(default="", alias="HF_TOKEN", description="Hugging Face token")
    device: str = Field(default="cuda", description="Device for ASR model")
    diarization_device: str = Field(
        default="cuda",
        alias="DIARIZATION_DEVICE",
        description="Device for diarization model"
    )
    asr_batch_size: int = Field(
        default=32,
        alias="ASR_BATCH_SIZE",
        description="Batch size for ASR processing"
    )
    diarization_batch_size: int = Field(
        default=32,
        alias="DIARIZATION_BATCH_SIZE",
        description="Batch size for diarization"
    )


class ProcessingSettings(BaseSettings):
    """Audio processing limits and parameters."""

    model_config = SettingsConfigDict(env_prefix="")

    max_audio_duration_minutes: int = Field(
        default=180,
        alias="MAX_AUDIO_DURATION_MINUTES",
        description="Maximum audio duration in minutes"
    )
    max_file_size_mb: int = Field(
        default=500,
        alias="MAX_FILE_SIZE_MB",
        description="Maximum file size in MB"
    )
    max_queue_size: int = Field(
        default=50,
        alias="MAX_QUEUE_SIZE",
        description="Maximum queue size"
    )
    max_user_concurrent: int = Field(
        default=3,
        alias="MAX_USER_CONCURRENT",
        description="Maximum concurrent files per user"
    )
    max_user_per_hour: int = Field(
        default=10,
        alias="MAX_USER_PER_HOUR",
        description="Maximum files per user per hour"
    )
    processing_timeout_minutes: int = Field(
        default=30,
        alias="PROCESSING_TIMEOUT_MINUTES",
        description="Processing timeout in minutes"
    )


class MemorySettings(BaseSettings):
    """Memory management configuration."""

    model_config = SettingsConfigDict(env_prefix="")

    max_vram_gb: float = Field(
        default=14.0,
        alias="MAX_VRAM_GB",
        description="Maximum VRAM allocation in GB"
    )
    chunk_duration_minutes: int = Field(
        default=10,
        alias="CHUNK_DURATION_MINUTES",
        description="Chunk duration for long audio processing"
    )
    segment_max_seconds: int = Field(
        default=30,
        alias="SEGMENT_MAX_SECONDS",
        description="Maximum segment duration for ASR"
    )
    segment_min_seconds: float = Field(
        default=0.5,
        alias="SEGMENT_MIN_SECONDS",
        description="Minimum segment duration"
    )
    segment_padding_seconds: float = Field(
        default=0.1,
        alias="SEGMENT_PADDING_SECONDS",
        description="Padding at segment boundaries"
    )
    chunk_overlap_seconds: int = Field(
        default=30,
        alias="CHUNK_OVERLAP_SECONDS",
        description="Overlap between chunks"
    )


class PathSettings(BaseSettings):
    """File path configuration."""

    model_config = SettingsConfigDict(env_prefix="")

    temp_dir: Path = Field(
        default=Path("./temp"),
        alias="TEMP_DIR",
        description="Temporary files directory"
    )
    log_dir: Path = Field(
        default=Path("./logs"),
        alias="LOG_DIR",
        description="Log files directory"
    )
    checkpoint_dir: Path = Field(
        default=Path("./temp/checkpoints"),
        alias="CHECKPOINT_DIR",
        description="Checkpoint files directory"
    )
    max_temp_per_user_gb: float = Field(
        default=5.0,
        alias="MAX_TEMP_PER_USER_GB",
        description="Maximum temp storage per user in GB"
    )


class DevelopmentSettings(BaseSettings):
    """Development configuration."""

    model_config = SettingsConfigDict(env_prefix="")

    debug: bool = Field(default=False, alias="DEBUG")
    hot_reload: bool = Field(default=False, alias="HOT_RELOAD")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


class Settings(BaseSettings):
    """Main settings class combining all configurations."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    dev: DevelopmentSettings = Field(default_factory=DevelopmentSettings)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.paths.temp_dir.mkdir(parents=True, exist_ok=True)
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)
        self.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (self.paths.temp_dir / "audio").mkdir(exist_ok=True)
        (self.paths.temp_dir / "results").mkdir(exist_ok=True)


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()
