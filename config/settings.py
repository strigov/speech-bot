"""Configuration management using Pydantic settings."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TelegramSettings(BaseSettings):
    """Telegram bot configuration."""

    model_config = SettingsConfigDict(env_prefix="TELEGRAM_", extra="ignore")

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

    # Telegram Client API (MTProto) for downloading large files
    api_id: Optional[int] = Field(
        default=None,
        description="Telegram API ID from my.telegram.org"
    )
    api_hash: Optional[str] = Field(
        default=None,
        description="Telegram API Hash from my.telegram.org"
    )
    use_client_api: bool = Field(
        default=False,
        description="Use Telegram Client API for downloading files > 20MB"
    )
    session_name: str = Field(
        default="bot_session",
        description="Session name for Telegram Client"
    )


class ModelSettings(BaseSettings):
    """ML model configuration."""

    model_config = SettingsConfigDict(env_prefix="MODEL_", extra="ignore")

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

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

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

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

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

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

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

    def model_post_init(self, __context) -> None:
        """Convert relative paths to absolute paths."""
        # Convert to absolute paths relative to project root
        self.temp_dir = self.temp_dir.resolve()
        self.log_dir = self.log_dir.resolve()
        self.checkpoint_dir = self.checkpoint_dir.resolve()


class DevelopmentSettings(BaseSettings):
    """Development configuration."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    debug: bool = Field(default=False, alias="DEBUG")
    hot_reload: bool = Field(default=False, alias="HOT_RELOAD")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


class PerUserLimits(BaseModel):
    """Per-user rate limits."""

    max_concurrent: int = 1
    max_per_hour: int = 10
    cooldown_seconds: int = 60


class GlobalLimits(BaseModel):
    """Global queue limits."""

    max_queue_size: int = 50
    max_active_tasks: int = 3


class RateLimits(BaseModel):
    """Rate limiting configuration."""

    model_config = ConfigDict(populate_by_name=True)

    per_user: PerUserLimits = Field(default_factory=PerUserLimits)
    global_: GlobalLimits = Field(default_factory=GlobalLimits, alias="global")


class FileLimits(BaseModel):
    """File size and format limits."""

    max_size_mb: int = 150
    max_duration_minutes: int = 180
    allowed_extensions: list[str] = Field(default_factory=lambda: [
        ".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".wma", ".opus", ".mp4", ".webm",
    ])
    allowed_mime_types: list[str] = Field(default_factory=lambda: [
        "audio/mpeg", "audio/wav", "audio/x-wav", "audio/ogg", "audio/flac",
        "audio/mp4", "audio/aac", "audio/x-ms-wma", "audio/opus", "video/mp4", "video/webm",
    ])


class TimeoutLimits(BaseModel):
    """Processing timeouts."""

    download_seconds: int = 300
    processing_minutes: int = 30
    queue_wait_minutes: int = 60


class StorageLimits(BaseModel):
    """Temporary storage limits and cleanup."""

    max_temp_per_user_gb: float = 0.5
    cleanup_interval_minutes: int = 30
    orphan_file_age_hours: int = 24


class LimitsSettings(BaseModel):
    """Wrapper for limits loaded from YAML."""

    rate_limits: RateLimits = Field(default_factory=RateLimits)
    file_limits: FileLimits = Field(default_factory=FileLimits)
    timeouts: TimeoutLimits = Field(default_factory=TimeoutLimits)
    storage: StorageLimits = Field(default_factory=StorageLimits)

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "LimitsSettings":
        """Load limits from YAML file, falling back to defaults on error."""
        config_path = Path(path) if path else Path(__file__).parent / "limits.yaml"
        if config_path.exists():
            try:
                data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                return cls.model_validate(data)
            except Exception:
                pass
        return cls()


class Settings(BaseSettings):
    """Main settings class combining all configurations."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    telegram: TelegramSettings = Field(default_factory=lambda: TelegramSettings(_env_file=".env"))
    model: ModelSettings = Field(default_factory=lambda: ModelSettings(_env_file=".env"))
    processing: ProcessingSettings = Field(default_factory=lambda: ProcessingSettings(_env_file=".env"))
    memory: MemorySettings = Field(default_factory=lambda: MemorySettings(_env_file=".env"))
    paths: PathSettings = Field(default_factory=lambda: PathSettings(_env_file=".env"))
    dev: DevelopmentSettings = Field(default_factory=lambda: DevelopmentSettings(_env_file=".env"))
    limits: LimitsSettings = Field(default_factory=LimitsSettings.from_yaml)

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
        (self.paths.temp_dir / "downloads").mkdir(exist_ok=True)


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()
