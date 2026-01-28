"""Custom exception hierarchy for the speech-bot application."""


class SpeechBotError(Exception):
    """Base exception for all speech-bot errors."""

    def __init__(self, message: str, user_message: str | None = None):
        """
        Initialize exception.

        Args:
            message: Technical error message for logging
            user_message: User-friendly message to display (defaults to message)
        """
        super().__init__(message)
        self.user_message = user_message or message


# File-related exceptions
class FileError(SpeechBotError):
    """Base exception for file-related errors."""
    pass


class FileValidationError(FileError):
    """Raised when file validation fails."""
    pass


class FileTooLargeError(FileError):
    """Raised when file exceeds size limit."""

    def __init__(self, actual_size: int, max_size: int):
        self.actual_size = actual_size
        self.max_size = max_size
        actual_mb = actual_size / (1024 * 1024)
        max_mb = max_size / (1024 * 1024)
        super().__init__(
            f"File too large: {actual_mb:.1f}MB (max: {max_mb:.0f}MB)",
            f"File too large: {actual_mb:.1f}MB (max: {max_mb:.0f}MB)"
        )


class FileDownloadError(FileError):
    """Raised when file download fails."""
    pass


class UnsupportedFormatError(FileError):
    """Raised when file format is not supported."""
    pass


# Processing exceptions
class ProcessingError(SpeechBotError):
    """Base exception for processing errors."""
    pass


class PreprocessingError(ProcessingError):
    """Raised when audio preprocessing fails."""
    pass


class DiarizationError(ProcessingError):
    """Raised when speaker diarization fails."""
    pass


class TranscriptionError(ProcessingError):
    """Raised when transcription fails."""
    pass


class AggregationError(ProcessingError):
    """Raised when result aggregation fails."""
    pass


class ProcessingTimeoutError(ProcessingError):
    """Raised when processing exceeds timeout."""

    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Processing exceeded {timeout_seconds}s timeout",
            "Processing took too long and was cancelled. Try a shorter audio file."
        )


# Queue exceptions
class QueueError(SpeechBotError):
    """Base exception for queue errors."""
    pass


class QueueFullError(QueueError):
    """Raised when queue is full."""

    def __init__(self):
        super().__init__(
            "Queue is full",
            "The queue is currently full. Please try again later."
        )


class TaskNotFoundError(QueueError):
    """Raised when task is not found."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        super().__init__(
            f"Task not found: {task_id}",
            "Task not found or has expired."
        )


class TaskCancelledError(QueueError):
    """Raised when task is cancelled."""
    pass


# Rate limiting exceptions
class RateLimitError(SpeechBotError):
    """Base exception for rate limiting errors."""
    pass


class CooldownError(RateLimitError):
    """Raised when user is in cooldown period."""

    def __init__(self, remaining_seconds: int):
        self.remaining_seconds = remaining_seconds
        super().__init__(
            f"Cooldown active: {remaining_seconds}s remaining",
            f"Please wait {remaining_seconds}s before sending another file."
        )


class HourlyLimitError(RateLimitError):
    """Raised when user exceeds hourly limit."""

    def __init__(self, limit: int):
        self.limit = limit
        super().__init__(
            f"Hourly limit reached: {limit} files/hour",
            f"Hourly limit reached ({limit} files/hour). Please try again later."
        )


class ConcurrentLimitError(RateLimitError):
    """Raised when user exceeds concurrent task limit."""

    def __init__(self, limit: int):
        self.limit = limit
        super().__init__(
            f"Concurrent limit reached: {limit} tasks",
            f"You already have {limit} tasks in progress. Please wait for them to complete."
        )


# Access control exceptions
class AccessError(SpeechBotError):
    """Base exception for access control errors."""
    pass


class UnauthorizedChatError(AccessError):
    """Raised when chat is not authorized."""

    def __init__(self, chat_id: int):
        self.chat_id = chat_id
        super().__init__(
            f"Unauthorized chat: {chat_id}",
            "This bot is restricted to authorized users only. Please contact the administrator."
        )


class AdminRequiredError(AccessError):
    """Raised when admin access is required."""

    def __init__(self):
        super().__init__(
            "Admin access required",
            "This command requires admin privileges."
        )


# Service exceptions
class ServiceError(SpeechBotError):
    """Base exception for service errors."""
    pass


class ServiceUnavailableError(ServiceError):
    """Raised when service is unavailable."""

    def __init__(self, service_name: str = "service"):
        self.service_name = service_name
        super().__init__(
            f"{service_name} is unavailable",
            "Service is starting up. Please try again in a moment."
        )


class GPUError(ServiceError):
    """Raised when GPU-related error occurs."""
    pass


class OutOfMemoryError(GPUError):
    """Raised when GPU runs out of memory."""

    def __init__(self):
        super().__init__(
            "GPU out of memory",
            "Server is currently overloaded. Please try again later."
        )
