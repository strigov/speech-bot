"""Telegram keyboard UI elements."""

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder, ReplyKeyboardBuilder


def _escape_markdown(text: str) -> str:
    """Escape special Markdown characters for Telegram."""
    # Escape special characters for Markdown v1
    chars_to_escape = ['_', '*', '`', '[']
    for char in chars_to_escape:
        text = text.replace(char, f'\\{char}')
    return text


def get_cancel_keyboard() -> InlineKeyboardMarkup:
    """Get keyboard with cancel button."""
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="❌ Cancel", callback_data="cancel_task"))
    return builder.as_markup()


def get_task_status_keyboard(task_id: str) -> InlineKeyboardMarkup:
    """Get keyboard for task status message."""
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="🔄 Refresh", callback_data=f"refresh_{task_id}"),
        InlineKeyboardButton(text="❌ Cancel", callback_data=f"cancel_{task_id}"),
    )
    return builder.as_markup()


def get_result_keyboard(task_id: str) -> InlineKeyboardMarkup:
    """Get keyboard for result message with download options."""
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="📄 TXT", callback_data=f"download_txt_{task_id}"),
        InlineKeyboardButton(text="📝 Plain", callback_data=f"download_plain_{task_id}"),
    )
    builder.row(
        InlineKeyboardButton(text="🎬 SRT", callback_data=f"download_srt_{task_id}"),
        InlineKeyboardButton(text="📊 JSON", callback_data=f"download_json_{task_id}"),
    )
    return builder.as_markup()


def get_confirm_keyboard(action: str) -> InlineKeyboardMarkup:
    """Get confirmation keyboard."""
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="✅ Yes", callback_data=f"confirm_{action}"),
        InlineKeyboardButton(text="❌ No", callback_data=f"deny_{action}"),
    )
    return builder.as_markup()


def get_admin_keyboard(public_mode: bool = False) -> InlineKeyboardMarkup:
    """Get admin panel keyboard."""
    mode_text = "🔓 Публичный" if public_mode else "🔒 Приватный"
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text=mode_text, callback_data="admin_mode"),
    )
    builder.row(
        InlineKeyboardButton(text="📊 GPU Status", callback_data="admin_gpu"),
        InlineKeyboardButton(text="📋 Queue", callback_data="admin_queue"),
    )
    builder.row(
        InlineKeyboardButton(text="🧹 Clear Stuck", callback_data="admin_clear"),
        InlineKeyboardButton(text="🔄 Restart Worker", callback_data="admin_restart"),
    )
    builder.row(
        InlineKeyboardButton(text="📈 Stats", callback_data="admin_stats"),
    )
    return builder.as_markup()


def get_main_reply_keyboard() -> ReplyKeyboardMarkup:
    """Get main reply keyboard with common actions."""
    builder = ReplyKeyboardBuilder()
    builder.row(
        KeyboardButton(text="📊 Status"),
        KeyboardButton(text="❓ Help"),
    )
    return builder.as_markup(resize_keyboard=True)


def get_processing_progress_text(
    status: str,
    progress_percent: float,
    current_step: str,
    message: str = "",
    queue_position: int = 0,
) -> str:
    """
    Generate progress display text.

    Args:
        status: Current status
        progress_percent: Progress percentage (0-100)
        current_step: Current processing step
        message: Optional message
        queue_position: Position in queue (0 if processing)

    Returns:
        Formatted progress text
    """
    # Progress bar
    filled = int(progress_percent / 10)
    empty = 10 - filled
    progress_bar = "█" * filled + "░" * empty

    lines = [
        f"🎵 **Processing Audio**",
        "",
        f"Status: {_get_status_emoji(status)} {status}",
        f"Progress: [{progress_bar}] {progress_percent:.0f}%",
    ]

    if current_step:
        lines.append(f"Step: {current_step}")

    if queue_position > 0:
        lines.append(f"Queue position: #{queue_position}")

    if message:
        lines.append(f"")
        lines.append(f"ℹ️ {message}")

    return "\n".join(lines)


def get_result_summary_text(
    duration_seconds: float,
    num_speakers: int,
    total_words: int,
    processing_time_seconds: float,
) -> str:
    """
    Generate result summary text.

    Args:
        duration_seconds: Audio duration
        num_speakers: Number of speakers detected
        total_words: Total word count
        processing_time_seconds: Processing time

    Returns:
        Formatted summary text
    """
    duration_str = _format_duration(duration_seconds)
    processing_str = _format_duration(processing_time_seconds)

    return (
        f"✅ **Transcription Complete**\n"
        f"\n"
        f"📊 **Summary:**\n"
        f"• Duration: {duration_str}\n"
        f"• Speakers: {num_speakers}\n"
        f"• Words: {total_words:,}\n"
        f"• Processing time: {processing_str}\n"
        f"\n"
        f"Choose a format to download:"
    )


def get_error_text(error_message: str) -> str:
    """Generate error display text with properly escaped Markdown."""
    # Escape special Markdown characters to prevent parsing errors
    escaped_error = _escape_markdown(error_message)
    return (
        f"❌ **Processing Failed**\n"
        f"\n"
        f"Error: {escaped_error}\n"
        f"\n"
        f"Please try again or contact support if the issue persists."
    )


def get_queue_full_text() -> str:
    """Get message for when queue is full."""
    return (
        "⚠️ **Queue Full**\n"
        "\n"
        "The processing queue is currently full. "
        "Please try again in a few minutes.\n"
        "\n"
        "You can check your status with /status"
    )


def get_rate_limit_text(wait_seconds: int) -> str:
    """Get message for rate limiting."""
    minutes = wait_seconds // 60
    seconds = wait_seconds % 60

    if minutes > 0:
        wait_str = f"{minutes}m {seconds}s"
    else:
        wait_str = f"{seconds}s"

    return (
        f"⏳ **Rate Limit**\n"
        f"\n"
        f"You've reached the maximum number of requests. "
        f"Please wait {wait_str} before submitting another file.\n"
        f"\n"
        f"Limits:\n"
        f"• Max 3 concurrent files\n"
        f"• Max 10 files per hour"
    )


def _get_status_emoji(status: str) -> str:
    """Get emoji for status."""
    status_emojis = {
        "pending": "⏳",
        "queued": "📋",
        "preprocessing": "🔄",
        "diarizing": "👥",
        "transcribing": "📝",
        "aggregating": "📊",
        "completed": "✅",
        "failed": "❌",
        "cancelled": "🚫",
        "timeout": "⏰",
    }
    return status_emojis.get(status.lower(), "❓")


def _format_duration(seconds: float) -> str:
    """Format duration for display."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
