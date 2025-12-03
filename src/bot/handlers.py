"""Telegram bot message handlers."""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, Set

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, FSInputFile, Message
import structlog

from config.settings import Settings
from src.bot.filters import AudioFileFilter, AdminFilter, AllowedChatFilter, get_file_info_from_message
from src.utils.telegram_client import get_telegram_client, init_telegram_client, should_use_client_api
from src.bot.keyboards import (
    get_admin_keyboard,
    get_cancel_keyboard,
    get_error_text,
    get_processing_progress_text,
    get_queue_full_text,
    get_rate_limit_text,
    get_result_keyboard,
    get_result_summary_text,
    get_task_status_keyboard,
)
from src.utils.rate_limiter import RateLimiter, RateLimitConfig
from src.utils.validation import FileValidator
from src.worker import (
    ProcessingTask,
    TaskProgress,
    TaskQueue,
    TaskStatus,
    Worker,
    create_task,
)

logger = structlog.get_logger(__name__)

# Create routers
router = Router()
admin_router = Router()


class BotState:
    """Shared state for bot handlers."""

    def __init__(self):
        self.worker: Optional[Worker] = None
        self.queue: Optional[TaskQueue] = None
        self.bot: Optional[Bot] = None
        self.settings: Optional[Settings] = None
        self.rate_limiter: Optional[RateLimiter] = None
        self.admin_ids: Set[int] = set()
        self.allowed_chat_ids: Set[int] = set()
        self._progress_messages: Dict[str, int] = {}  # task_id -> message_id
        self._user_progress_chats: Dict[str, int] = {}  # task_id -> chat_id

    def set_worker(self, worker: Worker) -> None:
        """Set the worker instance."""
        self.worker = worker
        self.queue = worker.queue

    def set_bot(self, bot: Bot) -> None:
        """Set the bot instance."""
        self.bot = bot

    def set_settings(self, settings: Settings) -> None:
        """Set settings and initialize rate limiter."""
        self.settings = settings
        if settings:
            # Load access control lists
            self.admin_ids = set(settings.telegram.admin_user_ids)
            self.allowed_chat_ids = set(settings.telegram.allowed_chat_ids)

            # Initialize rate limiter with admin bypass
            config = RateLimitConfig(
                max_concurrent=settings.limits.rate_limits.per_user.max_concurrent,
                max_per_hour=settings.limits.rate_limits.per_user.max_per_hour,
                cooldown_seconds=settings.limits.rate_limits.per_user.cooldown_seconds,
            )
            self.rate_limiter = RateLimiter(config, admin_ids=self.admin_ids)

    def add_admin(self, user_id: int) -> None:
        """Add an admin user."""
        self.admin_ids.add(user_id)


# Global state instance
state = BotState()


# Progress callback for worker
async def progress_callback(task_id: str, progress: TaskProgress) -> None:
    """Handle progress updates from worker."""
    if not state.bot:
        return

    chat_id = state._user_progress_chats.get(task_id)
    message_id = state._progress_messages.get(task_id)

    if not chat_id:
        return

    try:
        # Get queue position if queued
        queue_position = 0
        if progress.status == TaskStatus.QUEUED and state.queue:
            queue_position = state.queue.size

        text = get_processing_progress_text(
            status=progress.status.value,
            progress_percent=progress.progress_percent,
            current_step=progress.current_step,
            message=progress.message,
            queue_position=queue_position,
        )

        if progress.status == TaskStatus.COMPLETED:
            # Send completion message
            task = state.queue.get_task(task_id) if state.queue else None
            if task and task.result:
                text = get_result_summary_text(
                    duration_seconds=task.result.total_duration_seconds,
                    num_speakers=task.result.num_speakers,
                    total_words=task.result.total_words,
                    processing_time_seconds=task.result.processing_time_seconds,
                )
                keyboard = get_result_keyboard(task_id)

                if message_id:
                    await state.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=text,
                        reply_markup=keyboard,
                        parse_mode="Markdown",
                    )
                else:
                    await state.bot.send_message(
                        chat_id=chat_id,
                        text=text,
                        reply_markup=keyboard,
                        parse_mode="Markdown",
                    )

                # Send transcript file automatically
                if "transcript" in task.output_files:
                    await state.bot.send_document(
                        chat_id=chat_id,
                        document=FSInputFile(task.output_files["transcript"]),
                        caption="📄 Formatted transcript with timestamps",
                    )

        elif progress.status in (TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.CANCELLED):
            # Send error message
            task = state.queue.get_task(task_id) if state.queue else None
            error_msg = task.error if task else progress.message
            text = get_error_text(error_msg or "Unknown error")

            if message_id:
                await state.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=text,
                    parse_mode="Markdown",
                )
            else:
                await state.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode="Markdown",
                )

        else:
            # Update progress message
            keyboard = get_task_status_keyboard(task_id)

            if message_id:
                try:
                    await state.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=text,
                        reply_markup=keyboard,
                        parse_mode="Markdown",
                    )
                except Exception:
                    pass  # Message might not have changed

    except Exception as e:
        logger.error("progress_callback_error", task_id=task_id, error=str(e))


# Command handlers
@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    """Handle /start command."""
    await message.answer(
        "🎵 **Audio Transcriber Bot**\n"
        "\n"
        "I transcribe audio files with speaker diarization using "
        "GigaAM v3 ASR and PyAnnote.\n"
        "\n"
        "**Features:**\n"
        "• Multi-speaker detection\n"
        "• Timestamps for each segment\n"
        "• Support for long audio (up to 3 hours)\n"
        "• Multiple output formats (TXT, SRT, JSON)\n"
        "\n"
        "Simply send me an audio file to get started!\n"
        "\n"
        "**Commands:**\n"
        "/help - Detailed help\n"
        "/status - Check your tasks\n"
        "/cancel - Cancel current task",
        parse_mode="Markdown",
    )
    logger.info("user_started_bot", user_id=message.from_user.id)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    """Handle /help command."""
    await message.answer(
        "📖 **Audio Transcriber - Help**\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "\n"
        "**Supported formats:**\n"
        "MP3, WAV, OGG, FLAC, M4A, AAC, WMA, OPUS, WebM, MP4\n"
        "\n"
        "**Limits:**\n"
        "• Max file size: 500 MB\n"
        "• Max duration: 3 hours\n"
        "• Max concurrent: 3 files\n"
        "• Max per hour: 10 files\n"
        "\n"
        "**Output format:**\n"
        "`[HH:MM:SS - HH:MM:SS] SPEAKER_XX: Text`\n"
        "\n"
        "**Processing time:**\n"
        "• < 5 min audio: ~1 min\n"
        "• 5-30 min audio: ~5 min\n"
        "• 60 min audio: ~15 min\n"
        "\n"
        "**Commands:**\n"
        "/start - Welcome message\n"
        "/help - This message\n"
        "/status - Your task status\n"
        "/cancel - Cancel current task",
        parse_mode="Markdown",
    )


@router.message(Command("status"))
async def cmd_status(message: Message) -> None:
    """Handle /status command."""
    user_id = message.from_user.id

    if not state.queue:
        await message.answer("⚠️ Service is starting up. Please try again.")
        return

    user_tasks = state.queue.get_user_tasks(user_id)

    if not user_tasks:
        await message.answer(
            "📋 **Your Tasks**\n"
            "\n"
            "No active tasks.\n"
            "\n"
            "Send an audio file to start transcription!",
            parse_mode="Markdown",
        )
        return

    lines = ["📋 **Your Tasks**\n"]

    for task in user_tasks:
        status_emoji = _get_status_emoji(task.progress.status)
        status_text = task.progress.status.value.title()

        lines.append(f"\n{status_emoji} **{task.original_filename}**")
        lines.append(f"Status: {status_text}")

        if task.progress.progress_percent > 0:
            lines.append(f"Progress: {task.progress.progress_percent:.0f}%")

        if task.progress.current_step:
            lines.append(f"Step: {task.progress.current_step}")

    await message.answer("\n".join(lines), parse_mode="Markdown")


@router.message(Command("cancel"))
async def cmd_cancel(message: Message) -> None:
    """Handle /cancel command."""
    user_id = message.from_user.id

    if not state.queue:
        await message.answer("⚠️ Service is starting up.")
        return

    user_tasks = state.queue.get_user_tasks(user_id)
    pending_tasks = [
        t for t in user_tasks
        if t.progress.status in (TaskStatus.PENDING, TaskStatus.QUEUED)
    ]

    if not pending_tasks:
        await message.answer(
            "No pending tasks to cancel.\n"
            "Note: Tasks already being processed cannot be cancelled."
        )
        return

    # Cancel all pending tasks
    cancelled = 0
    for task in pending_tasks:
        if await state.queue.cancel_task(task.task_id):
            cancelled += 1

    await message.answer(f"✅ Cancelled {cancelled} pending task(s).")
    logger.info("tasks_cancelled", user_id=user_id, count=cancelled)


# Audio file handler
@router.message(F.audio | F.voice | F.video_note | F.document)
async def handle_audio(message: Message) -> None:
    """Handle incoming audio files."""
    user_id = message.from_user.id

    # Check if chat is allowed
    if state.allowed_chat_ids and message.chat.id not in state.allowed_chat_ids:
        logger.warning(
            "unauthorized_chat_access",
            user_id=user_id,
            chat_id=message.chat.id,
            username=message.from_user.username if message.from_user else None,
        )
        await message.answer(
            "⛔ This bot is restricted to authorized users only.\n"
            "Please contact the bot administrator if you need access."
        )
        return

    # Extract file info
    file_info = get_file_info_from_message(message)
    if not file_info:
        await message.answer(
            "⚠️ Please send an audio or video file.\n"
            "Supported formats: MP3, WAV, OGG, FLAC, M4A, MP4, WebM, etc."
        )
        return

    # Check queue availability
    if not state.queue or not state.worker:
        await message.answer("⚠️ Service is starting up. Please try again in a moment.")
        return

    # Check if queue is full
    if state.queue.is_full:
        await message.answer(get_queue_full_text(), parse_mode="Markdown")
        return

    # Check concurrent limits (skip for admins)
    is_admin = user_id in state.admin_ids
    if not is_admin and not state.queue.can_user_submit(user_id):
        await message.answer(get_rate_limit_text(300), parse_mode="Markdown")
        return

    # Check rate limits (hourly/cooldown) - admins bypass this in RateLimiter
    if state.rate_limiter:
        is_allowed, info = state.rate_limiter.check_user(user_id)
        if not is_allowed:
            await message.answer(f"⚠️ {info.get('message', 'Rate limit exceeded.')}")
            return

    # Validate file size
    max_size = state.settings.processing.max_file_size_mb * 1024 * 1024 if state.settings else 500 * 1024 * 1024
    if file_info["file_size"] > max_size:
        max_mb = max_size / (1024 * 1024)
        actual_mb = file_info["file_size"] / (1024 * 1024)
        await message.answer(
            f"⚠️ File too large: {actual_mb:.1f}MB (max: {max_mb:.0f}MB)"
        )
        return

    # Send initial status message
    status_msg = await message.answer(
        "📥 **Downloading file...**\n"
        "\n"
        "Please wait while I download your audio file.",
        parse_mode="Markdown",
    )

    try:
        # Prepare download path
        file_path = Path(state.settings.paths.temp_dir if state.settings else "./temp") / "downloads" / str(user_id)
        file_path.mkdir(parents=True, exist_ok=True)

        raw_name = file_info["file_name"]
        safe_name = "".join(c for c in Path(raw_name).name if c.isalnum() or c in "._-")
        if not safe_name:
            safe_name = "file"

        local_path = file_path / safe_name

        # Check if we need to use Client API for large files
        use_client_api = await should_use_client_api(file_info["file_size"])

        if use_client_api:
            # Use Telegram Client API for large files (> 20 MB)
            client = get_telegram_client()
            if client and client.is_started():
                logger.info(
                    "using_client_api_for_large_file",
                    user_id=user_id,
                    file_size=file_info["file_size"],
                    file_name=file_info["file_name"],
                )

                success = await client.download_file(
                    chat_id=message.chat.id,
                    message_id=message.message_id,
                    destination=local_path,
                )

                if not success:
                    await status_msg.edit_text(
                        "⚠️ Failed to download file using Client API.\n"
                        "Please ensure the bot has proper API credentials configured."
                    )
                    return
            else:
                await status_msg.edit_text(
                    "⚠️ File is too large (> 20 MB) but Client API is not configured.\n"
                    "Please contact the bot administrator to enable large file support."
                )
                return
        else:
            # Use standard Bot API for small files (<= 20 MB)
            file = await message.bot.get_file(file_info["file_id"])
            await message.bot.download_file(file.file_path, local_path)

        logger.info(
            "file_downloaded",
            user_id=user_id,
            file_name=file_info["file_name"],
            file_size=file_info["file_size"],
            downloaded_to=str(local_path),
            file_exists_after_download=local_path.exists(),
        )

        # Validate file (magic bytes)
        is_valid, error_msg = FileValidator.validate_file(local_path)
        if not is_valid:
            # Delete invalid file
            try:
                local_path.unlink()
            except Exception:
                pass
            
            await status_msg.edit_text(
                f"⚠️ Invalid file format: {error_msg}\n"
                "Please ensure you are sending a valid audio/video file."
            )
            return

        # Record request for rate limiting
        if state.rate_limiter:
            state.rate_limiter.record_request(user_id)

        # Create task
        task = create_task(
            user_id=user_id,
            input_path=local_path,
            original_filename=file_info["file_name"],
        )

        logger.info(
            "task_created",
            task_id=task.task_id,
            input_path=str(task.input_path),
            input_path_exists=task.input_path.exists(),
        )

        # Track progress message
        state._progress_messages[task.task_id] = status_msg.message_id
        state._user_progress_chats[task.task_id] = message.chat.id

        # Submit to queue
        if not await state.queue.submit(task):
            await status_msg.edit_text(
                get_queue_full_text(),
                parse_mode="Markdown",
            )
            return

        # Update status message
        await status_msg.edit_text(
            get_processing_progress_text(
                status="queued",
                progress_percent=0,
                current_step="Waiting in queue",
                message=f"Position: #{state.queue.size}",
                queue_position=state.queue.size,
            ),
            reply_markup=get_task_status_keyboard(task.task_id),
            parse_mode="Markdown",
        )

        logger.info(
            "task_submitted",
            user_id=user_id,
            task_id=task.task_id,
            file_name=file_info["file_name"],
        )

    except Exception as e:
        logger.error("file_processing_error", user_id=user_id, error=str(e))
        await status_msg.edit_text(
            get_error_text(f"Failed to process file: {str(e)}"),
            parse_mode="Markdown",
        )


# Callback handlers
@router.callback_query(F.data.startswith("cancel_"))
async def callback_cancel(callback: CallbackQuery) -> None:
    """Handle cancel button callback."""
    task_id = callback.data.replace("cancel_", "").replace("task", "").strip("_")

    if not task_id or task_id == "task":
        # Generic cancel - cancel all user's pending tasks
        await cmd_cancel(callback.message)
        await callback.answer("Cancelled")
        return

    if state.queue:
        success = await state.queue.cancel_task(task_id)
        if success:
            await callback.answer("Task cancelled")
            await callback.message.edit_text(
                "🚫 **Task Cancelled**\n\nThe task has been cancelled.",
                parse_mode="Markdown",
            )
        else:
            await callback.answer("Cannot cancel - task already processing")
    else:
        await callback.answer("Service unavailable")


@router.callback_query(F.data.startswith("refresh_"))
async def callback_refresh(callback: CallbackQuery) -> None:
    """Handle refresh button callback."""
    task_id = callback.data.replace("refresh_", "")

    if not state.queue:
        await callback.answer("Service unavailable")
        return

    task = state.queue.get_task(task_id)
    if not task:
        await callback.answer("Task not found")
        return

    text = get_processing_progress_text(
        status=task.progress.status.value,
        progress_percent=task.progress.progress_percent,
        current_step=task.progress.current_step,
        message=task.progress.message,
    )

    try:
        await callback.message.edit_text(
            text,
            reply_markup=get_task_status_keyboard(task_id),
            parse_mode="Markdown",
        )
    except Exception:
        pass

    await callback.answer("Refreshed")


@router.callback_query(F.data.startswith("download_"))
async def callback_download(callback: CallbackQuery) -> None:
    """Handle download button callback."""
    parts = callback.data.split("_")
    if len(parts) < 3:
        await callback.answer("Invalid request")
        return

    format_type = parts[1]  # txt, plain, srt, json
    task_id = parts[2]

    if not state.queue:
        await callback.answer("Service unavailable")
        return

    task = state.queue.get_task(task_id)
    if not task:
        await callback.answer("Task not found")
        return

    # Map format to file key
    format_map = {
        "txt": "transcript",
        "plain": "plain",
        "srt": "srt",
        "json": "json",
    }

    file_key = format_map.get(format_type)
    if not file_key or file_key not in task.output_files:
        await callback.answer("File not available")
        return

    file_path = task.output_files[file_key]
    if not file_path.exists():
        await callback.answer("File not found")
        return

    captions = {
        "transcript": "📄 Formatted transcript with timestamps",
        "plain": "📝 Plain text transcript",
        "srt": "🎬 SRT subtitles",
        "json": "📊 JSON with full metadata",
    }

    await callback.message.answer_document(
        document=FSInputFile(file_path),
        caption=captions.get(file_key, "Transcript"),
    )
    await callback.answer("File sent!")


# Admin handlers
@admin_router.message(Command("admin"))
async def cmd_admin(message: Message, command: CommandObject) -> None:
    """Handle /admin command."""
    # Check if user is admin
    if not state.admin_ids or message.from_user.id not in state.admin_ids:
        logger.warning(
            "unauthorized_admin_access",
            user_id=message.from_user.id,
            username=message.from_user.username if message.from_user else None,
        )
        await message.answer("⛔ Admin access required.")
        return

    if not command.args:
        await message.answer(
            "🔧 **Admin Panel**\n"
            "\n"
            "Choose an action:",
            reply_markup=get_admin_keyboard(),
            parse_mode="Markdown",
        )
        return

    # Handle subcommands
    subcommand = command.args.lower()

    if subcommand == "gpu":
        await admin_gpu_status(message)
    elif subcommand == "queue":
        await admin_queue_status(message)
    elif subcommand == "clear":
        await admin_clear_stuck(message)
    elif subcommand == "stats":
        await admin_stats(message)
    else:
        await message.answer(f"Unknown admin command: {subcommand}")


async def admin_gpu_status(message: Message) -> None:
    """Show GPU status."""
    if not state.worker:
        await message.answer("Worker not available")
        return

    stats = state.worker.gpu_monitor.get_stats()

    if not stats.is_available:
        await message.answer("⚠️ GPU not available")
        return

    await message.answer(
        f"🖥️ **GPU Status**\n"
        f"\n"
        f"Device: {stats.device_name}\n"
        f"CUDA: {stats.cuda_version}\n"
        f"\n"
        f"Memory:\n"
        f"• Total: {stats.total_memory_gb:.1f} GB\n"
        f"• Allocated: {stats.allocated_memory_gb:.2f} GB\n"
        f"• Free: {stats.free_memory_gb:.2f} GB\n"
        f"• Utilization: {stats.utilization_percent:.1f}%",
        parse_mode="Markdown",
    )


async def admin_queue_status(message: Message) -> None:
    """Show queue status."""
    if not state.queue:
        await message.answer("Queue not available")
        return

    stats = state.queue.get_queue_stats()

    status_lines = []
    for status, count in stats["tasks_by_status"].items():
        status_lines.append(f"• {status}: {count}")

    await message.answer(
        f"📋 **Queue Status**\n"
        f"\n"
        f"Queue size: {stats['queue_size']}/{stats['max_size']}\n"
        f"Total tasks: {stats['total_tasks']}\n"
        f"Active users: {stats['active_users']}\n"
        f"\n"
        f"**By status:**\n" + "\n".join(status_lines),
        parse_mode="Markdown",
    )


async def admin_clear_stuck(message: Message) -> None:
    """Clear stuck tasks."""
    await message.answer("🧹 Clearing stuck tasks... (not implemented yet)")


async def admin_stats(message: Message) -> None:
    """Show overall stats."""
    if not state.worker:
        await message.answer("Worker not available")
        return

    status = state.worker.get_status()

    await message.answer(
        f"📈 **System Stats**\n"
        f"\n"
        f"Worker running: {'✅' if status['is_running'] else '❌'}\n"
        f"Current task: {status['current_task'] or 'None'}\n"
        f"\n"
        f"Queue: {status['queue_stats']['queue_size']}/{status['queue_stats']['max_size']}",
        parse_mode="Markdown",
    )


@admin_router.callback_query(F.data.startswith("admin_"))
async def callback_admin(callback: CallbackQuery) -> None:
    """Handle admin callback buttons."""
    # Check if user is admin
    if not state.admin_ids or callback.from_user.id not in state.admin_ids:
        await callback.answer("⛔ Admin access required")
        return

    action = callback.data.replace("admin_", "")

    if action == "gpu":
        await admin_gpu_status(callback.message)
    elif action == "queue":
        await admin_queue_status(callback.message)
    elif action == "clear":
        await admin_clear_stuck(callback.message)
    elif action == "stats":
        await admin_stats(callback.message)
    elif action == "restart":
        await callback.message.answer("🔄 Restarting worker... (not implemented yet)")

    await callback.answer()


def _get_status_emoji(status: TaskStatus) -> str:
    """Get emoji for task status."""
    emojis = {
        TaskStatus.PENDING: "⏳",
        TaskStatus.QUEUED: "📋",
        TaskStatus.PREPROCESSING: "🔄",
        TaskStatus.DIARIZING: "👥",
        TaskStatus.TRANSCRIBING: "📝",
        TaskStatus.AGGREGATING: "📊",
        TaskStatus.COMPLETED: "✅",
        TaskStatus.FAILED: "❌",
        TaskStatus.CANCELLED: "🚫",
        TaskStatus.TIMEOUT: "⏰",
    }
    return emojis.get(status, "❓")


def create_bot(settings: Settings) -> tuple[Bot, Dispatcher]:
    """
    Create and configure the bot instance.

    Args:
        settings: Application settings

    Returns:
        Tuple of (Bot, Dispatcher)
    """
    bot = Bot(token=settings.telegram.bot_token)
    dp = Dispatcher()

    # Include routers
    dp.include_router(router)
    dp.include_router(admin_router)

    # Store references
    state.set_bot(bot)
    state.set_settings(settings)

    # Initialize Telegram Client API for large files if configured
    if settings.telegram.use_client_api and settings.telegram.api_id and settings.telegram.api_hash:
        try:
            init_telegram_client(
                api_id=settings.telegram.api_id,
                api_hash=settings.telegram.api_hash,
                bot_token=settings.telegram.bot_token,
                session_name=settings.telegram.session_name,
                session_dir=settings.paths.temp_dir / "sessions",
            )
            logger.info("telegram_client_api_initialized", supports_large_files=True)
        except Exception as e:
            logger.warning("telegram_client_api_init_failed", error=str(e))
    else:
        logger.info("telegram_client_api_not_configured", supports_large_files=False)

    logger.info("bot_created")
    return bot, dp


async def run_bot(bot: Bot, dp: Dispatcher) -> None:
    """
    Run the bot with polling.

    Args:
        bot: Bot instance
        dp: Dispatcher instance
    """
    logger.info("starting_bot_polling")
    await dp.start_polling(bot)


def setup_worker(worker: Worker) -> None:
    """
    Set up worker for bot handlers.

    Args:
        worker: Worker instance
    """
    state.set_worker(worker)

    # Set progress callback
    worker.progress_callback = lambda tid, prog: asyncio.create_task(
        progress_callback(tid, prog)
    )

    logger.info("worker_connected_to_bot")


def add_admin(user_id: int) -> None:
    """Add an admin user ID."""
    state.add_admin(user_id)
