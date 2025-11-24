"""File management utilities for temporary files and cleanup."""

import asyncio
import shutil
import time
from pathlib import Path
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)


class FileManager:
    """Manages temporary files and directories for audio processing."""

    def __init__(
        self,
        temp_dir: Path,
        max_per_user_gb: float = 5.0,
        cleanup_interval_seconds: int = 300,
        file_ttl_seconds: int = 3600,
    ):
        """
        Initialize file manager.

        Args:
            temp_dir: Base temporary directory
            max_per_user_gb: Maximum storage per user in GB
            cleanup_interval_seconds: Interval between cleanup runs
            file_ttl_seconds: Time-to-live for temp files in seconds
        """
        self.temp_dir = Path(temp_dir)
        self.max_per_user_bytes = int(max_per_user_gb * 1024 * 1024 * 1024)
        self.cleanup_interval = cleanup_interval_seconds
        self.file_ttl = file_ttl_seconds
        self._cleanup_task: Optional[asyncio.Task] = None

        # Ensure base directories exist
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        (self.temp_dir / "audio").mkdir(exist_ok=True)
        (self.temp_dir / "results").mkdir(exist_ok=True)

    def get_user_dir(self, user_id: int) -> Path:
        """Get or create user-specific temporary directory."""
        user_dir = self.temp_dir / "audio" / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    def get_task_dir(self, user_id: int, task_id: str) -> Path:
        """Get or create task-specific directory within user's space."""
        task_dir = self.get_user_dir(user_id) / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    def get_results_dir(self, user_id: int) -> Path:
        """Get or create user's results directory."""
        results_dir = self.temp_dir / "results" / str(user_id)
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    def get_user_storage_bytes(self, user_id: int) -> int:
        """Calculate total storage used by a user."""
        user_audio_dir = self.temp_dir / "audio" / str(user_id)
        user_results_dir = self.temp_dir / "results" / str(user_id)

        total = 0
        for directory in [user_audio_dir, user_results_dir]:
            if directory.exists():
                for file_path in directory.rglob("*"):
                    if file_path.is_file():
                        try:
                            total += file_path.stat().st_size
                        except OSError:
                            pass
        return total

    def check_user_quota(self, user_id: int, additional_bytes: int = 0) -> bool:
        """Check if user has storage quota available."""
        current = self.get_user_storage_bytes(user_id)
        return (current + additional_bytes) <= self.max_per_user_bytes

    def cleanup_task_files(self, user_id: int, task_id: str) -> None:
        """Clean up all files for a specific task."""
        task_dir = self.temp_dir / "audio" / str(user_id) / task_id
        if task_dir.exists():
            try:
                shutil.rmtree(task_dir)
                logger.info("task_files_cleaned", user_id=user_id, task_id=task_id)
            except Exception as e:
                logger.error(
                    "task_cleanup_failed",
                    user_id=user_id,
                    task_id=task_id,
                    error=str(e)
                )

    def cleanup_user_files(self, user_id: int) -> None:
        """Clean up all files for a user."""
        for subdir in ["audio", "results"]:
            user_dir = self.temp_dir / subdir / str(user_id)
            if user_dir.exists():
                try:
                    shutil.rmtree(user_dir)
                    logger.info("user_files_cleaned", user_id=user_id, subdir=subdir)
                except Exception as e:
                    logger.error(
                        "user_cleanup_failed",
                        user_id=user_id,
                        subdir=subdir,
                        error=str(e)
                    )

    def cleanup_old_files(self) -> int:
        """
        Remove files older than TTL.

        Returns:
            Number of files removed
        """
        removed_count = 0
        cutoff_time = time.time() - self.file_ttl

        for subdir in ["audio", "results"]:
            scan_dir = self.temp_dir / subdir
            if not scan_dir.exists():
                continue

            for file_path in scan_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                try:
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        removed_count += 1
                except OSError:
                    pass

        # Clean up empty directories
        self._cleanup_empty_dirs()

        if removed_count > 0:
            logger.info("old_files_cleaned", count=removed_count)

        return removed_count

    def _cleanup_empty_dirs(self) -> None:
        """Remove empty directories."""
        for subdir in ["audio", "results"]:
            scan_dir = self.temp_dir / subdir
            if not scan_dir.exists():
                continue

            # Walk bottom-up to remove empty dirs
            for dir_path in sorted(scan_dir.rglob("*"), reverse=True):
                if dir_path.is_dir():
                    try:
                        if not any(dir_path.iterdir()):
                            dir_path.rmdir()
                    except OSError:
                        pass

    async def start_cleanup_loop(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is not None:
            return

        async def _cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval)
                    # Run cleanup in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self.cleanup_old_files)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("cleanup_loop_error", error=str(e))

        self._cleanup_task = asyncio.create_task(_cleanup_loop())
        logger.info("cleanup_loop_started", interval=self.cleanup_interval)

    async def stop_cleanup_loop(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("cleanup_loop_stopped")

    def get_safe_path(self, user_id: int, filename: str) -> Path:
        """
        Get a safe file path that prevents path traversal attacks.

        Args:
            user_id: User ID for isolation
            filename: Original filename

        Returns:
            Safe path within user's directory
        """
        # Sanitize filename - remove path separators and dangerous characters
        safe_name = Path(filename).name
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in "._-")

        if not safe_name:
            safe_name = "file"

        return self.get_user_dir(user_id) / safe_name

    def get_temp_storage_stats(self) -> dict:
        """Get storage statistics for temp directory."""
        stats = {
            "total_bytes": 0,
            "file_count": 0,
            "user_count": 0,
            "users": {}
        }

        audio_dir = self.temp_dir / "audio"
        if audio_dir.exists():
            user_dirs = [d for d in audio_dir.iterdir() if d.is_dir()]
            stats["user_count"] = len(user_dirs)

            for user_dir in user_dirs:
                user_id = user_dir.name
                user_bytes = 0
                user_files = 0

                for file_path in user_dir.rglob("*"):
                    if file_path.is_file():
                        try:
                            user_bytes += file_path.stat().st_size
                            user_files += 1
                        except OSError:
                            pass

                stats["users"][user_id] = {
                    "bytes": user_bytes,
                    "files": user_files
                }
                stats["total_bytes"] += user_bytes
                stats["file_count"] += user_files

        return stats
