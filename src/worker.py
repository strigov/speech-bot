"""Worker module for processing audio transcription tasks."""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import structlog

from src.pipeline.aggregator import AggregationResult, OutputGenerator, ResultAggregator
from src.pipeline.diarizer import PyAnnoteDiarizer, get_diarizer
from src.pipeline.preprocessor import AudioPreprocessor, PreprocessingResult
from src.pipeline.transcriber import GigaAMTranscriber, get_transcriber
from src.utils.file_manager import FileManager
from src.utils.gpu_monitor import GPUMonitor, get_gpu_monitor

logger = structlog.get_logger(__name__)


class TaskStatus(Enum):
    """Task status enumeration."""

    PENDING = "pending"
    QUEUED = "queued"
    PREPROCESSING = "preprocessing"
    DIARIZING = "diarizing"
    TRANSCRIBING = "transcribing"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class TaskProgress:
    """Progress information for a task."""

    status: TaskStatus
    progress_percent: float = 0.0
    current_step: str = ""
    steps_completed: int = 0
    total_steps: int = 4  # preprocess, diarize, transcribe, aggregate
    message: str = ""
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ProcessingTask:
    """A task in the processing queue."""

    task_id: str
    user_id: int
    input_path: Path
    original_filename: str
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: TaskProgress = field(default_factory=lambda: TaskProgress(TaskStatus.PENDING))
    result: Optional[AggregationResult] = None
    output_files: Dict[str, Path] = field(default_factory=dict)
    error: Optional[str] = None
    checkpoint_path: Optional[Path] = None

    @property
    def is_complete(self) -> bool:
        """Check if task is in a terminal state."""
        return self.progress.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMEOUT,
        )

    @property
    def queue_time_seconds(self) -> float:
        """Get time spent in queue."""
        if self.started_at:
            return (self.started_at - self.created_at).total_seconds()
        return (datetime.now() - self.created_at).total_seconds()

    @property
    def processing_time_seconds(self) -> float:
        """Get time spent processing."""
        if not self.started_at:
            return 0.0
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()


@dataclass
class Checkpoint:
    """Checkpoint for resuming long-running tasks."""

    task_id: str
    user_id: int
    stage: str
    progress_data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)

    def save(self, checkpoint_dir: Path) -> Path:
        """Save checkpoint to file."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"{self.task_id}.json"

        data = {
            "task_id": self.task_id,
            "user_id": self.user_id,
            "stage": self.stage,
            "progress_data": self.progress_data,
            "created_at": self.created_at.isoformat(),
        }

        path.write_text(json.dumps(data, default=str), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> Optional["Checkpoint"]:
        """Load checkpoint from file."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                task_id=data["task_id"],
                user_id=data["user_id"],
                stage=data["stage"],
                progress_data=data["progress_data"],
                created_at=datetime.fromisoformat(data["created_at"]),
            )
        except Exception as e:
            logger.error("checkpoint_load_failed", path=str(path), error=str(e))
            return None


# Type alias for progress callback
ProgressCallback = Callable[[str, TaskProgress], None]


class TaskQueue:
    """Async task queue with size limits and user tracking."""

    def __init__(self, max_size: int = 50, max_per_user: int = 3):
        """
        Initialize task queue.

        Args:
            max_size: Maximum queue size
            max_per_user: Maximum concurrent tasks per user
        """
        self.max_size = max_size
        self.max_per_user = max_per_user
        self._queue: asyncio.Queue[ProcessingTask] = asyncio.Queue(maxsize=max_size)
        self._tasks: Dict[str, ProcessingTask] = {}
        self._user_tasks: Dict[int, List[str]] = {}
        self._lock = asyncio.Lock()

    @property
    def size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    @property
    def is_full(self) -> bool:
        """Check if queue is full."""
        return self._queue.full()

    def get_user_task_count(self, user_id: int) -> int:
        """Get number of active tasks for a user."""
        return len(self._user_tasks.get(user_id, []))

    def can_user_submit(self, user_id: int) -> bool:
        """Check if user can submit a new task."""
        return self.get_user_task_count(user_id) < self.max_per_user

    async def submit(self, task: ProcessingTask) -> bool:
        """
        Submit a task to the queue.

        Returns:
            True if submitted, False if queue full or user limit reached
        """
        async with self._lock:
            if self.is_full:
                return False

            if not self.can_user_submit(task.user_id):
                return False

            # Track task
            self._tasks[task.task_id] = task

            # Track user tasks
            if task.user_id not in self._user_tasks:
                self._user_tasks[task.user_id] = []
            self._user_tasks[task.user_id].append(task.task_id)

            # Update status
            task.progress = TaskProgress(
                status=TaskStatus.QUEUED,
                message=f"Position in queue: {self.size + 1}",
            )

            # Add to queue
            await self._queue.put(task)

            logger.info(
                "task_queued",
                task_id=task.task_id,
                user_id=task.user_id,
                queue_size=self.size,
            )

            return True

    async def get_next(self) -> ProcessingTask:
        """Get the next task from the queue."""
        task = await self._queue.get()
        return task

    def get_task(self, task_id: str) -> Optional[ProcessingTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_user_tasks(self, user_id: int) -> List[ProcessingTask]:
        """Get all tasks for a user."""
        task_ids = self._user_tasks.get(user_id, [])
        return [self._tasks[tid] for tid in task_ids if tid in self._tasks]

    async def complete_task(self, task_id: str) -> None:
        """Mark a task as complete and clean up tracking."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task:
                # Remove from user tracking
                if task.user_id in self._user_tasks:
                    if task_id in self._user_tasks[task.user_id]:
                        self._user_tasks[task.user_id].remove(task_id)

            self._queue.task_done()

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False

            if task.progress.status == TaskStatus.QUEUED:
                task.progress = TaskProgress(
                    status=TaskStatus.CANCELLED,
                    message="Cancelled by user",
                )
                return True

            return False

    def get_queue_stats(self) -> dict:
        """Get queue statistics."""
        return {
            "queue_size": self.size,
            "max_size": self.max_size,
            "total_tasks": len(self._tasks),
            "active_users": len(self._user_tasks),
            "tasks_by_status": self._count_by_status(),
        }

    def _count_by_status(self) -> Dict[str, int]:
        """Count tasks by status."""
        counts: Dict[str, int] = {}
        for task in self._tasks.values():
            status = task.progress.status.value
            counts[status] = counts.get(status, 0) + 1
        return counts


class Worker:
    """
    Main worker for processing transcription tasks.

    Handles the full pipeline: preprocess -> diarize -> transcribe -> aggregate
    """

    def __init__(
        self,
        queue: TaskQueue,
        preprocessor: AudioPreprocessor,
        transcriber: GigaAMTranscriber,
        diarizer: PyAnnoteDiarizer,
        file_manager: FileManager,
        gpu_monitor: GPUMonitor,
        checkpoint_dir: Path,
        timeout_seconds: float = 1800.0,  # 30 minutes
        progress_callback: Optional[ProgressCallback] = None,
    ):
        """
        Initialize worker.

        Args:
            queue: Task queue
            preprocessor: Audio preprocessor
            transcriber: GigaAM transcriber
            diarizer: PyAnnote diarizer
            file_manager: File manager for temp files
            gpu_monitor: GPU monitor
            checkpoint_dir: Directory for checkpoints
            timeout_seconds: Task timeout in seconds
            progress_callback: Callback for progress updates
        """
        self.queue = queue
        self.preprocessor = preprocessor
        self.transcriber = transcriber
        self.diarizer = diarizer
        self.file_manager = file_manager
        self.gpu_monitor = gpu_monitor
        self.checkpoint_dir = checkpoint_dir
        self.timeout_seconds = timeout_seconds
        self.progress_callback = progress_callback

        self._running = False
        self._current_task: Optional[ProcessingTask] = None
        self._worker_task: Optional[asyncio.Task] = None

    @property
    def is_running(self) -> bool:
        """Check if worker is running."""
        return self._running

    @property
    def current_task(self) -> Optional[ProcessingTask]:
        """Get currently processing task."""
        return self._current_task

    def _update_progress(
        self,
        task: ProcessingTask,
        status: TaskStatus,
        progress_percent: float = 0.0,
        current_step: str = "",
        message: str = "",
    ) -> None:
        """Update task progress and notify callback."""
        task.progress = TaskProgress(
            status=status,
            progress_percent=progress_percent,
            current_step=current_step,
            message=message,
            updated_at=datetime.now(),
        )

        if self.progress_callback:
            try:
                self.progress_callback(task.task_id, task.progress)
            except Exception as e:
                logger.error("progress_callback_error", error=str(e))

    async def _process_task(self, task: ProcessingTask) -> None:
        """Process a single task through the full pipeline."""
        task.started_at = datetime.now()
        self._current_task = task

        try:
            # Step 1: Preprocessing
            self._update_progress(
                task,
                TaskStatus.PREPROCESSING,
                progress_percent=0.0,
                current_step="Preprocessing",
                message="Converting audio format...",
            )

            preprocess_result = await self.preprocessor.preprocess(
                task.input_path,
                task.user_id,
                task_id=task.task_id,
            )

            if not preprocess_result.is_valid:
                raise ValueError(f"Preprocessing failed: {preprocess_result.error}")

            audio_path = preprocess_result.processed_path
            duration = preprocess_result.duration_seconds

            logger.info(
                "preprocessing_complete",
                task_id=task.task_id,
                duration=f"{duration:.1f}s",
            )

            # Save checkpoint
            self._save_checkpoint(task, "preprocessed", {
                "audio_path": str(audio_path),
                "duration": duration,
            })

            # Step 2: Diarization
            self._update_progress(
                task,
                TaskStatus.DIARIZING,
                progress_percent=25.0,
                current_step="Diarization",
                message="Identifying speakers...",
            )

            def diarization_progress(chunk: int, total: int, status: str):
                percent = 25.0 + (25.0 * chunk / total) if total > 0 else 25.0
                self._update_progress(
                    task,
                    TaskStatus.DIARIZING,
                    progress_percent=percent,
                    current_step="Diarization",
                    message=f"{status} ({chunk}/{total})",
                )

            diarization_result = await self.diarizer.diarize(
                audio_path,
                progress_callback=diarization_progress,
            )

            if not diarization_result.is_successful:
                raise ValueError(f"Diarization failed: {diarization_result.error}")

            logger.info(
                "diarization_complete",
                task_id=task.task_id,
                speakers=diarization_result.num_speakers,
                segments=len(diarization_result.segments),
            )

            # Save checkpoint
            self._save_checkpoint(task, "diarized", {
                "num_speakers": diarization_result.num_speakers,
                "segment_count": len(diarization_result.segments),
            })

            # Clear GPU cache before transcription
            self.gpu_monitor.clear_cache()

            # Step 3: Transcription & Aggregation
            self._update_progress(
                task,
                TaskStatus.TRANSCRIBING,
                progress_percent=50.0,
                current_step="Transcription",
                message="Transcribing audio...",
            )

            aggregator = ResultAggregator(self.transcriber)

            def aggregation_progress(current: int, total: int, status: str):
                percent = 50.0 + (40.0 * current / total) if total > 0 else 50.0
                self._update_progress(
                    task,
                    TaskStatus.TRANSCRIBING,
                    progress_percent=percent,
                    current_step="Transcription",
                    message=f"{status} ({current}/{total} segments)",
                )

            aggregation_result = await aggregator.aggregate(
                audio_path,
                diarization_result,
                progress_callback=aggregation_progress,
            )

            if not aggregation_result.is_successful:
                raise ValueError(f"Aggregation failed: {aggregation_result.error}")

            task.result = aggregation_result

            logger.info(
                "transcription_complete",
                task_id=task.task_id,
                segments=len(aggregation_result.segments),
                words=aggregation_result.total_words,
            )

            # Step 4: Generate outputs
            self._update_progress(
                task,
                TaskStatus.AGGREGATING,
                progress_percent=90.0,
                current_step="Generating output",
                message="Creating transcript files...",
            )

            results_dir = self.file_manager.get_results_dir(task.user_id)
            output_dir = results_dir / task.task_id
            output_generator = OutputGenerator(output_dir)

            # Generate all output formats
            task.output_files["transcript"] = output_generator.generate_transcript(
                aggregation_result
            )
            task.output_files["plain"] = output_generator.generate_plain_text(
                aggregation_result
            )
            task.output_files["srt"] = output_generator.generate_srt(
                aggregation_result
            )
            task.output_files["json"] = output_generator.generate_json(
                aggregation_result
            )

            # Complete
            task.completed_at = datetime.now()
            self._update_progress(
                task,
                TaskStatus.COMPLETED,
                progress_percent=100.0,
                current_step="Complete",
                message=f"Transcribed {aggregation_result.total_words} words from {aggregation_result.num_speakers} speakers",
            )

            logger.info(
                "task_completed",
                task_id=task.task_id,
                user_id=task.user_id,
                processing_time=f"{task.processing_time_seconds:.1f}s",
            )

            # Clean up checkpoint
            self._clear_checkpoint(task)

        except asyncio.CancelledError:
            task.progress = TaskProgress(
                status=TaskStatus.CANCELLED,
                message="Task was cancelled",
            )
            raise

        except asyncio.TimeoutError:
            task.error = "Task timed out"
            task.progress = TaskProgress(
                status=TaskStatus.TIMEOUT,
                message=f"Task exceeded {self.timeout_seconds}s timeout",
            )
            logger.error("task_timeout", task_id=task.task_id)

        except Exception as e:
            task.error = str(e)
            task.completed_at = datetime.now()
            self._update_progress(
                task,
                TaskStatus.FAILED,
                message=f"Error: {str(e)[:100]}",
            )
            logger.error(
                "task_failed",
                task_id=task.task_id,
                error=str(e),
            )

        finally:
            self._current_task = None

            # Clean up temp files (keep results)
            try:
                self.preprocessor.cleanup_task(task.user_id, task.task_id)
            except Exception as e:
                logger.warning("cleanup_failed", task_id=task.task_id, error=str(e))

            # Clear GPU cache
            self.gpu_monitor.clear_cache()

    def _save_checkpoint(
        self,
        task: ProcessingTask,
        stage: str,
        data: Dict[str, Any],
    ) -> None:
        """Save a checkpoint for task recovery."""
        try:
            checkpoint = Checkpoint(
                task_id=task.task_id,
                user_id=task.user_id,
                stage=stage,
                progress_data=data,
            )
            task.checkpoint_path = checkpoint.save(self.checkpoint_dir)
        except Exception as e:
            logger.warning("checkpoint_save_failed", error=str(e))

    def _clear_checkpoint(self, task: ProcessingTask) -> None:
        """Clear checkpoint after successful completion."""
        if task.checkpoint_path and task.checkpoint_path.exists():
            try:
                task.checkpoint_path.unlink()
            except Exception:
                pass

    async def _worker_loop(self) -> None:
        """Main worker loop."""
        logger.info("worker_started")

        while self._running:
            try:
                # Get next task with timeout to allow checking _running flag
                try:
                    task = await asyncio.wait_for(
                        self.queue.get_next(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                # Check if task was cancelled while in queue
                if task.progress.status == TaskStatus.CANCELLED:
                    await self.queue.complete_task(task.task_id)
                    continue

                # Process with timeout
                try:
                    await asyncio.wait_for(
                        self._process_task(task),
                        timeout=self.timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    task.error = "Processing timeout"
                    task.progress = TaskProgress(
                        status=TaskStatus.TIMEOUT,
                        message=f"Task exceeded {self.timeout_seconds}s timeout",
                    )

                await self.queue.complete_task(task.task_id)

            except asyncio.CancelledError:
                logger.info("worker_cancelled")
                break
            except Exception as e:
                logger.error("worker_loop_error", error=str(e))
                await asyncio.sleep(1)

        logger.info("worker_stopped")

    async def start(self) -> None:
        """Start the worker."""
        if self._running:
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self, timeout: float = 30.0) -> None:
        """Stop the worker gracefully."""
        if not self._running:
            return

        self._running = False

        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=timeout)
            except asyncio.TimeoutError:
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except asyncio.CancelledError:
                    pass

        self._worker_task = None

    def get_status(self) -> dict:
        """Get worker status."""
        return {
            "is_running": self._running,
            "current_task": self._current_task.task_id if self._current_task else None,
            "queue_stats": self.queue.get_queue_stats(),
            "gpu_stats": self.gpu_monitor.get_stats().__dict__ if self.gpu_monitor.is_cuda_available() else None,
        }


def create_worker(
    temp_dir: Path,
    checkpoint_dir: Path,
    hf_token: str,
    max_queue_size: int = 50,
    max_per_user: int = 3,
    max_vram_gb: float = 14.0,
    timeout_seconds: float = 1800.0,
    progress_callback: Optional[ProgressCallback] = None,
) -> Worker:
    """
    Create a fully configured worker instance.

    Args:
        temp_dir: Temporary files directory
        checkpoint_dir: Checkpoint directory
        hf_token: Hugging Face token for PyAnnote
        max_queue_size: Maximum queue size
        max_per_user: Maximum tasks per user
        max_vram_gb: Maximum VRAM allocation
        timeout_seconds: Task timeout
        progress_callback: Progress update callback

    Returns:
        Configured Worker instance
    """
    # Create components
    file_manager = FileManager(temp_dir)
    gpu_monitor = get_gpu_monitor(max_vram_gb)
    queue = TaskQueue(max_size=max_queue_size, max_per_user=max_per_user)

    preprocessor = AudioPreprocessor(file_manager)
    transcriber = get_transcriber()
    diarizer = get_diarizer(hf_token=hf_token)

    return Worker(
        queue=queue,
        preprocessor=preprocessor,
        transcriber=transcriber,
        diarizer=diarizer,
        file_manager=file_manager,
        gpu_monitor=gpu_monitor,
        checkpoint_dir=checkpoint_dir,
        timeout_seconds=timeout_seconds,
        progress_callback=progress_callback,
    )


def create_task(
    user_id: int,
    input_path: Path,
    original_filename: str,
    task_id: Optional[str] = None,
) -> ProcessingTask:
    """
    Create a new processing task.

    Args:
        user_id: User ID
        input_path: Path to input audio file
        original_filename: Original filename
        task_id: Optional task ID (generated if not provided)

    Returns:
        ProcessingTask ready for submission
    """
    return ProcessingTask(
        task_id=task_id or str(uuid.uuid4())[:8],
        user_id=user_id,
        input_path=input_path,
        original_filename=original_filename,
    )
