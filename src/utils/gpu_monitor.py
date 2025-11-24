"""GPU monitoring utilities for VRAM tracking and management."""

import asyncio
from dataclasses import dataclass
from typing import Callable, Optional
import structlog

logger = structlog.get_logger(__name__)

# Try to import torch - may not be available during initial setup
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("pytorch_not_available", message="GPU monitoring limited")


@dataclass
class GPUStats:
    """GPU statistics snapshot."""

    device_name: str
    total_memory_gb: float
    allocated_memory_gb: float
    reserved_memory_gb: float
    free_memory_gb: float
    utilization_percent: float
    cuda_version: str
    is_available: bool


class GPUMonitor:
    """Monitors GPU memory usage and provides management utilities."""

    def __init__(
        self,
        max_vram_gb: float = 14.0,
        warning_threshold: float = 0.85,
        critical_threshold: float = 0.95,
        device_id: int = 0,
    ):
        """
        Initialize GPU monitor.

        Args:
            max_vram_gb: Maximum VRAM to use (leave headroom for system)
            warning_threshold: Fraction of max_vram to trigger warning
            critical_threshold: Fraction of max_vram to trigger critical alert
            device_id: CUDA device ID to monitor
        """
        self.max_vram_gb = max_vram_gb
        self.max_vram_bytes = int(max_vram_gb * 1024 * 1024 * 1024)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.device_id = device_id

        self._on_warning: Optional[Callable[[GPUStats], None]] = None
        self._on_critical: Optional[Callable[[GPUStats], None]] = None
        self._monitor_task: Optional[asyncio.Task] = None

        self._check_cuda_available()

    def _check_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        if not TORCH_AVAILABLE:
            logger.warning("cuda_check_skipped", reason="PyTorch not imported")
            return False

        if not torch.cuda.is_available():
            logger.warning("cuda_not_available")
            return False

        device_count = torch.cuda.device_count()
        if self.device_id >= device_count:
            logger.warning(
                "invalid_device_id",
                device_id=self.device_id,
                available_devices=device_count
            )
            return False

        return True

    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return TORCH_AVAILABLE and torch.cuda.is_available()

    def get_stats(self) -> GPUStats:
        """Get current GPU statistics."""
        if not self.is_cuda_available():
            return GPUStats(
                device_name="N/A",
                total_memory_gb=0.0,
                allocated_memory_gb=0.0,
                reserved_memory_gb=0.0,
                free_memory_gb=0.0,
                utilization_percent=0.0,
                cuda_version="N/A",
                is_available=False,
            )

        try:
            device = torch.device(f"cuda:{self.device_id}")

            # Get memory info
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            reserved_memory = torch.cuda.memory_reserved(device)
            free_memory = total_memory - allocated_memory

            # Calculate utilization based on our configured max
            effective_max = min(total_memory, self.max_vram_bytes)
            utilization = (allocated_memory / effective_max) * 100 if effective_max > 0 else 0

            return GPUStats(
                device_name=torch.cuda.get_device_name(device),
                total_memory_gb=total_memory / (1024**3),
                allocated_memory_gb=allocated_memory / (1024**3),
                reserved_memory_gb=reserved_memory / (1024**3),
                free_memory_gb=free_memory / (1024**3),
                utilization_percent=utilization,
                cuda_version=torch.version.cuda or "Unknown",
                is_available=True,
            )
        except Exception as e:
            logger.error("gpu_stats_error", error=str(e))
            return GPUStats(
                device_name="Error",
                total_memory_gb=0.0,
                allocated_memory_gb=0.0,
                reserved_memory_gb=0.0,
                free_memory_gb=0.0,
                utilization_percent=0.0,
                cuda_version="Error",
                is_available=False,
            )

    def get_available_memory_gb(self) -> float:
        """Get available VRAM in GB within configured limits."""
        stats = self.get_stats()
        if not stats.is_available:
            return 0.0

        # Available is min of (physical free, configured limit - allocated)
        physical_free = stats.free_memory_gb
        configured_available = self.max_vram_gb - stats.allocated_memory_gb

        return max(0.0, min(physical_free, configured_available))

    def can_allocate(self, required_gb: float) -> bool:
        """Check if the requested memory can be allocated."""
        available = self.get_available_memory_gb()
        return available >= required_gb

    def clear_cache(self) -> None:
        """Clear CUDA memory cache."""
        if self.is_cuda_available():
            torch.cuda.empty_cache()
            logger.debug("cuda_cache_cleared")

    def synchronize(self) -> None:
        """Synchronize CUDA operations."""
        if self.is_cuda_available():
            torch.cuda.synchronize()

    def set_warning_callback(self, callback: Callable[[GPUStats], None]) -> None:
        """Set callback for warning threshold events."""
        self._on_warning = callback

    def set_critical_callback(self, callback: Callable[[GPUStats], None]) -> None:
        """Set callback for critical threshold events."""
        self._on_critical = callback

    def check_thresholds(self) -> Optional[str]:
        """
        Check if memory usage exceeds thresholds.

        Returns:
            'critical', 'warning', or None
        """
        stats = self.get_stats()
        if not stats.is_available:
            return None

        usage_fraction = stats.allocated_memory_gb / self.max_vram_gb

        if usage_fraction >= self.critical_threshold:
            if self._on_critical:
                self._on_critical(stats)
            return "critical"
        elif usage_fraction >= self.warning_threshold:
            if self._on_warning:
                self._on_warning(stats)
            return "warning"

        return None

    async def start_monitoring(self, interval_seconds: float = 5.0) -> None:
        """Start background memory monitoring."""
        if self._monitor_task is not None:
            return

        async def _monitor_loop():
            while True:
                try:
                    await asyncio.sleep(interval_seconds)

                    status = self.check_thresholds()
                    if status == "critical":
                        stats = self.get_stats()
                        logger.warning(
                            "gpu_memory_critical",
                            allocated_gb=stats.allocated_memory_gb,
                            max_gb=self.max_vram_gb,
                            utilization=stats.utilization_percent,
                        )
                    elif status == "warning":
                        stats = self.get_stats()
                        logger.info(
                            "gpu_memory_warning",
                            allocated_gb=stats.allocated_memory_gb,
                            max_gb=self.max_vram_gb,
                        )
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("gpu_monitor_error", error=str(e))

        self._monitor_task = asyncio.create_task(_monitor_loop())
        logger.info("gpu_monitoring_started", interval=interval_seconds)

    async def stop_monitoring(self) -> None:
        """Stop background memory monitoring."""
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
            logger.info("gpu_monitoring_stopped")

    def log_status(self) -> None:
        """Log current GPU status."""
        stats = self.get_stats()

        if stats.is_available:
            logger.info(
                "gpu_status",
                device=stats.device_name,
                total_gb=f"{stats.total_memory_gb:.1f}",
                allocated_gb=f"{stats.allocated_memory_gb:.2f}",
                free_gb=f"{stats.free_memory_gb:.2f}",
                utilization=f"{stats.utilization_percent:.1f}%",
                cuda_version=stats.cuda_version,
            )
        else:
            logger.warning("gpu_status", available=False, reason="CUDA not available")

    def get_optimal_batch_size(
        self,
        base_batch_size: int,
        memory_per_sample_gb: float,
        min_batch_size: int = 1,
    ) -> int:
        """
        Calculate optimal batch size based on available memory.

        Args:
            base_batch_size: Desired batch size
            memory_per_sample_gb: Estimated memory per sample
            min_batch_size: Minimum acceptable batch size

        Returns:
            Optimal batch size
        """
        available = self.get_available_memory_gb()
        if available <= 0:
            return min_batch_size

        # Leave 10% headroom
        usable = available * 0.9
        max_possible = int(usable / memory_per_sample_gb) if memory_per_sample_gb > 0 else base_batch_size

        return max(min_batch_size, min(base_batch_size, max_possible))


# Global instance for convenience
_gpu_monitor: Optional[GPUMonitor] = None


def get_gpu_monitor(max_vram_gb: float = 14.0) -> GPUMonitor:
    """Get or create global GPU monitor instance."""
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor(max_vram_gb=max_vram_gb)
    return _gpu_monitor
