"""
Tqdm adapter for broadcasting progress to WebSocket.

This module provides adapters that integrate tqdm progress bars with the
Dashboard WebSocket system, enabling real-time progress visualization.
"""

import sys
import time
from typing import Optional, Any, Dict
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class TqdmWebSocketAdapter(tqdm):
    """
    tqdm adapter that broadcasts progress to WebSocket for Dashboard display.

    Features:
    - Dual mode: stdout (CLI) + WebSocket (Dashboard)
    - Automatic broadcast throttling to avoid performance impact
    - Graceful fallback when WebSocket is unavailable

    Example:
        >>> from fuxictr.pytorch.utils.tqdm_adapter import TqdmWebSocketAdapter
        >>> # With WorkflowLogger
        >>> progress_bar = TqdmWebSocketAdapter(
        ...     data_generator,
        ...     logger=workflow_logger,
        ...     step_name="train",
        ...     desc="Training"
        ... )
        >>> for batch in progress_bar:
        ...     train(batch)

    Args:
        iterable: Iterable to decorate with a progressbar.
        logger: WorkflowLogger instance for WebSocket broadcasting.
        step_name: Name of the step for progress tracking.
        rank: Process rank for distributed training.
        world_size: Total number of processes.
        broadcast_interval: Minimum seconds between broadcasts (default: 0.5).
        broadcast_threshold: Minimum percentage change between broadcasts (default: 1).
    """

    def __init__(
        self,
        iterable=None,
        logger=None,
        step_name: str = "unknown",
        rank: Optional[int] = None,
        world_size: int = 1,
        broadcast_interval: float = 0.5,
        broadcast_threshold: float = 1.0,
        **kwargs
    ):
        self._ws_logger = logger
        self._step_name = step_name
        self._rank = rank
        self._world_size = world_size
        self._broadcast_interval = broadcast_interval
        self._broadcast_threshold = broadcast_threshold

        # Track broadcast state
        self._last_broadcast = 0
        self._last_broadcast_time = 0
        self._last_broadcast_pct = 0

        # Always allow stdout output for CLI visibility
        kwargs['file'] = kwargs.get('file', sys.stdout)

        super().__init__(iterable, **kwargs)

    def _should_broadcast(self) -> bool:
        """Determine if progress should be broadcast based on throttling rules."""
        if not self._ws_logger:
            return False

        current_time = time.time()
        current_pct = self.n * 100 / self.total if self.total > 0 else 0

        # Always broadcast at 0% and 100%
        if self.n == 0 or self.n == self.total:
            return True

        # Check time threshold
        time_elapsed = current_time - self._last_broadcast_time
        if time_elapsed < self._broadcast_interval:
            return False

        # Check percentage change threshold
        pct_change = abs(current_pct - self._last_broadcast_pct)
        if pct_change < self._broadcast_threshold:
            return False

        return True

    def _broadcast_progress(self):
        """Broadcast current progress to WebSocket."""
        if not self._ws_logger:
            return

        try:
            current_pct = int(self.n * 100 / self.total) if self.total > 0 else 0

            # Build message with context
            message_parts = []
            if self._world_size > 1 and self._rank is not None:
                message_parts.append(f"Rank {self._rank}/{self._world_size}")

            # Add postfix if available (loss, speed, etc.)
            if self.postfix:
                message_parts.append(str(self.postfix))

            message = " | ".join(message_parts) if message_parts else ""

            # Broadcast progress
            self._ws_logger.progress(
                step=self._step_name,
                current=self.n,
                total=self.total,
                message=message
            )

            # Update broadcast state
            self._last_broadcast = self.n
            self._last_broadcast_time = time.time()
            self._last_broadcast_pct = current_pct

        except Exception as e:
            # Disable future broadcasts to avoid spamming
            logger.warning(f"WebSocket broadcast failed: {e}. Disabling future broadcasts.")
            self._ws_logger = None

    def update(self, n=1):
        """Override to broadcast progress updates."""
        updated = super().update(n)

        if updated and self._should_broadcast():
            self._broadcast_progress()

        return updated

    def close(self):
        """Close the progressbar and broadcast completion."""
        # Final broadcast at 100%
        if self._ws_logger and self.n < self.total:
            try:
                self._ws_logger.progress(
                    step=self._step_name,
                    current=self.total,
                    total=self.total,
                    message="Completed"
                )
            except Exception:
                pass  # Already handled in _broadcast_progress

        super().close()


class DistributedTqdmAdapter(TqdmWebSocketAdapter):
    """
    Tqdm adapter for distributed training that aggregates progress across all ranks.

    Only rank 0 broadcasts to WebSocket, but collects progress from all ranks
    using torch.distributed communication.

    Example:
        >>> progress_bar = DistributedTqdmAdapter(
        ...     data_generator,
        ...     logger=workflow_logger,
        ...     step_name="train",
        ...     rank=dist.get_rank(),
        ...     world_size=dist.get_world_size()
        ... )

    Note:
        Requires torch.distributed to be initialized.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._all_ranks_progress = [0] * self._world_size
        self._dist_available = False

        # Check if torch.distributed is available
        try:
            import torch.distributed as dist
            self._dist = dist
            self._dist_available = dist.is_available()
        except ImportError:
            logger.debug("torch.distributed not available, using non-distributed mode")
            self._dist_available = False

    def _gather_all_ranks_progress(self):
        """Gather progress from all ranks."""
        if not self._dist_available:
            return

        try:
            # Check if distributed is initialized
            if not self._dist.is_initialized():
                return

            # Create tensor for local progress
            local_progress = self.n

            # Gather all ranks' progress
            gathered = [0] * self._world_size
            self._dist.all_gather_object(gathered, local_progress)

            self._all_ranks_progress = gathered

        except Exception as e:
            logger.warning(f"Failed to gather progress from all ranks: {e}")

    def _broadcast_progress(self):
        """Broadcast aggregated progress from all ranks."""
        # Gather progress from all ranks first
        if self._world_size > 1:
            self._gather_all_ranks_progress()

        # Only rank 0 broadcasts to WebSocket
        if self._rank != 0 or not self._ws_logger:
            return

        try:
            current_pct = int(self.n * 100 / self.total) if self.total > 0 else 0

            # Build aggregated message
            message_parts = []

            if self._world_size > 1:
                # Show individual rank progress
                rank_progress = ", ".join([f"R{i}:{p}" for i, p in enumerate(self._all_ranks_progress)])
                message_parts.append(f"[{rank_progress}]")

                # Total progress across all ranks
                total_processed = sum(self._all_ranks_progress)
                message_parts.append(f"Total: {total_processed}/{self.total * self._world_size}")

            # Add postfix if available
            if self.postfix:
                message_parts.append(str(self.postfix))

            message = " | ".join(message_parts) if message_parts else ""

            # Broadcast aggregated progress
            self._ws_logger.progress(
                step=self._step_name,
                current=sum(self._all_ranks_progress),
                total=self.total * self._world_size,
                message=message
            )

            # Update broadcast state
            self._last_broadcast = self.n
            self._last_broadcast_time = time.time()
            self._last_broadcast_pct = current_pct

        except Exception as e:
            logger.warning(f"WebSocket broadcast failed: {e}. Disabling future broadcasts.")
            self._ws_logger = None


def create_progress_adapter(
    iterable,
    logger=None,
    step_name: str = "unknown",
    rank: Optional[int] = None,
    world_size: int = 1,
    enable_distributed: bool = False,
    **kwargs
) -> tqdm:
    """
    Factory function to create the appropriate tqdm adapter.

    Args:
        iterable: Iterable to decorate with a progressbar.
        logger: WorkflowLogger instance for WebSocket broadcasting.
        step_name: Name of the step for progress tracking.
        rank: Process rank for distributed training.
        world_size: Total number of processes.
        enable_distributed: Whether to use distributed adapter.
        **kwargs: Additional arguments passed to tqdm.

    Returns:
        tqdm: Either TqdmWebSocketAdapter, DistributedTqdmAdapter, or standard tqdm.
    """
    # No logger provided, use standard tqdm
    if logger is None:
        return tqdm(iterable, **kwargs)

    # Distributed training requested
    if enable_distributed and world_size > 1:
        return DistributedTqdmAdapter(
            iterable,
            logger=logger,
            step_name=step_name,
            rank=rank,
            world_size=world_size,
            **kwargs
        )

    # Standard WebSocket adapter
    return TqdmWebSocketAdapter(
        iterable,
        logger=logger,
        step_name=step_name,
        rank=rank,
        world_size=world_size,
        **kwargs
    )
