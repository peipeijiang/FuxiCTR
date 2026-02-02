# =========================================================================
# Copyright (C) 2026. The FuxiCTR Library. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""
Workflow Logger for real-time log streaming.

Provides logging functionality that broadcasts messages to WebSocket clients.
"""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class WorkflowLogger:
    """
    Workflow Logger with real-time WebSocket broadcasting.

    This logger supports two modes:
    1. Direct broadcasting via callback (preferred for real-time)
    2. Queue-based logging (for async processing)
    """
    task_id: int
    websocket_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    broadcast_callback: Optional[Callable[[int, Dict[str, Any]], Any]] = None

    def _format_message(self, msg_type: str, step: str, data: Any) -> Dict:
        """Format a log message for WebSocket transmission"""
        return {
            "type": msg_type,
            "task_id": self.task_id,
            "step": step,
            "data": data,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def _broadcast(self, message: Dict[str, Any]):
        """
        Broadcast message via callback or put in queue.

        Priority:
        1. Direct callback (real-time broadcasting)
        2. Queue fallback (async processing)
        """
        if self.broadcast_callback:
            try:
                # Direct callback for real-time broadcasting
                asyncio.create_task(self._call_broadcast(message))
            except Exception as e:
                logger.warning(f"Broadcast callback failed: {e}")
                # Fallback to queue
                asyncio.create_task(self.websocket_queue.put(message))

        # Also put in queue for persistence/retrieval
        asyncio.create_task(self.websocket_queue.put(message))

    async def _call_broadcast(self, message: Dict[str, Any]):
        """Call the broadcast callback synchronously."""
        if self.broadcast_callback:
            try:
                result = self.broadcast_callback(self.task_id, message)
                # If callback returns a coroutine, await it
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Broadcast callback error: {e}")

    def log(self, step: str, message: str, level: str = "INFO") -> Dict:
        """Log a message"""
        msg = self._format_message("log", step, {
            "level": level,
            "message": message
        })
        self._broadcast(msg)
        return msg

    def progress(self, step: str, current: int, total: int, message: str = "") -> Dict:
        """Log progress update"""
        percent = int((current / total) * 100) if total > 0 else 0
        msg = self._format_message("progress", step, {
            "current": current,
            "total": total,
            "percent": percent,
            "message": message
        })
        self._broadcast(msg)
        return msg

    def metric(self, step: str, metric_name: str, value: Any, unit: str = "") -> Dict:
        """Log a metric update"""
        msg = self._format_message("metric", step, {
            "name": metric_name,
            "value": value,
            "unit": unit
        })
        self._broadcast(msg)
        return msg

    def error(self, step: str, error: str) -> Dict:
        """Log an error"""
        msg = self._format_message("error", step, {
            "message": error
        })
        self._broadcast(msg)
        return msg

    def complete(self, step: str, result: Optional[Dict[str, Any]] = None) -> Dict:
        """Mark step as complete with optional result"""
        msg = self._format_message("complete", step, {
            "result": result or {}
        })
        self._broadcast(msg)
        return msg


class LoggerBroadcaster:
    """
    Manages log broadcasting for multiple tasks.

    This component runs a background task that processes log queues
    and broadcasts messages to WebSocket clients.
    """

    def __init__(self):
        self._active_tasks: Dict[int, WorkflowLogger] = {}
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

    def register_logger(self, task_id: int, logger_obj: WorkflowLogger):
        """Register a logger for broadcasting."""
        self._active_tasks[task_id] = logger_obj

    def unregister_logger(self, task_id: int):
        """Unregister a logger."""
        self._active_tasks.pop(task_id, None)

    async def process_queues(self):
        """
        Background worker that processes log queues.

        This runs continuously and pulls messages from each logger's queue,
        then broadcasts them via the callback.
        """
        self._running = True
        logger.info("Logger broadcaster started")

        while self._running:
            try:
                # Process queues from all active loggers
                for task_id, logger_obj in list(self._active_tasks.items()):
                    try:
                        # Process all available messages from this logger
                        while not logger_obj.websocket_queue.empty():
                            message = await asyncio.wait_for(
                                logger_obj.websocket_queue.get(),
                                timeout=0.1
                            )
                            # Broadcast via callback
                            if logger_obj.broadcast_callback:
                                result = logger_obj.broadcast_callback(task_id, message)
                                if asyncio.iscoroutine(result):
                                    await result
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing queue for task {task_id}: {e}")

                # Small sleep to prevent busy-waiting
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
                await asyncio.sleep(1)

        logger.info("Logger broadcaster stopped")

    def start(self):
        """Start the background worker."""
        if not self._running:
            self._worker_task = asyncio.create_task(self.process_queues())

    async def stop(self):
        """Stop the background worker."""
        self._running = False
        if self._worker_task:
            await asyncio.gather(self._worker_task, return_exceptions=True)
            self._worker_task = None


# Global broadcaster instance
_broadcaster: Optional[LoggerBroadcaster] = None


def get_broadcaster() -> LoggerBroadcaster:
    """Get the global broadcaster instance."""
    global _broadcaster
    if _broadcaster is None:
        _broadcaster = LoggerBroadcaster()
    return _broadcaster
