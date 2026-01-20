import json
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, field
import asyncio

@dataclass
class WorkflowLogger:
    task_id: int
    websocket_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    def _format_message(self, msg_type: str, step: str, data: Any) -> Dict:
        """Format a log message for WebSocket transmission"""
        return {
            "type": msg_type,
            "task_id": self.task_id,
            "step": step,
            "data": data,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def log(self, step: str, message: str, level: str = "INFO") -> Dict:
        """Log a message"""
        msg = self._format_message("log", step, {
            "level": level,
            "message": message
        })
        asyncio.create_task(self.websocket_queue.put(msg))
        return msg

    def progress(self, step: str, current: int, total: int) -> Dict:
        """Log progress"""
        percent = int((current / total) * 100) if total > 0 else 0
        msg = self._format_message("progress", step, {
            "current": current,
            "total": total,
            "percent": percent
        })
        asyncio.create_task(self.websocket_queue.put(msg))
        return msg

    def error(self, step: str, error: str) -> Dict:
        """Log an error"""
        msg = self._format_message("error", step, {
            "message": error
        })
        asyncio.create_task(self.websocket_queue.put(msg))
        return msg

    def complete(self, step: str) -> Dict:
        """Mark step as complete"""
        msg = self._format_message("complete", step, {})
        asyncio.create_task(self.websocket_queue.put(msg))
        return msg
