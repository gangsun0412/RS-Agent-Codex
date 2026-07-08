"""
任务日志模块
记录每次 Agent 交互和工具调用的详细信息，支持 JSON Lines 格式持久化
"""
import os
import json
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TaskLog:
    """单条任务日志"""
    task_id: str
    timestamp: str
    user_input: str
    intent: str = ""
    confidence: float = 0.0
    route: str = ""
    tool_calls: list = field(default_factory=list)
    final_answer: str = ""
    status: str = "pending"  # pending, running, success, failed, cancelled
    error: Optional[str] = None
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class TaskLogger:
    """任务日志记录器，将日志写入 JSON Lines 文件"""

    def __init__(self, log_dir: str):
        """
        Args:
            log_dir: 日志目录路径
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._current_task: Optional[TaskLog] = None
        self._start_time: float = 0.0

    def start_task(self, user_input: str) -> str:
        """
        开始一个新任务，返回 task_id

        Args:
            user_input: 用户原始输入

        Returns:
            task_id
        """
        task_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self._start_time = time.time()

        self._current_task = TaskLog(
            task_id=task_id,
            timestamp=datetime.now().isoformat(),
            user_input=user_input,
            status="running"
        )
        return task_id

    def log_intent(self, intent: str, confidence: float, route: str):
        """记录意图识别结果"""
        if self._current_task:
            self._current_task.intent = intent
            self._current_task.confidence = confidence
            self._current_task.route = route

    def log_tool_call(self, tool_name: str, tool_args: dict, result_preview: str):
        """记录工具调用"""
        if self._current_task:
            self._current_task.tool_calls.append({
                "tool_name": tool_name,
                "args": tool_args,
                "result_preview": result_preview[:500],  # 截断过长结果
                "timestamp": datetime.now().isoformat()
            })

    def finish_task(self, status: str = "success", final_answer: str = "",
                    error: str = None):
        """结束任务并写入日志文件"""
        if self._current_task:
            self._current_task.status = status
            self._current_task.final_answer = final_answer[:1000]  # 截断
            self._current_task.error = error
            self._current_task.elapsed_seconds = round(
                time.time() - self._start_time, 2
            )

            # 写入 JSON Lines 文件
            log_file = os.path.join(
                self.log_dir,
                f"tasks_{datetime.now().strftime('%Y%m%d')}.jsonl"
            )
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(self._current_task.to_dict(), ensure_ascii=False) + "\n")

            self._current_task = None
            return log_file

    def cancel_task(self, reason: str = "用户取消"):
        """取消当前任务"""
        self.finish_task(status="cancelled", error=reason)
