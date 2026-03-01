"""Audit Logger — Structured JSON-L logging with PII protection."""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

from src.utils import hash_user_id


class AuditLogger:
    """Logs every pipeline request to a JSON-Lines file.
    
    Logged fields: timestamp, user_id_hash (never raw), prompt_hash,
    response_length, latency_ms, guardrail_flags, cache_hit.
    """

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "audit.jsonl")

    def log(
        self,
        user_id: str,
        prompt: str,
        response_length: int,
        latency_ms: float,
        guardrail_flags: List[str],
        cache_hit: bool,
        visualizations: List[str],
        error: Optional[str] = None,
    ) -> None:
        """Write a single audit log entry."""
        import hashlib

        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id_hash": hash_user_id(user_id),
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "prompt_length": len(prompt),
            "response_length": response_length,
            "latency_ms": round(latency_ms, 2),
            "guardrail_flags": guardrail_flags,
            "cache_hit": cache_hit,
            "num_visualizations": len(visualizations),
            "error": error,
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
