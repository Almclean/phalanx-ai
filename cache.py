"""
cache.py — Content-addressed cache for summaries.

Keyed by SHA-256 of the source content + prompt layer, so re-runs
only re-summarize files that have actually changed.
"""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Optional


class SummaryCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, str] = {}  # in-memory layer
        self._load()

    def _cache_file(self) -> Path:
        return self.cache_dir / "summaries.json"

    def _load(self):
        f = self._cache_file()
        if f.exists():
            try:
                self._mem = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                self._mem = {}

    def _save(self):
        self._cache_file().write_text(json.dumps(self._mem, indent=2))

    @staticmethod
    def hash(content: str, layer: str) -> str:
        return hashlib.sha256(f"{layer}:{content}".encode()).hexdigest()[:32]

    def get(self, content: str, layer: str) -> Optional[str]:
        key = self.hash(content, layer)
        return self._mem.get(key)

    def set(self, content: str, layer: str, summary: str):
        key = self.hash(content, layer)
        self._mem[key] = summary
        self._save()

    def stats(self) -> dict:
        return {"cached_entries": len(self._mem)}
