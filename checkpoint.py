"""
checkpoint.py — Persistent DAG checkpoint state for resumable runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Optional


PHASES = ("files", "docs", "dirs", "modules")


def _default_phase_state() -> dict[str, bool]:
    return {phase: False for phase in PHASES}


@dataclass
class CheckpointState:
    run_id: str
    repo_path: str
    started_at: str
    file_summaries: dict[str, str] = field(default_factory=dict)
    dir_summaries: dict[str, str] = field(default_factory=dict)
    module_summaries: list[dict] = field(default_factory=list)
    doc_summaries: list[dict] = field(default_factory=list)
    phase_complete: dict[str, bool] = field(default_factory=_default_phase_state)

    def mark_phase_complete(self, phase_name: str) -> None:
        if phase_name not in PHASES:
            raise ValueError(f"Unknown phase: {phase_name}")
        self.phase_complete[phase_name] = True


class CheckpointStore:
    def __init__(self, checkpoint_dir: Optional[Path] = None):
        if checkpoint_dir is None:
            checkpoint_dir = Path.home() / ".repo_summarizer_cache" / "checkpoints"
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint_path(self, run_id: str) -> Path:
        return self.checkpoint_dir / f"{run_id}.json"

    def save(self, state: CheckpointState) -> Path:
        path = self.checkpoint_path(state.run_id)
        tmp = path.with_suffix(".tmp")
        payload = json.dumps(asdict(state), indent=2)
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
        return path

    def load(self, checkpoint_path: Path) -> Optional[CheckpointState]:
        try:
            raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        try:
            phase_complete = _default_phase_state()
            phase_complete.update(raw.get("phase_complete", {}))
            return CheckpointState(
                run_id=raw["run_id"],
                repo_path=raw["repo_path"],
                started_at=raw["started_at"],
                file_summaries=raw.get("file_summaries", {}),
                dir_summaries=raw.get("dir_summaries", {}),
                module_summaries=raw.get("module_summaries", []),
                doc_summaries=raw.get("doc_summaries", []),
                phase_complete=phase_complete,
            )
        except (KeyError, TypeError):
            return None

    def find_latest(self, repo_path: str | Path) -> Optional[Path]:
        repo_path = str(Path(repo_path).resolve())
        candidates: list[tuple[datetime, Path]] = []
        for p in self.checkpoint_dir.glob("*.json"):
            state = self.load(p)
            if state is None:
                continue
            if str(Path(state.repo_path).resolve()) != repo_path:
                continue
            try:
                dt = datetime.fromisoformat(state.started_at.replace("Z", "+00:00"))
            except ValueError:
                continue
            candidates.append((dt, p))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
