import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from orchestrator import RepoOrchestrator
from parser import FileUnits


@dataclass
class _DummyTracker:
    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    cache_hits: int = 0

    def report(self) -> str:
        return "dummy"


class _SlowSummarizer:
    def __init__(self, delays: dict[str, float], fail_name: str | None = None):
        self.delays = delays
        self.fail_name = fail_name
        self.tracker = _DummyTracker()

    async def summarize_file(self, file_units: FileUnits) -> str:
        name = Path(file_units.path).name
        await asyncio.sleep(self.delays.get(name, 0.0))
        if self.fail_name == name:
            raise RuntimeError("boom")
        return f"summary:{name}"


def test_phase3_logs_heartbeat_and_progress(tmp_path: Path):
    orch = RepoOrchestrator(
        api_key="test",
        cache_dir=tmp_path / "cache",
        checkpoint_dir=tmp_path / "checkpoints",
        verbose=True,
        progress_heartbeat_secs=0.5,
    )
    orch.summarizer = _SlowSummarizer({"a.py": 0.6, "b.py": 0.6, "c.py": 0.6})
    logs: list[str] = []
    orch._log = logs.append  # type: ignore[method-assign]

    files = [
        FileUnits(path="a.py", language="python"),
        FileUnits(path="b.py", language="python"),
        FileUnits(path="c.py", language="python"),
    ]
    result = asyncio.run(orch._summarize_files(files))

    assert set(result.keys()) == {"a.py", "b.py", "c.py"}
    assert any("Phase 3 heartbeat:" in msg for msg in logs)
    assert any("Phase 3 progress:" in msg for msg in logs)


def test_phase3_failure_includes_file_path(tmp_path: Path):
    orch = RepoOrchestrator(
        api_key="test",
        cache_dir=tmp_path / "cache",
        checkpoint_dir=tmp_path / "checkpoints",
        verbose=False,
        progress_heartbeat_secs=0.5,
    )
    orch.summarizer = _SlowSummarizer({"a.py": 1.0, "b.py": 0.0}, fail_name="b.py")

    files = [
        FileUnits(path="a.py", language="python"),
        FileUnits(path="b.py", language="python"),
    ]
    with pytest.raises(RuntimeError, match=r"Phase 3 failed while summarizing `b\.py`"):
        asyncio.run(orch._summarize_files(files))
