import asyncio
from dataclasses import dataclass
from pathlib import Path

from checkpoint import CheckpointState, CheckpointStore
from orchestrator import RepoOrchestrator


@dataclass
class _DummyTracker:
    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    cache_hits: int = 0

    def report(self) -> str:
        return "dummy"


class _DummySummarizer:
    def __init__(self):
        self.tracker = _DummyTracker()
        self.file_calls = 0
        self.doc_calls = 0
        self.dir_calls = 0
        self.module_calls = 0
        self.final_calls = 0

    async def summarize_file(self, file_units):
        self.file_calls += 1
        return f"file-summary:{Path(file_units.path).name}"

    async def summarize_doc_files_parallel(self, docs):
        self.doc_calls += len(docs)
        return [(d, f"doc-summary:{Path(d.path).name}") for d in docs]

    async def summarize_directory(self, *, dir_path, file_summaries, subdirectory_summaries):
        self.dir_calls += 1
        return f"dir-summary:{dir_path}:{len(file_summaries)}:{len(subdirectory_summaries)}"

    async def summarize_module(self, *, module_name, directory_summaries):
        self.module_calls += 1
        return f"module-summary:{module_name}:{len(directory_summaries)}"

    async def synthesize_final(self, **kwargs):
        self.final_calls += 1
        return "final-summary"


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_orchestrator_resume_skips_completed_phases(tmp_path: Path):
    repo = tmp_path / "repo"
    _write_file(repo / "src" / "main.py", "def main():\n    return 1\n")
    _write_file(repo / "src" / "core" / "util.py", "def util():\n    return 2\n")
    _write_file(repo / "README.md", "# Test\n")

    cache_dir = tmp_path / "cache"
    checkpoint_dir = tmp_path / "checkpoints"

    first = RepoOrchestrator(
        api_key="test",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        verbose=False,
        resume=True,
    )
    first_dummy = _DummySummarizer()
    first.summarizer = first_dummy
    first_result = asyncio.run(first.run(repo))

    assert first_dummy.file_calls == 2
    assert first_dummy.doc_calls == 1
    assert first_dummy.dir_calls >= 2
    assert first_dummy.module_calls == 1
    assert first_dummy.final_calls == 1

    second = RepoOrchestrator(
        api_key="test",
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        verbose=False,
        resume=True,
    )
    second_dummy = _DummySummarizer()
    second.summarizer = second_dummy
    second_result = asyncio.run(second.run(repo))

    assert second_dummy.file_calls == 0
    assert second_dummy.doc_calls == 0
    assert second_dummy.dir_calls == 0
    assert second_dummy.module_calls == 0
    assert second_dummy.final_calls == 1
    assert second_result.file_summaries == first_result.file_summaries
    assert second_result.module_summaries == first_result.module_summaries


def test_orchestrator_resume_skips_individual_file_nodes(tmp_path: Path):
    repo = tmp_path / "repo2"
    file_a = repo / "a.py"
    file_b = repo / "b.py"
    _write_file(file_a, "def a():\n    return 1\n")
    _write_file(file_b, "def b():\n    return 2\n")

    checkpoint_dir = tmp_path / "checkpoints"
    store = CheckpointStore(checkpoint_dir)
    state = CheckpointState(
        run_id="partial1",
        repo_path=str(repo.resolve()),
        started_at="2026-02-22T13:00:00Z",
        file_summaries={str(file_a.resolve()): "cached-a"},
    )
    store.save(state)

    orch = RepoOrchestrator(
        api_key="test",
        cache_dir=tmp_path / "cache2",
        checkpoint_dir=checkpoint_dir,
        verbose=False,
        resume=True,
    )
    dummy = _DummySummarizer()
    orch.summarizer = dummy
    result = asyncio.run(orch.run(repo))

    assert dummy.file_calls == 1
    assert str(file_a.resolve()) in result.file_summaries
    assert str(file_b.resolve()) in result.file_summaries
