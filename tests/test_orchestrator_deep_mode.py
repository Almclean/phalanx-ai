import asyncio
from dataclasses import dataclass
from pathlib import Path

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
        self.dir_chunk_calls = 0
        self.dir_merge_calls = 0
        self.module_calls = 0
        self.cluster_calls = 0
        self.cluster_merge_calls = 0
        self.final_calls = 0

    async def summarize_file(self, file_units):
        self.file_calls += 1
        return f"file:{Path(file_units.path).name}"

    async def summarize_doc_files_parallel(self, docs):
        self.doc_calls += len(docs)
        return [(d, "doc") for d in docs]

    async def summarize_directory(self, *, dir_path, file_summaries, subdirectory_summaries):
        self.dir_calls += 1
        return f"dir:{dir_path}"

    async def summarize_directory_chunk(self, *, dir_path, file_summaries_subset, chunk_index, total_chunks):
        self.dir_chunk_calls += 1
        return f"chunk:{dir_path}:{chunk_index}/{total_chunks}"

    async def merge_directory_chunks(self, *, dir_path, chunk_summaries):
        self.dir_merge_calls += 1
        return f"merged:{dir_path}:{len(chunk_summaries)}"

    async def summarize_module(self, *, module_name, directory_summaries):
        self.module_calls += 1
        return f"module:{module_name}"

    async def summarize_module_cluster(self, cluster_modules):
        self.cluster_calls += 1
        return f"cluster:{len(cluster_modules)}"

    async def merge_module_clusters(self, cluster_summaries):
        self.cluster_merge_calls += 1
        return f"cluster-merged:{len(cluster_summaries)}"

    async def synthesize_final(self, **kwargs):
        self.final_calls += 1
        return "final"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_deep_mode_uses_chunked_l3(tmp_path: Path):
    repo = tmp_path / "repo"
    for i in range(4):
        _write(repo / "src" / f"f{i}.py", f"def f{i}():\n    return {i}\n")

    orch = RepoOrchestrator(
        api_key="test",
        cache_dir=tmp_path / "cache",
        checkpoint_dir=tmp_path / "checkpoints",
        verbose=False,
        deep_mode_threshold=1,
        l3_chunk_size=2,
        l4_cluster_size=2,
    )
    dummy = _DummySummarizer()
    orch.summarizer = dummy
    asyncio.run(orch.run(repo))

    assert dummy.file_calls == 4
    assert dummy.dir_chunk_calls == 2
    assert dummy.dir_merge_calls == 1
    assert dummy.final_calls == 1


def test_dry_run_skips_all_summarizer_calls(tmp_path: Path):
    repo = tmp_path / "repo_dry"
    _write(repo / "a.py", "def a():\n    return 1\n")
    _write(repo / "README.md", "# docs\n")

    orch = RepoOrchestrator(
        api_key="test",
        cache_dir=tmp_path / "cache2",
        checkpoint_dir=tmp_path / "checkpoints2",
        verbose=False,
        dry_run=True,
    )
    dummy = _DummySummarizer()
    orch.summarizer = dummy
    result = asyncio.run(orch.run(repo))

    assert "Dry Run Estimate" in result.final_summary
    assert result.stats["dry_run"] is True
    assert dummy.file_calls == 0
    assert dummy.doc_calls == 0
    assert dummy.dir_calls == 0
    assert dummy.final_calls == 0


def test_dry_run_reports_excluded_directories(tmp_path: Path):
    repo = tmp_path / "repo_excluded"
    _write(repo / "app.py", "def app():\n    return 1\n")
    _write(repo / "vendor" / "lib.c", "int lib(void) { return 1; }\n")

    orch = RepoOrchestrator(
        api_key="test",
        cache_dir=tmp_path / "cache3",
        checkpoint_dir=tmp_path / "checkpoints3",
        verbose=False,
        dry_run=True,
    )
    dummy = _DummySummarizer()
    orch.summarizer = dummy
    result = asyncio.run(orch.run(repo))

    assert "vendor" in result.stats["excluded_directories"]
    assert "Excluded directories: 1" in result.final_summary
