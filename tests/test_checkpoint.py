from pathlib import Path

from checkpoint import CheckpointState, CheckpointStore


def _state(run_id: str, repo_path: Path, started_at: str) -> CheckpointState:
    return CheckpointState(
        run_id=run_id,
        repo_path=str(repo_path),
        started_at=started_at,
        file_summaries={"a.py": "summary"},
        dir_summaries={"src": "dir summary"},
        module_summaries=[{"name": "src", "summary": "module summary"}],
        doc_summaries=[{"path": "README.md", "summary": "doc summary"}],
    )


def test_checkpoint_save_load_roundtrip(tmp_path: Path):
    store = CheckpointStore(tmp_path / "checkpoints")
    state = _state("run1", tmp_path / "repo", "2026-02-22T12:00:00Z")
    state.mark_phase_complete("files")
    state.mark_phase_complete("docs")

    path = store.save(state)
    loaded = store.load(path)

    assert loaded is not None
    assert loaded.run_id == "run1"
    assert loaded.file_summaries == {"a.py": "summary"}
    assert loaded.phase_complete["files"] is True
    assert loaded.phase_complete["docs"] is True
    assert loaded.phase_complete["dirs"] is False


def test_checkpoint_load_handles_corrupt_json(tmp_path: Path):
    store = CheckpointStore(tmp_path / "checkpoints")
    bad = store.checkpoint_dir / "bad.json"
    bad.write_text('{"run_id": "oops",')

    assert store.load(bad) is None


def test_checkpoint_save_uses_atomic_replace_and_leaves_no_tmp(tmp_path: Path):
    store = CheckpointStore(tmp_path / "checkpoints")
    state = _state("run2", tmp_path / "repo", "2026-02-22T12:05:00Z")

    path = store.save(state)
    assert path.exists()
    assert not path.with_suffix(".tmp").exists()


def test_find_latest_returns_most_recent_for_repo(tmp_path: Path):
    store = CheckpointStore(tmp_path / "checkpoints")
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"

    older = _state("a1", repo_a, "2026-02-22T10:00:00Z")
    newer = _state("a2", repo_a, "2026-02-22T11:00:00Z")
    other = _state("b1", repo_b, "2026-02-22T12:00:00Z")

    path_older = store.save(older)
    path_newer = store.save(newer)
    store.save(other)

    assert store.find_latest(repo_a) == path_newer
    assert store.find_latest(repo_b) == store.checkpoint_path("b1")
    assert store.find_latest(tmp_path / "missing") is None
    assert path_older != path_newer
