import asyncio
from pathlib import Path

from agents import Summarizer
from parser import CodeUnit


def _unit(name: str, source: str) -> CodeUnit:
    return CodeUnit(
        name=name,
        kind="function",
        source=source,
        language="rust",
        file_path="src/lib.rs",
        start_line=1,
        end_line=3,
    )


def test_small_units_are_batched(tmp_path: Path):
    summarizer = Summarizer(
        api_key="test",
        cache_dir=tmp_path / "cache",
        l1_batch_size=8,
        l1_batch_threshold=500,
    )

    async def fake_call(*args, **kwargs):
        if kwargs.get("layer") == "l1_batch":
            return '["summary-a", "summary-b"]'
        raise AssertionError("Unexpected call path")

    summarizer._call = fake_call  # type: ignore[method-assign]

    units = [_unit("a", "fn a() {}"), _unit("b", "fn b() {}")]
    results = asyncio.run(summarizer.summarize_units_batched(units))
    summaries = [s for _, s in results]
    assert summaries == ["summary-a", "summary-b"]


def test_batch_parse_failure_falls_back_to_solo(tmp_path: Path):
    summarizer = Summarizer(
        api_key="test",
        cache_dir=tmp_path / "cache",
        l1_batch_size=8,
        l1_batch_threshold=500,
    )

    async def fake_call(*args, **kwargs):
        return "not-json"

    async def fake_summarize_unit(unit: CodeUnit) -> str:
        return f"solo:{unit.name}"

    summarizer._call = fake_call  # type: ignore[method-assign]
    summarizer.summarize_unit = fake_summarize_unit  # type: ignore[method-assign]

    units = [_unit("a", "fn a() {}"), _unit("b", "fn b() {}")]
    results = asyncio.run(summarizer.summarize_units_batched(units))
    summaries = [s for _, s in results]
    assert summaries == ["solo:a", "solo:b"]


def test_large_units_are_not_batched(tmp_path: Path):
    summarizer = Summarizer(
        api_key="test",
        cache_dir=tmp_path / "cache",
        l1_batch_size=8,
        l1_batch_threshold=20,
    )

    async def fake_call(*args, **kwargs):
        if kwargs.get("layer") == "l1_batch":
            raise AssertionError("Large units should not trigger batch path")
        return "unexpected"

    async def fake_summarize_unit(unit: CodeUnit) -> str:
        return f"solo:{unit.name}"

    summarizer._call = fake_call  # type: ignore[method-assign]
    summarizer.summarize_unit = fake_summarize_unit  # type: ignore[method-assign]

    units = [_unit("a", "x" * 50), _unit("b", "y" * 60)]
    results = asyncio.run(summarizer.summarize_units_batched(units))
    summaries = [s for _, s in results]
    assert summaries == ["solo:a", "solo:b"]
