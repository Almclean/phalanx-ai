import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from agents import Summarizer


@dataclass
class _FakeResponse:
    stop_reason: str
    content: list
    usage: object


class _FakeMessagesAPI:
    def __init__(self, responses: list[_FakeResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse]):
        self.messages = _FakeMessagesAPI(responses)


def _tool_use_block(tool_id: str, name: str, payload: dict):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=payload)


def _text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def _usage():
    return SimpleNamespace(input_tokens=10, output_tokens=5)


def test_l5_tool_use_loop_returns_final_text(tmp_path: Path):
    responses = [
        _FakeResponse("tool_use", [_tool_use_block("t1", "list_modules", {})], _usage()),
        _FakeResponse("tool_use", [_tool_use_block("t2", "get_module_summary", {"module_name": "src"})], _usage()),
        _FakeResponse("end_turn", [_text_block("Final synthesized summary.")], _usage()),
    ]
    fake_client = _FakeClient(responses)
    summarizer = Summarizer(
        api_key="test",
        client=fake_client,
        cache_dir=tmp_path / "cache",
        verbose=False,
    )

    result = asyncio.run(
        summarizer.synthesize_final(
            repo_name="repo",
            repo_structure="repo/\n  src/",
            module_summaries=[{"name": "src", "summary": "Source module summary"}],
            languages=["python"],
            file_count=2,
            readme_excerpt=None,
            manifest_excerpt=None,
            doc_summaries=None,
            file_summaries={"src/main.py": "Main file summary"},
        )
    )

    assert result == "Final synthesized summary."
    assert summarizer.tracker.api_calls == 3
    assert len(fake_client.messages.calls) == 3
    # By call 2, a tool result message should have been added to the transcript.
    assert any(m.get("role") == "user" and isinstance(m.get("content"), list) for m in fake_client.messages.calls[1]["messages"])


def test_l5_tool_executor_file_lookup_partial_match(tmp_path: Path):
    summarizer = Summarizer(api_key="test", cache_dir=tmp_path / "cache")
    payload = summarizer._l5_execute_tool(
        tool_name="get_file_summary",
        tool_input={"file_path": "main.py"},
        module_summaries=[],
        file_summaries={"src/main.py": "summary-main"},
    )
    assert payload["path"] == "src/main.py"
    assert payload["summary"] == "summary-main"
