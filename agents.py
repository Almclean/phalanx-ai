"""
agents.py — Async LLM agents for each summarization layer.

Design:
- L1 agents (function/class level): use claude-haiku-4-5 — cheap, fast, parallelized heavily
- L2 agents (file level): use claude-haiku-4-5 — still cheap, moderate context
- L3/L4 agents (dir/module level): use claude-sonnet-4-6 — needs more synthesis ability
- L5 agent (final synthesizer): use claude-sonnet-4-6 — highest-context synthesis

Rate limiting uses tiered semaphores:
- Haiku: higher concurrency
- Sonnet: lower concurrency
"""

from __future__ import annotations
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
import anthropic

from cache import SummaryCache
from prompts import (
    l1_unit_prompt, l2_file_prompt, l3_directory_prompt,
    l4_module_prompt, l5_final_prompt, doc_file_prompt,
    l3_chunk_prompt, l3_merge_prompt, l4_cluster_prompt, l4_final_merge_prompt, l5_tool_use_prompt,
)
from parser import CodeUnit, FileUnits, DocFile


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

HAIKU  = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"

L1_SYSTEM_PROMPT = """You are a senior engineer building a repository index by summarizing code units.
For each code unit you receive, write a concise 2-4 sentence technical summary covering:
1. What the unit does (its purpose and responsibility)
2. Key inputs, outputs, or side effects (if meaningful)
3. Any important patterns, algorithms, or gotchas worth noting
If a doc comment or docstring is provided, treat it as authoritative.
Be specific and technical. Do not start with "This <kind>...".
Output only the summary text, no headers or bullet points."""

L1_BATCH_SYSTEM_PROMPT = """You are a senior engineer building a repository index by summarizing code units.
For each UNIT block below, write a concise 2-3 sentence technical summary.
Rules:
- If a doc comment or docstring is provided, treat it as authoritative.
- Be specific and technical. Do not start with "This <kind>...".
- Output ONLY a JSON array of strings, one per unit, in order.
- Example: ["Summary one.", "Summary two."]"""


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

@dataclass
class CostTracker:
    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    cache_hits: int = 0

    # Approximate pricing per million tokens (Feb 2026 estimates)
    HAIKU_IN   = 0.80
    HAIKU_OUT  = 4.00
    SONNET_IN  = 3.00
    SONNET_OUT = 15.00

    def add(self, model: str, input_tok: int, output_tok: int):
        self.input_tokens += input_tok
        self.output_tokens += output_tok
        self.api_calls += 1

    def estimate_usd(self) -> float:
        # Rough blended estimate (assumes ~70% haiku, 30% sonnet)
        haiku_cost  = (self.input_tokens * 0.7 * self.HAIKU_IN  / 1_000_000
                     + self.output_tokens * 0.7 * self.HAIKU_OUT / 1_000_000)
        sonnet_cost = (self.input_tokens * 0.3 * self.SONNET_IN  / 1_000_000
                     + self.output_tokens * 0.3 * self.SONNET_OUT / 1_000_000)
        return haiku_cost + sonnet_cost

    def report(self) -> str:
        return (
            f"API calls: {self.api_calls} | Cache hits: {self.cache_hits} | "
            f"Tokens in: {self.input_tokens:,} | Tokens out: {self.output_tokens:,} | "
            f"Est. cost: ~${self.estimate_usd():.3f}"
        )


# ---------------------------------------------------------------------------
# Core LLM caller
# ---------------------------------------------------------------------------

class Summarizer:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        cache_dir: Path = Path(".repo_summarizer_cache"),
        max_concurrent: int = 20,
        haiku_concurrency: Optional[int] = None,
        sonnet_concurrency: Optional[int] = None,
        l1_batch_size: int = 8,
        l1_batch_threshold: int = 500,
        verbose: bool = False,
    ):
        if haiku_concurrency is None:
            haiku_concurrency = max_concurrent
        if sonnet_concurrency is None:
            sonnet_concurrency = max_concurrent

        self.client = client or anthropic.AsyncAnthropic(api_key=api_key)
        self.cache = SummaryCache(cache_dir)
        self.haiku_semaphore = asyncio.Semaphore(haiku_concurrency)
        self.sonnet_semaphore = asyncio.Semaphore(sonnet_concurrency)
        self.tracker = CostTracker()
        self.l1_batch_size = max(1, l1_batch_size)
        self.l1_batch_threshold = max(1, l1_batch_threshold)
        self.verbose = verbose

    def _semaphore_for_model(self, model: str) -> asyncio.Semaphore:
        return self.haiku_semaphore if model == HAIKU else self.sonnet_semaphore

    async def _call(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 512,
        cache_key: Optional[str] = None,
        layer: str = "unknown",
        system_prompt: Optional[str] = None,
    ) -> str:
        # Cache check
        if cache_key is not None:
            cached = self.cache.get(cache_key, layer)
            if cached is not None:
                self.tracker.cache_hits += 1
                return cached

        async with self._semaphore_for_model(model):
            try:
                request = dict(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                if system_prompt is not None:
                    request["system"] = system_prompt

                response = await self.client.messages.create(**request)
                text = response.content[0].text.strip()
                self.tracker.add(
                    model,
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                )
                if cache_key is not None:
                    self.cache.set(cache_key, layer, text)
                return text
            except anthropic.RateLimitError:
                if self.verbose:
                    print(f"  [{layer}] rate limited on {model}; retrying in 30s", flush=True)
                await asyncio.sleep(30)
                return await self._call(
                    prompt,
                    model,
                    max_tokens,
                    cache_key,
                    layer,
                    system_prompt=system_prompt,
                )
            except anthropic.APIError as e:
                if self.verbose:
                    print(f"  [{layer}] API error on {model}: {e}", flush=True)
                return f"[ERROR: {e}]"

    # -----------------------------------------------------------------------
    # L1: Unit-level
    # -----------------------------------------------------------------------

    async def summarize_unit(self, unit: CodeUnit) -> str:
        prompt = l1_unit_prompt(
            language=unit.language,
            file_path=unit.file_path,
            unit_kind=unit.kind,
            unit_name=unit.name,
            parent_name=unit.parent_name,
            source=unit.source,
            docstring=unit.docstring,
            doc_comment=unit.doc_comment,
        )
        if self.verbose:
            print(f"  L1 {unit.kind} `{unit.name}` in {Path(unit.file_path).name}")
        return await self._call(
            prompt,
            model=HAIKU,
            max_tokens=256,
            cache_key=unit.source,
            layer="l1",
            system_prompt=L1_SYSTEM_PROMPT,
        )

    def _build_l1_batch_prompt(self, units: list[CodeUnit]) -> str:
        blocks: list[str] = []
        for i, unit in enumerate(units, start=1):
            doc_parts = []
            if unit.doc_comment:
                doc_parts.append(f"Doc comment: {unit.doc_comment[:300]}")
            if unit.docstring:
                doc_parts.append(f"Docstring: {unit.docstring[:300]}")
            doc_block = ("\n" + "\n".join(doc_parts)) if doc_parts else ""
            blocks.append(
                f"### UNIT {i}: {unit.kind} `{unit.name}` [{unit.language} • {Path(unit.file_path).name}]"
                f"{doc_block}\n```{unit.language}\n{unit.source}\n```"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _parse_batch_response(text: str) -> Optional[list[str]]:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, list) or not all(isinstance(s, str) for s in parsed):
            return None
        return parsed

    async def _summarize_small_batch(self, batch: list[tuple[int, CodeUnit]]) -> list[tuple[int, str]]:
        units = [u for _, u in batch]
        prompt = self._build_l1_batch_prompt(units)
        cache_key = "|".join(u.source for u in units)
        response = await self._call(
            prompt,
            model=HAIKU,
            max_tokens=max(256, 140 * len(units)),
            cache_key=cache_key,
            layer="l1_batch",
            system_prompt=L1_BATCH_SYSTEM_PROMPT,
        )

        parsed = self._parse_batch_response(response)
        if parsed is None or len(parsed) != len(units):
            if self.verbose:
                print("  L1 batch parse/length mismatch; falling back to solo calls")
            solo_summaries = await asyncio.gather(*[self.summarize_unit(u) for u in units])
            return [(idx, summary) for (idx, _), summary in zip(batch, solo_summaries)]

        results: list[tuple[int, str]] = []
        for (idx, unit), summary in zip(batch, parsed):
            cleaned = summary.strip()
            if not cleaned or len(cleaned) > 500:
                if self.verbose:
                    print(f"  L1 batch invalid item for `{unit.name}`; retrying solo")
                cleaned = await self.summarize_unit(unit)
            results.append((idx, cleaned))
        return results

    async def summarize_units_batched(self, units: list[CodeUnit]) -> list[tuple[CodeUnit, str]]:
        indexed_units = list(enumerate(units))
        small = [(i, u) for i, u in indexed_units if len(u.source) <= self.l1_batch_threshold]
        large = [(i, u) for i, u in indexed_units if len(u.source) > self.l1_batch_threshold]

        summary_by_index: dict[int, str] = {}

        if large:
            large_summaries = await asyncio.gather(*[self.summarize_unit(u) for _, u in large])
            for (idx, _), summary in zip(large, large_summaries):
                summary_by_index[idx] = summary

        if small:
            batches = [
                small[i:i + self.l1_batch_size]
                for i in range(0, len(small), self.l1_batch_size)
            ]
            batch_results = await asyncio.gather(*[self._summarize_small_batch(b) for b in batches])
            for batch_result in batch_results:
                for idx, summary in batch_result:
                    summary_by_index[idx] = summary

        return [(units[i], summary_by_index[i]) for i in range(len(units))]

    async def summarize_units_parallel(self, units: list[CodeUnit]) -> list[tuple[CodeUnit, str]]:
        """Summarize all units in a file, batching small units where possible."""
        return await self.summarize_units_batched(units)

    # -----------------------------------------------------------------------
    # L2: File-level
    # -----------------------------------------------------------------------

    async def summarize_file(self, file_units: FileUnits) -> str:
        if not file_units.units:
            # File with no parseable units — summarize raw (truncated)
            excerpt = file_units.raw_source[:3000]
            prompt = (
                f"Summarize this {file_units.language} file in 2-3 sentences. "
                f"File: {file_units.path}\n\n```\n{excerpt}\n```"
            )
            return await self._call(
                prompt,
                model=HAIKU,
                max_tokens=200,
                cache_key=file_units.raw_source[:500],
                layer="l2_raw",
            )

        # Summarize all units in parallel first
        unit_results = await self.summarize_units_parallel(file_units.units)

        unit_summaries = [
            {
                "name": u.name,
                "kind": u.kind,
                "parent": u.parent_name,
                "summary": s,
            }
            for u, s in unit_results
        ]

        prompt = l2_file_prompt(
            file_path=file_units.path,
            language=file_units.language,
            unit_summaries=unit_summaries,
            file_line_count=file_units.raw_source.count("\n"),
        )
        if self.verbose:
            print(f"  L2 file: {Path(file_units.path).name} ({len(file_units.units)} units)")

        # Cache key: all unit summaries concatenated
        cache_key = "|".join(s for _, s in unit_results)
        return await self._call(
            prompt,
            model=HAIKU,
            max_tokens=400,
            cache_key=cache_key,
            layer="l2",
        )

    # -----------------------------------------------------------------------
    # L3: Directory-level
    # -----------------------------------------------------------------------

    async def summarize_directory(
        self,
        dir_path: str,
        file_summaries: list[dict],
        subdirectory_summaries: list[dict],
    ) -> str:
        prompt = l3_directory_prompt(
            dir_path=dir_path,
            file_summaries=file_summaries,
            subdirectory_summaries=subdirectory_summaries,
        )
        if self.verbose:
            print(f"  L3 directory: {dir_path}")

        cache_key = dir_path + "|" + "|".join(f["summary"] for f in file_summaries)
        return await self._call(
            prompt,
            model=SONNET,
            max_tokens=400,
            cache_key=cache_key,
            layer="l3",
        )

    async def summarize_directory_chunk(
        self,
        *,
        dir_path: str,
        file_summaries_subset: list[dict],
        chunk_index: int,
        total_chunks: int,
    ) -> str:
        prompt = l3_chunk_prompt(
            dir_path=dir_path,
            file_summaries_subset=file_summaries_subset,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
        )
        return await self._call(
            prompt,
            model=SONNET,
            max_tokens=400,
            layer="l3_chunk",
        )

    async def merge_directory_chunks(
        self,
        *,
        dir_path: str,
        chunk_summaries: list[str],
    ) -> str:
        prompt = l3_merge_prompt(
            dir_path=dir_path,
            chunk_summaries=chunk_summaries,
        )
        return await self._call(
            prompt,
            model=SONNET,
            max_tokens=450,
            layer="l3_merge",
        )

    # -----------------------------------------------------------------------
    # L4: Module-level
    # -----------------------------------------------------------------------

    async def summarize_module(
        self,
        module_name: str,
        directory_summaries: list[dict],
    ) -> str:
        prompt = l4_module_prompt(
            module_name=module_name,
            directory_summaries=directory_summaries,
        )
        if self.verbose:
            print(f"  L4 module: {module_name}")

        cache_key = module_name + "|" + "|".join(d["summary"] for d in directory_summaries)
        return await self._call(
            prompt,
            model=SONNET,
            max_tokens=500,
            cache_key=cache_key,
            layer="l4",
        )

    async def summarize_module_cluster(self, cluster_modules: list[dict]) -> str:
        prompt = l4_cluster_prompt(cluster_modules=cluster_modules)
        return await self._call(
            prompt,
            model=SONNET,
            max_tokens=450,
            layer="l4_cluster",
        )

    async def merge_module_clusters(self, cluster_summaries: list[str]) -> str:
        prompt = l4_final_merge_prompt(cluster_summaries=cluster_summaries)
        return await self._call(
            prompt,
            model=SONNET,
            max_tokens=600,
            layer="l4_cluster_merge",
        )

    # -----------------------------------------------------------------------
    # Doc file summarization
    # -----------------------------------------------------------------------

    async def summarize_doc_file(self, doc: DocFile) -> str:
        prompt = doc_file_prompt(
            file_path=doc.path,
            extension=doc.extension,
            content=doc.content,
        )
        if self.verbose:
            print(f"  DOC {Path(doc.path).name}")
        return await self._call(
            prompt,
            model=HAIKU,
            max_tokens=200,
            cache_key=doc.content[:500],
            layer="doc",
        )

    async def summarize_doc_files_parallel(self, docs: list[DocFile]) -> list[tuple[DocFile, str]]:
        tasks = [self.summarize_doc_file(d) for d in docs]
        summaries = await asyncio.gather(*tasks)
        return list(zip(docs, summaries))

    # -----------------------------------------------------------------------
    # L5: Final synthesizer
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_l5_tools() -> list[dict]:
        return [
            {
                "name": "list_modules",
                "description": "List available top-level module names.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "get_module_summary",
                "description": "Retrieve summary for a specific module.",
                "input_schema": {
                    "type": "object",
                    "properties": {"module_name": {"type": "string"}},
                    "required": ["module_name"],
                },
            },
            {
                "name": "get_file_summary",
                "description": "Retrieve summary for a specific source file path (exact or partial).",
                "input_schema": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            },
        ]

    @staticmethod
    def _l5_execute_tool(
        *,
        tool_name: str,
        tool_input: dict,
        module_summaries: list[dict],
        file_summaries: dict[str, str],
    ) -> dict:
        module_by_name = {
            str(m.get("name")): str(m.get("summary"))
            for m in module_summaries
            if isinstance(m, dict) and "name" in m and "summary" in m
        }

        if tool_name == "list_modules":
            return {"modules": sorted(module_by_name.keys())}

        if tool_name == "get_module_summary":
            requested = str(tool_input.get("module_name", "")).strip()
            if not requested:
                return {"error": "module_name is required"}
            if requested in module_by_name:
                return {"name": requested, "summary": module_by_name[requested]}
            lower = requested.lower()
            for name, summary in module_by_name.items():
                if lower in name.lower():
                    return {"name": name, "summary": summary}
            return {"error": f"module not found: {requested}"}

        if tool_name == "get_file_summary":
            requested = str(tool_input.get("file_path", "")).strip()
            if not requested:
                return {"error": "file_path is required"}
            if requested in file_summaries:
                return {"path": requested, "summary": file_summaries[requested]}
            lower = requested.lower()
            for path, summary in file_summaries.items():
                if lower in path.lower() or lower in Path(path).name.lower():
                    return {"path": path, "summary": summary}
            return {"error": f"file not found: {requested}"}

        return {"error": f"unknown tool: {tool_name}"}

    async def synthesize_final(
        self,
        *,
        repo_name: str,
        repo_structure: str,
        module_summaries: list[dict],
        languages: list[str],
        file_count: int,
        readme_excerpt: Optional[str],
        manifest_excerpt: Optional[str],
        doc_summaries: Optional[list[dict]] = None,
        file_summaries: Optional[dict[str, str]] = None,
    ) -> str:
        print(f"  L5 final synthesis for {repo_name}...")
        file_summaries = file_summaries or {}
        tools = self._build_l5_tools()
        system_prompt = l5_tool_use_prompt(
            repo_name=repo_name,
            languages=languages,
            file_count=file_count,
            repo_structure=repo_structure,
            readme_excerpt=readme_excerpt,
        )
        messages: list[dict] = [{
            "role": "user",
            "content": (
                "Write a 700-900 word report. Output markdown body only with no H1 title "
                "and no preamble. Start with `## Executive Summary` and use exactly these H2 "
                "sections in order: Executive Summary, System Architecture, Key Components, "
                "Operational Model, Risks & Limitations, Testing Coverage Snapshot."
            ),
        }]

        max_tool_calls = 30
        tool_call_count = 0

        while True:
            async with self._semaphore_for_model(SONNET):
                response = await self.client.messages.create(
                    model=SONNET,
                    max_tokens=2000,
                    system=system_prompt,
                    tools=tools,
                    messages=messages,
                )

            self.tracker.add(
                SONNET,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )

            if getattr(response, "stop_reason", None) != "tool_use":
                final_chunks: list[str] = []
                for block in response.content:
                    if getattr(block, "type", "") != "text":
                        continue
                    txt = getattr(block, "text", "").strip()
                    if txt:
                        final_chunks.append(txt)
                if final_chunks:
                    return "\n\n".join(final_chunks).strip()
                break

            messages.append({"role": "assistant", "content": response.content})
            tool_result_blocks: list[dict] = []
            for block in response.content:
                if getattr(block, "type", "") != "tool_use":
                    continue
                tool_call_count += 1
                result = self._l5_execute_tool(
                    tool_name=getattr(block, "name", ""),
                    tool_input=getattr(block, "input", {}) or {},
                    module_summaries=module_summaries,
                    file_summaries=file_summaries,
                )
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": getattr(block, "id", ""),
                        "content": json.dumps(result),
                    }
                )

            messages.append({"role": "user", "content": tool_result_blocks})
            if tool_call_count >= max_tool_calls:
                messages.append(
                    {
                        "role": "user",
                        "content": "You have enough context. Produce the final summary now.",
                    }
                )
                break

        # Fallback to the original non-tool prompt if tool loop exits without text.
        prompt = l5_final_prompt(
            repo_name=repo_name,
            repo_structure=repo_structure,
            module_summaries=module_summaries,
            languages=languages,
            file_count=file_count,
            readme_excerpt=readme_excerpt,
            manifest_excerpt=manifest_excerpt,
            doc_summaries=doc_summaries,
        )
        return await self._call(
            prompt,
            model=SONNET,
            max_tokens=2000,
            layer="l5",
        )

    async def summarize_diff_digest(self, prompt: str) -> str:
        return await self._call(
            prompt,
            model=SONNET,
            max_tokens=900,
            layer="diff_digest",
        )
