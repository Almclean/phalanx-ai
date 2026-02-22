"""
agents.py — Async LLM agents for each summarization layer.

Design:
- L1 agents (function/class level): use claude-haiku-4-5 — cheap, fast, parallelized heavily
- L2 agents (file level): use claude-haiku-4-5 — still cheap, moderate context
- L3/L4 agents (dir/module level): use claude-sonnet-4-6 — needs more synthesis ability
- L5 agent (final synthesizer): use claude-sonnet-4-6 — the money shot

Rate limiting via a semaphore to avoid hammering the API.
"""

from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import anthropic

from cache import SummaryCache
from prompts import (
    l1_unit_prompt, l2_file_prompt, l3_directory_prompt,
    l4_module_prompt, l5_final_prompt, doc_file_prompt,
)
from parser import CodeUnit, FileUnits, DocFile


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

HAIKU  = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"


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
        cache_dir: Path = Path(".repo_summarizer_cache"),
        max_concurrent: int = 20,
        verbose: bool = False,
    ):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.cache = SummaryCache(cache_dir)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.tracker = CostTracker()
        self.verbose = verbose

    async def _call(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 512,
        cache_key: Optional[str] = None,
        layer: str = "unknown",
    ) -> str:
        # Cache check
        if cache_key is not None:
            cached = self.cache.get(cache_key, layer)
            if cached is not None:
                self.tracker.cache_hits += 1
                return cached

        async with self.semaphore:
            try:
                response = await self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
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
                await asyncio.sleep(30)
                return await self._call(prompt, model, max_tokens, cache_key, layer)
            except anthropic.APIError as e:
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
        )

    async def summarize_units_parallel(self, units: list[CodeUnit]) -> list[tuple[CodeUnit, str]]:
        """Summarize all units in a file concurrently."""
        tasks = [self.summarize_unit(u) for u in units]
        summaries = await asyncio.gather(*tasks)
        return list(zip(units, summaries))

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
    ) -> str:
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
        print(f"  L5 final synthesis for {repo_name}...")
        return await self._call(
            prompt,
            model=SONNET,
            max_tokens=2000,
            layer="l5",
        )
