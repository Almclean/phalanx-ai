"""
orchestrator.py — DAG execution engine for the recursive summarization pipeline.

Execution order:
  1. Discover all files in the repo
  2. Parse all files to extract code units (CPU-bound, done synchronously)
  3. L1+L2: Summarize all files in parallel (I/O-bound, async)
  4. L3: Summarize directories bottom-up (must wait for all children)
  5. L4: Summarize top-level modules
  6. L5: Final synthesis

The directory tree is processed bottom-up so each directory summary
has access to its children's summaries before being computed.
"""

from __future__ import annotations
import asyncio
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from parser import discover_files, discover_doc_files, parse_file, parse_doc_file, FileUnits, DocFile
from agents import Summarizer


# ---------------------------------------------------------------------------
# Repo metadata helpers
# ---------------------------------------------------------------------------

def _read_readme(root: Path) -> Optional[str]:
    for name in ("README.md", "README.rst", "README.txt", "README"):
        p = root / name
        if p.exists():
            try:
                return p.read_text(errors="replace")[:3000]
            except OSError:
                pass
    return None


def _read_manifest(root: Path) -> Optional[str]:
    for name in (
        "pyproject.toml", "setup.py", "requirements.txt",
        "package.json", "Cargo.toml", "go.mod", "go.sum",
        "Gemfile", "pom.xml",
    ):
        p = root / name
        if p.exists():
            try:
                return p.read_text(errors="replace")[:1500]
            except OSError:
                pass
    return None


def _build_tree_string(root: Path, files: list[Path], max_lines: int = 80) -> str:
    """Build a compact directory tree string from file list."""
    # Collect unique dirs
    dirs: set[str] = set()
    for f in files:
        rel = f.relative_to(root)
        for parent in rel.parents:
            if str(parent) != ".":
                dirs.add(str(parent))

    lines = [str(root.name) + "/"]
    seen_dirs: set[str] = set()

    for f in sorted(files):
        rel = f.relative_to(root)
        parts = rel.parts
        for i, part in enumerate(parts[:-1]):
            dir_key = "/".join(parts[:i+1])
            if dir_key not in seen_dirs:
                seen_dirs.add(dir_key)
                lines.append("  " * (i+1) + part + "/")
        lines.append("  " * len(parts) + parts[-1])
        if len(lines) > max_lines:
            lines.append("  ... [truncated]")
            break

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SummaryResult:
    repo_name: str
    final_summary: str
    module_summaries: list[dict]
    directory_summaries: dict[str, str]
    file_summaries: dict[str, str]
    doc_summaries: list[dict]
    stats: dict


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class RepoOrchestrator:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        max_concurrent: int = 20,
        verbose: bool = True,
        exclude_dirs: Optional[set[str]] = None,
        max_files: int = 2000,
        skip_docs: bool = False,
    ):
        self.verbose = verbose
        self.exclude_dirs = exclude_dirs
        self.max_files = max_files
        self.skip_docs = skip_docs

        if cache_dir is None:
            cache_dir = Path.home() / ".repo_summarizer_cache"

        self.summarizer = Summarizer(
            api_key=api_key,
            cache_dir=cache_dir,
            max_concurrent=max_concurrent,
            verbose=verbose,
        )

    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    # -----------------------------------------------------------------------
    # Phase 1+2: Discovery and parsing
    # -----------------------------------------------------------------------

    def _discover_and_parse(self, root: Path) -> tuple[list[FileUnits], list[Path]]:
        self._log(f"\n[Phase 1] Discovering source files in {root}...")
        files = discover_files(root, self.exclude_dirs)
        self._log(f"  Found {len(files)} source files")

        if len(files) > self.max_files:
            raise ValueError(
                f"Repository contains {len(files)} source files, which exceeds the "
                f"--max-files limit of {self.max_files}. This would be expensive to summarize. "
                f"Use --max-files {len(files)} to override, or --exclude-dir to narrow scope."
            )

        self._log(f"\n[Phase 2] Parsing ASTs ({len(files)} files)...")
        file_units_list: list[FileUnits] = []
        errors = 0
        for i, f in enumerate(files):
            fu = parse_file(f)
            if fu is not None:
                file_units_list.append(fu)
                if fu.parse_error:
                    errors += 1
            if self.verbose and (i + 1) % 20 == 0:
                print(f"  Parsed {i+1}/{len(files)}...", flush=True)

        self._log(f"  Parsed {len(file_units_list)} files ({errors} with errors)")
        return file_units_list, files

    # -----------------------------------------------------------------------
    # Phase 3: File-level summarization (L1+L2 in parallel)
    # -----------------------------------------------------------------------

    async def _summarize_files(
        self, file_units_list: list[FileUnits]
    ) -> dict[str, str]:
        self._log(f"\n[Phase 3] Summarizing {len(file_units_list)} files (L1+L2, parallel)...")
        start = asyncio.get_event_loop().time()

        tasks = {fu.path: self.summarizer.summarize_file(fu) for fu in file_units_list}
        results = await asyncio.gather(*tasks.values())
        file_summaries = dict(zip(tasks.keys(), results))

        elapsed = asyncio.get_event_loop().time() - start
        self._log(f"  Done in {elapsed:.1f}s")
        return file_summaries

    # -----------------------------------------------------------------------
    # Phase 4: Directory summarization (L3, bottom-up)
    # -----------------------------------------------------------------------

    async def _summarize_directories(
        self,
        root: Path,
        file_summaries: dict[str, str],
        all_files: list[Path],
    ) -> dict[str, str]:
        self._log(f"\n[Phase 4] Summarizing directories (L3, bottom-up)...")

        # Group files by their immediate parent directory (relative to root)
        dir_to_files: dict[str, list[str]] = defaultdict(list)
        for f in all_files:
            try:
                rel = f.relative_to(root)
                # Use the immediate parent relative to root
                parent = str(rel.parent) if str(rel.parent) != "." else "__root__"
                dir_to_files[parent].append(str(f))
            except ValueError:
                pass

        # Build a set of all directories (including nested)
        all_dirs: set[str] = set()
        for f in all_files:
            try:
                rel = f.relative_to(root)
                for parent in rel.parents:
                    s = str(parent)
                    if s and s != ".":
                        all_dirs.add(s)
            except ValueError:
                pass

        # Sort directories by depth (deepest first — bottom-up)
        sorted_dirs = sorted(all_dirs, key=lambda d: d.count(os.sep), reverse=True)

        dir_summaries: dict[str, str] = {}

        for dir_rel in sorted_dirs:
            dir_abs = root / dir_rel
            files_in_dir = dir_to_files.get(dir_rel, [])

            file_summ_list = [
                {"name": Path(fp).name, "summary": file_summaries.get(fp, "[no summary]")}
                for fp in files_in_dir
                if fp in file_summaries
            ]

            # Immediate subdirectories
            subdir_summ_list = [
                {"name": Path(sd).name, "summary": dir_summaries[sd]}
                for sd in dir_summaries
                if Path(sd).parent == Path(dir_rel) and sd in dir_summaries
            ]

            if not file_summ_list and not subdir_summ_list:
                continue

            summary = await self.summarizer.summarize_directory(
                dir_path=dir_rel,
                file_summaries=file_summ_list,
                subdirectory_summaries=subdir_summ_list,
            )
            dir_summaries[dir_rel] = summary

        # Also handle root-level files
        root_files = dir_to_files.get("__root__", [])
        if root_files:
            file_summ_list = [
                {"name": Path(fp).name, "summary": file_summaries.get(fp, "[no summary]")}
                for fp in root_files
                if fp in file_summaries
            ]
            if file_summ_list:
                subdir_summ_list = [
                    {"name": Path(sd).name, "summary": dir_summaries[sd]}
                    for sd in dir_summaries
                    if "/" not in sd  # only top-level dirs
                ]
                summary = await self.summarizer.summarize_directory(
                    dir_path="[root]",
                    file_summaries=file_summ_list,
                    subdirectory_summaries=subdir_summ_list,
                )
                dir_summaries["__root__"] = summary

        self._log(f"  Summarized {len(dir_summaries)} directories")
        return dir_summaries

    # -----------------------------------------------------------------------
    # Phase 5: Module-level summarization (L4)
    # -----------------------------------------------------------------------

    async def _summarize_modules(
        self,
        root: Path,
        dir_summaries: dict[str, str],
    ) -> list[dict]:
        self._log(f"\n[Phase 5] Summarizing modules (L4)...")

        # Top-level directories are the "modules"
        top_level_dirs = {
            k: v for k, v in dir_summaries.items()
            if "/" not in k and k != "__root__"
        }

        if not top_level_dirs:
            # Flat repo — use root summary as the only module
            root_summary = dir_summaries.get("__root__", "")
            return [{"name": root.name, "summary": root_summary}]

        module_summaries: list[dict] = []

        for top_dir, top_summary in sorted(top_level_dirs.items()):
            # Collect all child directory summaries for this module
            child_dirs = [
                {"name": Path(k).name, "summary": v}
                for k, v in dir_summaries.items()
                if k.startswith(top_dir + os.sep)
            ]

            if child_dirs:
                summary = await self.summarizer.summarize_module(
                    module_name=top_dir,
                    directory_summaries=[{"name": top_dir, "summary": top_summary}] + child_dirs,
                )
            else:
                summary = top_summary  # single dir = module, use dir summary directly

            module_summaries.append({"name": top_dir, "summary": summary})

        # Include root-level files module if it exists
        if "__root__" in dir_summaries:
            module_summaries.insert(0, {
                "name": f"{root.name} [root files]",
                "summary": dir_summaries["__root__"],
            })

        self._log(f"  Summarized {len(module_summaries)} modules")
        return module_summaries

    # -----------------------------------------------------------------------
    # Phase 3.5: Doc file summarization (parallel with file summarization)
    # -----------------------------------------------------------------------

    async def _summarize_doc_files(self, root: Path) -> list[dict]:
        if self.skip_docs:
            return []

        doc_paths = discover_doc_files(root, self.exclude_dirs)
        if not doc_paths:
            return []

        self._log(f"\n[Phase 3b] Summarizing {len(doc_paths)} doc/config files...")
        docs: list[DocFile] = []
        for p in doc_paths:
            df = parse_doc_file(p)
            if df is not None:
                docs.append(df)

        if not docs:
            return []

        results = await self.summarizer.summarize_doc_files_parallel(docs)
        return [{"path": d.path, "summary": s} for d, s in results]

    # -----------------------------------------------------------------------
    # Phase 6: Final synthesis (L5)
    # -----------------------------------------------------------------------

    async def _final_synthesis(
        self,
        *,
        root: Path,
        all_files: list[Path],
        file_units_list: list[FileUnits],
        module_summaries: list[dict],
        doc_summaries: list[dict],
    ) -> str:
        self._log(f"\n[Phase 6] Final synthesis (L5)...")

        languages = sorted(set(fu.language for fu in file_units_list))
        tree_str = _build_tree_string(root, all_files)
        readme = _read_readme(root)
        manifest = _read_manifest(root)

        return await self.summarizer.synthesize_final(
            repo_name=root.name,
            repo_structure=tree_str,
            module_summaries=module_summaries,
            languages=languages,
            file_count=len(file_units_list),
            readme_excerpt=readme,
            manifest_excerpt=manifest,
            doc_summaries=doc_summaries if doc_summaries else None,
        )

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    async def run(self, repo_path: Path) -> SummaryResult:
        root = repo_path.resolve()
        if not root.exists():
            raise ValueError(f"Path does not exist: {root}")

        # Phase 1+2: Discover and parse
        file_units_list, all_files = self._discover_and_parse(root)

        if not file_units_list:
            raise ValueError("No parseable source files found in the repository.")

        # Phase 3: File summaries (L1+L2) + doc files in parallel
        file_summaries, doc_summaries = await asyncio.gather(
            self._summarize_files(file_units_list),
            self._summarize_doc_files(root),
        )

        # Phase 4: Directory summaries (L3)
        dir_summaries = await self._summarize_directories(root, file_summaries, all_files)

        # Phase 5: Module summaries (L4)
        module_summaries = await self._summarize_modules(root, dir_summaries)

        # Phase 6: Final synthesis (L5)
        final_summary = await self._final_synthesis(
            root=root,
            all_files=all_files,
            file_units_list=file_units_list,
            module_summaries=module_summaries,
            doc_summaries=doc_summaries,
        )

        # Compile stats
        stats = {
            "files_analyzed": len(file_units_list),
            "directories": len(dir_summaries),
            "modules": len(module_summaries),
            "languages": sorted(set(fu.language for fu in file_units_list)),
            "total_units": sum(len(fu.units) for fu in file_units_list),
            "doc_files_summarized": len(doc_summaries),
            **vars(self.summarizer.tracker),
        }

        self._log(f"\n[Done] {self.summarizer.tracker.report()}")

        return SummaryResult(
            repo_name=root.name,
            final_summary=final_summary,
            module_summaries=module_summaries,
            directory_summaries=dir_summaries,
            file_summaries=file_summaries,
            doc_summaries=doc_summaries,
            stats=stats,
        )
