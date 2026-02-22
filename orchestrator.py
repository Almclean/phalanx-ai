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
import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from parser import discover_files, discover_doc_files, parse_file, parse_doc_file, FileUnits, DocFile
from agents import Summarizer
from checkpoint import CheckpointState, CheckpointStore, now_iso_utc


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
        haiku_concurrency: Optional[int] = None,
        sonnet_concurrency: Optional[int] = None,
        verbose: bool = True,
        exclude_dirs: Optional[set[str]] = None,
        max_files: int = 10000,
        skip_docs: bool = False,
        deep_mode_threshold: int = 500,
        l3_chunk_size: int = 15,
        l4_cluster_size: int = 8,
        l1_batch_size: int = 8,
        l1_batch_threshold: int = 500,
        dry_run: bool = False,
        checkpoint_dir: Optional[Path] = None,
        resume: bool = True,
    ):
        self.verbose = verbose
        self.exclude_dirs = exclude_dirs
        self.max_files = max_files
        self.skip_docs = skip_docs
        self.deep_mode_threshold = deep_mode_threshold
        self.l3_chunk_size = max(1, l3_chunk_size)
        self.l4_cluster_size = max(1, l4_cluster_size)
        self.dry_run = dry_run
        self.resume = resume

        if cache_dir is None:
            cache_dir = Path.home() / ".repo_summarizer_cache"
        self.cache_dir = cache_dir

        if haiku_concurrency is None:
            haiku_concurrency = max_concurrent
        if sonnet_concurrency is None:
            sonnet_concurrency = max_concurrent

        self.summarizer = Summarizer(
            api_key=api_key,
            cache_dir=cache_dir,
            max_concurrent=max_concurrent,
            haiku_concurrency=haiku_concurrency,
            sonnet_concurrency=sonnet_concurrency,
            l1_batch_size=l1_batch_size,
            l1_batch_threshold=l1_batch_threshold,
            verbose=verbose,
        )
        if checkpoint_dir is None:
            checkpoint_dir = self.cache_dir / "checkpoints"
        self.checkpoints: Optional[CheckpointStore] = None if dry_run else CheckpointStore(checkpoint_dir)

    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    @staticmethod
    def _chunked(items: list, size: int) -> list[list]:
        return [items[i:i + size] for i in range(0, len(items), size)]

    @staticmethod
    def _new_run_id(root: Path) -> str:
        seed = f"{root.resolve()}|{now_iso_utc()}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]

    def _load_or_create_checkpoint(self, root: Path) -> CheckpointState:
        if self.checkpoints is None:
            raise RuntimeError("Checkpoint store is not initialized")

        if self.resume:
            latest = self.checkpoints.find_latest(root)
            if latest is not None:
                state = self.checkpoints.load(latest)
                if state is not None:
                    self._log(f"  Resuming from checkpoint: {latest.name}")
                    return state

        state = CheckpointState(
            run_id=self._new_run_id(root),
            repo_path=str(root),
            started_at=now_iso_utc(),
        )
        self.checkpoints.save(state)
        return state

    def _estimate_dry_run(
        self,
        *,
        root: Path,
        all_files: list[Path],
        file_units_list: list[FileUnits],
    ) -> tuple[str, dict]:
        languages = sorted(set(fu.language for fu in file_units_list))
        total_units = sum(len(fu.units) for fu in file_units_list)
        unit_sources = [u.source for fu in file_units_list for u in fu.units]
        l1_batch_threshold = getattr(self.summarizer, "l1_batch_threshold", 500)
        l1_batch_size = getattr(self.summarizer, "l1_batch_size", 8)
        small_units = sum(1 for s in unit_sources if len(s) <= l1_batch_threshold)
        large_units = total_units - small_units
        l1_calls = large_units + (small_units + l1_batch_size - 1) // l1_batch_size
        l2_calls = len(file_units_list)

        dir_to_files: dict[str, int] = defaultdict(int)
        for f in all_files:
            try:
                rel = f.relative_to(root)
                parent = str(rel.parent) if str(rel.parent) != "." else "__root__"
                dir_to_files[parent] += 1
            except ValueError:
                pass

        directory_count = len({d for d in dir_to_files if d != "__root__"}) + (1 if "__root__" in dir_to_files else 0)
        l3_calls = 0
        for d, count in dir_to_files.items():
            if d == "__root__":
                continue
            chunk_count = max(1, (count + self.l3_chunk_size - 1) // self.l3_chunk_size)
            l3_calls += chunk_count + (1 if chunk_count > 1 else 0)
        if "__root__" in dir_to_files:
            l3_calls += 1

        top_level_dirs = {
            str(p.relative_to(root).parts[0])
            for p in all_files
            if len(p.relative_to(root).parts) > 1
        }
        module_count = max(1, len(top_level_dirs))
        if module_count > self.l4_cluster_size:
            cluster_count = (module_count + self.l4_cluster_size - 1) // self.l4_cluster_size
            l4_calls = cluster_count + 1
        else:
            l4_calls = module_count

        doc_files = 0 if self.skip_docs else len(discover_doc_files(root, self.exclude_dirs))
        l5_calls = 1

        # Rough token heuristics for a planning estimate only.
        est_in = int(total_units * 110 + l2_calls * 350 + l3_calls * 320 + l4_calls * 420 + l5_calls * 5000 + doc_files * 220)
        est_out = int(total_units * 75 + l2_calls * 120 + l3_calls * 90 + l4_calls * 120 + l5_calls * 1200 + doc_files * 60)
        est_cost = (
            est_in * 0.7 * 0.80 / 1_000_000
            + est_out * 0.7 * 4.00 / 1_000_000
            + est_in * 0.3 * 3.00 / 1_000_000
            + est_out * 0.3 * 15.00 / 1_000_000
        )

        report = (
            "## Dry Run Estimate\n\n"
            f"- Repository: `{root.name}`\n"
            f"- Files analyzed: {len(file_units_list)}\n"
            f"- Code units: {total_units}\n"
            f"- Languages: {', '.join(languages)}\n"
            f"- Estimated calls: L1={l1_calls}, L2={l2_calls}, Docs={doc_files}, L3={l3_calls}, L4={l4_calls}, L5={l5_calls}\n"
            f"- Estimated tokens: in~{est_in:,}, out~{est_out:,}\n"
            f"- Estimated cost: ~${est_cost:.2f}\n"
        )
        stats = {
            "dry_run": True,
            "files_analyzed": len(file_units_list),
            "directories": directory_count,
            "modules": module_count,
            "languages": languages,
            "total_units": total_units,
            "doc_files_summarized": doc_files,
            "estimated_calls": {
                "l1": l1_calls,
                "l2": l2_calls,
                "docs": doc_files,
                "l3": l3_calls,
                "l4": l4_calls,
                "l5": l5_calls,
            },
            "estimated_input_tokens": est_in,
            "estimated_output_tokens": est_out,
            "estimated_cost_usd": round(est_cost, 2),
            **vars(self.summarizer.tracker),
        }
        return report, stats

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
        self,
        file_units_list: list[FileUnits],
        *,
        existing: Optional[dict[str, str]] = None,
    ) -> dict[str, str]:
        existing_map = dict(existing or {})
        pending = [fu for fu in file_units_list if fu.path not in existing_map]
        self._log(f"\n[Phase 3] Summarizing {len(pending)} files (L1+L2, parallel)...")

        if not pending:
            self._log("  No pending file summaries (all loaded from checkpoint)")
            return existing_map

        start = asyncio.get_event_loop().time()

        tasks = {fu.path: self.summarizer.summarize_file(fu) for fu in pending}
        results = await asyncio.gather(*tasks.values())
        existing_map.update(dict(zip(tasks.keys(), results)))

        elapsed = asyncio.get_event_loop().time() - start
        self._log(f"  Done in {elapsed:.1f}s")
        return existing_map

    async def _summarize_directory_chunked(
        self,
        *,
        dir_path: str,
        file_summaries: list[dict],
        subdirectory_summaries: list[dict],
    ) -> str:
        chunks = self._chunked(file_summaries, self.l3_chunk_size)
        chunk_tasks = [
            self.summarizer.summarize_directory_chunk(
                dir_path=dir_path,
                file_summaries_subset=chunk,
                chunk_index=i + 1,
                total_chunks=len(chunks),
            )
            for i, chunk in enumerate(chunks)
        ]
        chunk_summaries = await asyncio.gather(*chunk_tasks)

        if len(chunk_summaries) == 1:
            merged_files_summary = chunk_summaries[0]
        else:
            merged_files_summary = await self.summarizer.merge_directory_chunks(
                dir_path=dir_path,
                chunk_summaries=chunk_summaries,
            )

        if not subdirectory_summaries:
            return merged_files_summary

        return await self.summarizer.summarize_directory(
            dir_path=dir_path,
            file_summaries=[{"name": "[chunked files]", "summary": merged_files_summary}],
            subdirectory_summaries=subdirectory_summaries,
        )

    # -----------------------------------------------------------------------
    # Phase 4: Directory summarization (L3, bottom-up)
    # -----------------------------------------------------------------------

    async def _summarize_directories(
        self,
        root: Path,
        file_summaries: dict[str, str],
        all_files: list[Path],
        *,
        deep_mode: bool = False,
        existing: Optional[dict[str, str]] = None,
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

        dir_summaries: dict[str, str] = dict(existing or {})

        for dir_rel in sorted_dirs:
            if dir_rel in dir_summaries:
                continue
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

            if deep_mode and len(file_summ_list) > self.l3_chunk_size:
                summary = await self._summarize_directory_chunked(
                    dir_path=dir_rel,
                    file_summaries=file_summ_list,
                    subdirectory_summaries=subdir_summ_list,
                )
            else:
                summary = await self.summarizer.summarize_directory(
                    dir_path=dir_rel,
                    file_summaries=file_summ_list,
                    subdirectory_summaries=subdir_summ_list,
                )
            dir_summaries[dir_rel] = summary

        # Also handle root-level files
        root_files = dir_to_files.get("__root__", [])
        if root_files and "__root__" not in dir_summaries:
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
        *,
        existing: Optional[list[dict]] = None,
    ) -> list[dict]:
        self._log(f"\n[Phase 5] Summarizing modules (L4)...")
        existing_by_name = {
            m.get("name"): m.get("summary")
            for m in (existing or [])
            if isinstance(m, dict) and "name" in m and "summary" in m
        }

        # Top-level directories are the "modules"
        top_level_dirs = {
            k: v for k, v in dir_summaries.items()
            if "/" not in k and k != "__root__"
        }

        if not top_level_dirs:
            # Flat repo — use root summary as the only module
            if root.name in existing_by_name:
                return [{"name": root.name, "summary": existing_by_name[root.name]}]
            root_summary = dir_summaries.get("__root__", "")
            return [{"name": root.name, "summary": root_summary}]

        module_summaries: list[dict] = []

        for top_dir, top_summary in sorted(top_level_dirs.items()):
            if top_dir in existing_by_name:
                module_summaries.append({"name": top_dir, "summary": existing_by_name[top_dir]})
                continue

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
            root_module_name = f"{root.name} [root files]"
            if root_module_name in existing_by_name:
                module_summaries.insert(0, {"name": root_module_name, "summary": existing_by_name[root_module_name]})
                self._log(f"  Summarized {len(module_summaries)} modules")
                return module_summaries
            module_summaries.insert(0, {
                "name": root_module_name,
                "summary": dir_summaries["__root__"],
            })

        self._log(f"  Summarized {len(module_summaries)} modules")
        return module_summaries

    async def _summarize_modules_clustered(
        self,
        root: Path,
        dir_summaries: dict[str, str],
        *,
        existing: Optional[list[dict]] = None,
    ) -> list[dict]:
        self._log(f"\n[Phase 5] Summarizing modules (L4, clustered)...")

        root_module: list[dict] = []
        if "__root__" in dir_summaries:
            root_module = [{
                "name": f"{root.name} [root files]",
                "summary": dir_summaries["__root__"],
            }]

        base_modules = [
            {"name": k, "summary": v}
            for k, v in sorted(dir_summaries.items())
            if "/" not in k and k != "__root__"
        ]
        if not base_modules:
            root_summary = dir_summaries.get("__root__", "")
            return [{"name": root.name, "summary": root_summary}]

        if len(base_modules) <= self.l4_cluster_size:
            return await self._summarize_modules(root, dir_summaries, existing=existing)

        clusters = self._chunked(base_modules, self.l4_cluster_size)
        cluster_summaries = await asyncio.gather(
            *[self.summarizer.summarize_module_cluster(cluster) for cluster in clusters]
        )
        merged = await self.summarizer.merge_module_clusters(cluster_summaries)

        cluster_entries = [
            {"name": f"[cluster {i + 1}]", "summary": summary}
            for i, summary in enumerate(cluster_summaries)
        ]
        # Include an overall merged module context entry for L5.
        cluster_entries.append({"name": "[all clusters]", "summary": merged})
        result = root_module + cluster_entries
        self._log(f"  Summarized {len(result)} clustered module entries")
        return result

    # -----------------------------------------------------------------------
    # Phase 3.5: Doc file summarization (parallel with file summarization)
    # -----------------------------------------------------------------------

    async def _summarize_doc_files(
        self,
        root: Path,
        *,
        existing: Optional[list[dict]] = None,
    ) -> list[dict]:
        if self.skip_docs:
            return []

        doc_paths = discover_doc_files(root, self.exclude_dirs)
        if not doc_paths:
            return []

        existing_by_path = {
            d.get("path"): d.get("summary")
            for d in (existing or [])
            if isinstance(d, dict) and "path" in d and "summary" in d
        }
        self._log(f"\n[Phase 3b] Summarizing doc/config files...")
        docs: list[DocFile] = []
        for p in doc_paths:
            df = parse_doc_file(p)
            if df is not None:
                docs.append(df)

        if not docs:
            return []

        pending_docs = [d for d in docs if d.path not in existing_by_path]
        self._log(f"  Pending doc files: {len(pending_docs)} / {len(docs)}")
        if pending_docs:
            results = await self.summarizer.summarize_doc_files_parallel(pending_docs)
            for d, s in results:
                existing_by_path[d.path] = s

        return [
            {"path": d.path, "summary": existing_by_path[d.path]}
            for d in docs
            if d.path in existing_by_path
        ]

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
        file_summaries: dict[str, str],
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
            file_summaries=file_summaries,
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

        deep_mode = len(file_units_list) >= self.deep_mode_threshold
        if deep_mode:
            self._log(
                f"\n[Mode] Deep mode enabled: {len(file_units_list)} files >= "
                f"threshold {self.deep_mode_threshold}"
            )

        if self.dry_run:
            report, stats = self._estimate_dry_run(
                root=root,
                all_files=all_files,
                file_units_list=file_units_list,
            )
            return SummaryResult(
                repo_name=root.name,
                final_summary=report,
                module_summaries=[],
                directory_summaries={},
                file_summaries={},
                doc_summaries=[],
                stats=stats,
            )

        checkpoint = self._load_or_create_checkpoint(root)

        valid_file_paths = {fu.path for fu in file_units_list}
        checkpoint.file_summaries = {
            path: summary
            for path, summary in checkpoint.file_summaries.items()
            if path in valid_file_paths
        }

        # Phase 3: File summaries (L1+L2)
        if checkpoint.phase_complete.get("files", False) and len(checkpoint.file_summaries) == len(valid_file_paths):
            self._log("\n[Phase 3] Skipping file summaries (checkpoint complete)")
            file_summaries = checkpoint.file_summaries
        else:
            file_summaries = await self._summarize_files(
                file_units_list,
                existing=checkpoint.file_summaries,
            )
            checkpoint.file_summaries = file_summaries
            checkpoint.mark_phase_complete("files")
            self.checkpoints.save(checkpoint)

        # Phase 3b: Doc summaries
        if checkpoint.phase_complete.get("docs", False):
            self._log("\n[Phase 3b] Skipping doc summaries (checkpoint complete)")
            doc_summaries = checkpoint.doc_summaries
        else:
            doc_summaries = await self._summarize_doc_files(
                root,
                existing=checkpoint.doc_summaries,
            )
            checkpoint.doc_summaries = doc_summaries
            checkpoint.mark_phase_complete("docs")
            self.checkpoints.save(checkpoint)

        # Phase 4: Directory summaries (L3)
        if checkpoint.phase_complete.get("dirs", False):
            self._log("\n[Phase 4] Skipping directory summaries (checkpoint complete)")
            dir_summaries = checkpoint.dir_summaries
        else:
            dir_summaries = await self._summarize_directories(
                root,
                file_summaries,
                all_files,
                deep_mode=deep_mode,
                existing=checkpoint.dir_summaries,
            )
            checkpoint.dir_summaries = dir_summaries
            checkpoint.mark_phase_complete("dirs")
            self.checkpoints.save(checkpoint)

        # Phase 5: Module summaries (L4)
        if checkpoint.phase_complete.get("modules", False):
            self._log("\n[Phase 5] Skipping module summaries (checkpoint complete)")
            module_summaries = checkpoint.module_summaries
        else:
            if deep_mode:
                module_summaries = await self._summarize_modules_clustered(
                    root,
                    dir_summaries,
                    existing=checkpoint.module_summaries,
                )
            else:
                module_summaries = await self._summarize_modules(
                    root,
                    dir_summaries,
                    existing=checkpoint.module_summaries,
                )
            checkpoint.module_summaries = module_summaries
            checkpoint.mark_phase_complete("modules")
            self.checkpoints.save(checkpoint)

        # Phase 6: Final synthesis (L5)
        final_summary = await self._final_synthesis(
            root=root,
            all_files=all_files,
            file_units_list=file_units_list,
            module_summaries=module_summaries,
            doc_summaries=doc_summaries,
            file_summaries=file_summaries,
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
