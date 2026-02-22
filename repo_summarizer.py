#!/usr/bin/env python3
"""
repo_summarizer.py — CLI entry point.

Usage:
    # Local path
    python repo_summarizer.py /path/to/repo

    # GitHub URL (public)
    python repo_summarizer.py https://github.com/owner/repo

    # GitHub URL (private, with token)
    python repo_summarizer.py https://github.com/owner/repo --github-token ghp_...

    # Specific branch
    python repo_summarizer.py https://github.com/owner/repo --branch develop

    # Specific commit/tag
    python repo_summarizer.py https://github.com/owner/repo --ref v2.1.0

    # Save output
    python repo_summarizer.py https://github.com/owner/repo --output summary.md

    # Skip doc files
    python repo_summarizer.py /path/to/repo --skip-docs

    # Large repo — increase limit
    python repo_summarizer.py /path/to/big-repo --max-files 5000

Environment:
    ANTHROPIC_API_KEY  — required (or pass --api-key)
    GITHUB_TOKEN       — optional, used for private repos (or pass --github-token)

Supported languages: Python, TypeScript/JavaScript, Rust, Go
"""

from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from github import is_github_url, clone_repo, CloneError
from orchestrator import RepoOrchestrator
from manifest import ManifestStore, build_run_manifest, new_run_id, compute_repo_changes
from diff_report import manifest_diff_to_dict, write_diff_json, generate_diff_digest
from parser import parse_file


def build_markdown_report(result) -> str:
    excluded_dirs = result.stats.get("excluded_directories", [])
    lines = [
        f"# Repository Summary: `{result.repo_name}`",
        "",
        f"*Analyzed {result.stats['files_analyzed']} files across "
        f"{result.stats['modules']} modules | "
        f"Languages: {', '.join(result.stats['languages'])} | "
        f"{result.stats['total_units']} code units parsed*",
        "",
        "---",
        "",
        result.final_summary,
        "",
        "---",
        "",
        "## Module Summaries",
        "",
    ]

    for m in result.module_summaries:
        lines.append(f"### `{m['name']}`")
        lines.append("")
        lines.append(m["summary"])
        lines.append("")

    if result.stats.get("doc_files_summarized", 0) > 0:
        lines += ["---", "", "## Documentation & Config Files", ""]
        for d in result.doc_summaries:
            lines.append(f"### `{Path(d['path']).name}`")
            lines.append(f"*{d['path']}*")
            lines.append("")
            lines.append(d["summary"])
            lines.append("")

    lines += ["---", "", "## File Summaries", ""]
    for path, summary in sorted(result.file_summaries.items()):
        lines.append(f"### `{Path(path).name}`")
        lines.append(f"*{path}*")
        lines.append("")
        lines.append(summary)
        lines.append("")

    lines += [
        "---", "", "## Analysis Stats", "",
        f"- Files analyzed: {result.stats['files_analyzed']}",
        f"- Code units parsed: {result.stats['total_units']}",
        f"- Excluded directories: {len(excluded_dirs)}",
        f"- Doc files summarized: {result.stats.get('doc_files_summarized', 0)}",
        f"- Directories summarized: {result.stats['directories']}",
        f"- Modules summarized: {result.stats['modules']}",
        f"- Languages: {', '.join(result.stats['languages'])}",
        f"- API calls: {result.stats['api_calls']}",
        f"- Cache hits: {result.stats['cache_hits']}",
        f"- Input tokens: {result.stats['input_tokens']:,}",
        f"- Output tokens: {result.stats['output_tokens']:,}",
    ]

    if excluded_dirs:
        lines += ["", "### Excluded Directory Paths", ""]
        max_listed = 100
        for path in excluded_dirs[:max_listed]:
            lines.append(f"- `{path}`")
        if len(excluded_dirs) > max_listed:
            lines.append(f"- ... (+{len(excluded_dirs) - max_listed} more)")

    return "\n".join(lines)


async def summarize_changed_files(orchestrator: RepoOrchestrator, changed_paths: list[str]) -> dict[str, str]:
    file_units = []
    for path in changed_paths:
        fu = parse_file(Path(path))
        if fu is not None:
            file_units.append(fu)
    if not file_units:
        return {}

    tasks = {fu.path: orchestrator.summarizer.summarize_file(fu) for fu in file_units}
    results = await asyncio.gather(*tasks.values())
    return dict(zip(tasks.keys(), results))


def build_diff_only_summary(previous_summary: str, *, added: int, modified: int, deleted: int) -> str:
    return (
        "Diff-only manifest update.\n\n"
        f"Changed files since previous run: +{added} added, {modified} modified, -{deleted} deleted.\n\n"
        "Previous repository summary context:\n"
        f"{previous_summary[:2000]}"
    )


async def main():
    parser = argparse.ArgumentParser(
        description="Recursively summarize a code repository using hierarchical LLM agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "target",
        help="Local path OR GitHub/git URL (e.g. https://github.com/owner/repo)",
    )
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Write markdown report to this file")
    parser.add_argument("--json-output", type=Path, default=None,
                        help="Also write full JSON report")
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (default: ANTHROPIC_API_KEY env var)")
    parser.add_argument("--github-token", default=None,
                        help="GitHub PAT for private repos (default: GITHUB_TOKEN env var)")
    parser.add_argument("--branch", default=None,
                        help="Branch to clone (default: repo default branch)")
    parser.add_argument("--ref", default=None,
                        help="Specific commit SHA or tag to checkout after clone")
    parser.add_argument("--max-concurrent", type=int, default=20,
                        help="Max concurrent API calls (default: 20)")
    parser.add_argument("--haiku-concurrency", type=int, default=None,
                        help="Max concurrent Haiku API calls (default: --max-concurrent)")
    parser.add_argument("--sonnet-concurrency", type=int, default=None,
                        help="Max concurrent Sonnet API calls (default: --max-concurrent)")
    parser.add_argument("--max-files", type=int, default=10000,
                        help="Abort if repo exceeds this many source files (default: 10000)")
    parser.add_argument("--deep-mode-threshold", type=int, default=500,
                        help="Enable deep mode when file count >= threshold (default: 500)")
    parser.add_argument("--l3-chunk-size", type=int, default=15,
                        help="Max files per directory chunk in deep mode (default: 15)")
    parser.add_argument("--l4-cluster-size", type=int, default=8,
                        help="Max modules per cluster in deep mode (default: 8)")
    parser.add_argument("--l1-batch-size", type=int, default=8,
                        help="Units per L1 batch request (default: 8)")
    parser.add_argument("--l1-batch-threshold", type=int, default=500,
                        help="Max source chars for L1 batch eligibility (default: 500)")
    parser.add_argument("--progress-heartbeat-secs", type=float, default=20.0,
                        help="Emit long-running phase heartbeat logs every N seconds in --verbose mode (default: 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate cost and exit without making API calls")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Cache directory (default: ~/.repo_summarizer_cache)")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Checkpoint directory for resumable runs")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="Resume from latest checkpoint when available (default: on)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Disable checkpoint resume")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable caching (always re-summarize)")
    parser.add_argument("--skip-docs", action="store_true",
                        help="Skip summarization of documentation and config files")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print progress for each agent")
    parser.add_argument("--exclude-dir", action="append", dest="exclude_dirs",
                        default=[], metavar="DIR",
                        help="Additional directory names or relative paths to exclude (repeatable)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print only the final ~1200 word summary")
    parser.add_argument("--diff", dest="diff", action="store_true", default=True,
                        help="Generate diff report against previous manifest (default: on)")
    parser.add_argument("--no-diff", dest="diff", action="store_false",
                        help="Disable diff report generation")
    parser.add_argument("--diff-output", type=Path, default=None,
                        help="Path for JSON diff output (default: <output>_diff.json)")
    parser.add_argument("--diff-digest", dest="diff_digest", action="store_true", default=True,
                        help="Generate prose digest for the diff (default: on)")
    parser.add_argument("--no-diff-digest", dest="diff_digest", action="store_false",
                        help="Disable prose digest generation")
    parser.add_argument("--diff-only", action="store_true",
                        help="Fast path: summarize changed files only and output diff report content")
    parser.add_argument("--since", default=None,
                        help="Diff against a specific prior run_id prefix")
    parser.add_argument("--manifest-dir", type=Path, default=None,
                        help="Manifest directory for run history and diffing")
    parser.add_argument("--keep-manifests", type=int, default=None,
                        help="Keep only the most recent N manifests per repo")

    args = parser.parse_args()

    # API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not args.dry_run and not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    # GitHub token
    github_token = args.github_token or os.environ.get("GITHUB_TOKEN")

    # Cache
    cache_dir = args.cache_dir
    if args.no_cache:
        import tempfile
        cache_dir = Path(tempfile.mkdtemp())

    extra_excludes = set(args.exclude_dirs) if args.exclude_dirs else None

    # -----------------------------------------------------------------------
    # Resolve target: URL vs local path
    # -----------------------------------------------------------------------
    tmp_dir = None
    target = args.target
    repo_url: str | None = None

    if is_github_url(target):
        repo_url = target
        ref_desc = f" @ {args.branch or args.ref or 'HEAD'}"
        print(f"\n🔗 Cloning {target}{ref_desc}...", flush=True)
        try:
            tmp_dir = clone_repo(
                target,
                token=github_token,
                branch=args.branch,
                ref=args.ref,
            )
            repo_path = Path(tmp_dir.name)
            print(f"   Cloned to {repo_path}", flush=True)
        except CloneError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        repo_path = Path(target)
        if not repo_path.exists():
            print(f"Error: Path does not exist: {repo_path}", file=sys.stderr)
            sys.exit(1)

    display_repo_name = repo_path.resolve().name or str(repo_path.resolve())
    print(f"\n🔍 Repo Summarizer — analyzing `{display_repo_name}`", flush=True)
    start = time.time()

    orchestrator = RepoOrchestrator(
        api_key=api_key,
        cache_dir=cache_dir,
        max_concurrent=args.max_concurrent,
        haiku_concurrency=args.haiku_concurrency,
        sonnet_concurrency=args.sonnet_concurrency,
        verbose=args.verbose,
        exclude_dirs=extra_excludes,
        max_files=args.max_files,
        skip_docs=args.skip_docs,
        deep_mode_threshold=args.deep_mode_threshold,
        l3_chunk_size=args.l3_chunk_size,
        l4_cluster_size=args.l4_cluster_size,
        l1_batch_size=args.l1_batch_size,
        l1_batch_threshold=args.l1_batch_threshold,
        progress_heartbeat_secs=args.progress_heartbeat_secs,
        dry_run=args.dry_run,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
    )
    manifest_store = ManifestStore(args.manifest_dir) if (args.diff and not args.dry_run) else None
    diff_payload = None
    diff_digest_text = None
    diff_output_path = None

    if args.diff_only:
        if manifest_store is None:
            print("Error: --diff-only requires --diff (and cannot be used with --dry-run).", file=sys.stderr)
            if tmp_dir is not None:
                tmp_dir.cleanup()
            sys.exit(1)

        previous_manifest = manifest_store.find_latest(repo_path, run_id_prefix=args.since)
        if previous_manifest is None:
            output_text = "No previous manifest found; cannot run --diff-only yet."
            elapsed = time.time() - start
            print(f"\n✅ Complete in {elapsed:.1f}s\n", flush=True)
            if args.output:
                args.output.write_text(output_text)
                print(f"📄 Report written to: {args.output}")
            else:
                print("\n" + "=" * 80)
                print(output_text)
            if tmp_dir is not None:
                tmp_dir.cleanup()
            return

        changes = compute_repo_changes(
            repo_path=repo_path.resolve(),
            previous_manifest=previous_manifest,
            extra_excludes=extra_excludes,
        )
        changed_paths = sorted(changes.added + changes.modified)
        changed_summaries = await summarize_changed_files(orchestrator, changed_paths)

        merged_file_summaries: dict[str, str] = {}
        for path in sorted(changes.current_hashes.keys()):
            if path in changed_summaries:
                merged_file_summaries[path] = changed_summaries[path]
            elif path in previous_manifest.files:
                merged_file_summaries[path] = previous_manifest.files[path].summary

        elapsed = time.time() - start
        current_manifest = build_run_manifest(
            repo_path=repo_path,
            repo_url=repo_url,
            run_id=new_run_id(repo_path),
            duration_secs=elapsed,
            api_cost_usd=orchestrator.summarizer.tracker.estimate_usd(),
            final_summary=build_diff_only_summary(
                previous_manifest.final_summary,
                added=len(changes.added),
                modified=len(changes.modified),
                deleted=len(changes.deleted),
            ),
            file_summaries=merged_file_summaries,
        )
        manifest_path = manifest_store.save(current_manifest)
        if args.verbose:
            print(f"🗂️  Manifest written to: {manifest_path}")
        if args.keep_manifests is not None:
            pruned = manifest_store.prune(repo_path, keep=args.keep_manifests)
            if args.verbose and pruned:
                print(f"🧹 Pruned {len(pruned)} old manifests")

        diff_obj = manifest_store.diff(previous_manifest, current_manifest)
        history = manifest_store.find_all(repo_path, limit=5)
        diff_obj.churn_hotspots = manifest_store.compute_churn_hotspots(history)
        diff_payload = manifest_diff_to_dict(
            diff_obj,
            old_run_id=previous_manifest.run_id,
            new_run_id=current_manifest.run_id,
        )

        if args.diff_output is not None:
            diff_output_path = args.diff_output
        elif args.output is not None:
            diff_output_path = args.output.with_name(f"{args.output.stem}_diff.json")
        else:
            diff_output_path = Path(f"{repo_path.resolve().name}_diff.json")
        write_diff_json(diff_output_path, diff_payload)
        print(f"📎 Diff JSON written to: {diff_output_path}")

        if args.diff_digest:
            diff_digest_text = await generate_diff_digest(
                summarizer=orchestrator.summarizer,
                diff=diff_obj,
                old_manifest=previous_manifest,
                new_manifest=current_manifest,
            )

        print(f"\n✅ Complete in {elapsed:.1f}s\n", flush=True)
        if diff_digest_text:
            output_text = diff_digest_text
        else:
            output_text = json.dumps(diff_payload, indent=2)

        if args.output:
            args.output.write_text(output_text)
            print(f"📄 Report written to: {args.output}")
        else:
            print("\n" + "=" * 80)
            print(output_text)

        if args.json_output:
            json_data = {
                "repo_name": display_repo_name,
                "mode": "diff_only",
                "diff": diff_payload,
                "diff_digest": diff_digest_text,
                "changed_files_summarized": len(changed_summaries),
                "stats": vars(orchestrator.summarizer.tracker),
            }
            args.json_output.write_text(json.dumps(json_data, indent=2))
            print(f"📊 JSON report written to: {args.json_output}")

        print(f"\n💰 {orchestrator.summarizer.tracker.report()}")
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return

    try:
        result = await orchestrator.run(repo_path)
    except ValueError as e:
        if tmp_dir is not None:
            tmp_dir.cleanup()
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - start
    print(f"\n✅ Complete in {elapsed:.1f}s\n", flush=True)

    if manifest_store is not None:
        previous_manifest = manifest_store.find_latest(repo_path, run_id_prefix=args.since)

        current_manifest = build_run_manifest(
            repo_path=repo_path,
            repo_url=repo_url,
            run_id=new_run_id(repo_path),
            duration_secs=elapsed,
            api_cost_usd=orchestrator.summarizer.tracker.estimate_usd(),
            final_summary=result.final_summary,
            file_summaries=result.file_summaries,
        )
        manifest_path = manifest_store.save(current_manifest)
        if args.verbose:
            print(f"🗂️  Manifest written to: {manifest_path}")
        if args.keep_manifests is not None:
            pruned = manifest_store.prune(repo_path, keep=args.keep_manifests)
            if args.verbose and pruned:
                print(f"🧹 Pruned {len(pruned)} old manifests")

        if previous_manifest is not None:
            diff_obj = manifest_store.diff(previous_manifest, current_manifest)
            history = manifest_store.find_all(repo_path, limit=5)
            diff_obj.churn_hotspots = manifest_store.compute_churn_hotspots(history)

            diff_payload = manifest_diff_to_dict(
                diff_obj,
                old_run_id=previous_manifest.run_id,
                new_run_id=current_manifest.run_id,
            )

            if args.diff_output is not None:
                diff_output_path = args.diff_output
            elif args.output is not None:
                diff_output_path = args.output.with_name(f"{args.output.stem}_diff.json")
            else:
                diff_output_path = Path(f"{repo_path.resolve().name}_diff.json")
            write_diff_json(diff_output_path, diff_payload)
            print(f"📎 Diff JSON written to: {diff_output_path}")

            if args.diff_digest:
                diff_digest_text = await generate_diff_digest(
                    summarizer=orchestrator.summarizer,
                    diff=diff_obj,
                    old_manifest=previous_manifest,
                    new_manifest=current_manifest,
                )
        elif args.verbose:
            print("ℹ️  No previous manifest found; skipping diff generation.")

    if args.diff_only:
        if diff_digest_text:
            output_text = diff_digest_text
        elif diff_payload:
            output_text = json.dumps(diff_payload, indent=2)
        else:
            output_text = "No diff available for this run."
    elif args.dry_run or args.summary_only:
        output_text = result.final_summary
    else:
        output_text = build_markdown_report(result)
        if diff_digest_text:
            output_text += "\n\n---\n\n## Diff Digest\n\n" + diff_digest_text

    if args.output:
        args.output.write_text(output_text)
        print(f"📄 Report written to: {args.output}")
    else:
        print("\n" + "=" * 80)
        print(output_text)

    if args.json_output:
        json_data = {
            "repo_name": result.repo_name,
            "final_summary": result.final_summary,
            "module_summaries": result.module_summaries,
            "doc_summaries": result.doc_summaries,
            "directory_summaries": result.directory_summaries,
            "file_summaries": result.file_summaries,
            "stats": result.stats,
            "diff": diff_payload,
            "diff_digest": diff_digest_text,
        }
        args.json_output.write_text(json.dumps(json_data, indent=2))
        print(f"📊 JSON report written to: {args.json_output}")

    print(f"\n💰 {orchestrator.summarizer.tracker.report()}")

    # Always clean up temp clone dir after all post-processing that needs repo files.
    if tmp_dir is not None:
        tmp_dir.cleanup()


def run_cli() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run_cli()
