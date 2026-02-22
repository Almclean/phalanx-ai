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


def build_markdown_report(result) -> str:
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
        f"- Doc files summarized: {result.stats.get('doc_files_summarized', 0)}",
        f"- Directories summarized: {result.stats['directories']}",
        f"- Modules summarized: {result.stats['modules']}",
        f"- Languages: {', '.join(result.stats['languages'])}",
        f"- API calls: {result.stats['api_calls']}",
        f"- Cache hits: {result.stats['cache_hits']}",
        f"- Input tokens: {result.stats['input_tokens']:,}",
        f"- Output tokens: {result.stats['output_tokens']:,}",
    ]

    return "\n".join(lines)


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
    parser.add_argument("--max-files", type=int, default=2000,
                        help="Abort if repo exceeds this many source files (default: 2000)")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Cache directory (default: ~/.repo_summarizer_cache)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable caching (always re-summarize)")
    parser.add_argument("--skip-docs", action="store_true",
                        help="Skip summarization of documentation and config files")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print progress for each agent")
    parser.add_argument("--exclude-dir", action="append", dest="exclude_dirs",
                        default=[], metavar="DIR",
                        help="Additional directory names to exclude (repeatable)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print only the final ~1200 word summary")

    args = parser.parse_args()

    # API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
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

    if is_github_url(target):
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

    print(f"\n🔍 Repo Summarizer — analyzing `{repo_path.name}`", flush=True)
    start = time.time()

    orchestrator = RepoOrchestrator(
        api_key=api_key,
        cache_dir=cache_dir,
        max_concurrent=args.max_concurrent,
        verbose=args.verbose,
        exclude_dirs=extra_excludes,
        max_files=args.max_files,
        skip_docs=args.skip_docs,
    )

    try:
        result = await orchestrator.run(repo_path)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Always clean up temp clone dir
        if tmp_dir is not None:
            tmp_dir.cleanup()

    elapsed = time.time() - start
    print(f"\n✅ Complete in {elapsed:.1f}s\n", flush=True)

    if args.summary_only:
        output_text = result.final_summary
    else:
        output_text = build_markdown_report(result)

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
        }
        args.json_output.write_text(json.dumps(json_data, indent=2))
        print(f"📊 JSON report written to: {args.json_output}")

    print(f"\n💰 {orchestrator.summarizer.tracker.report()}")


def run_cli() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run_cli()
