"""
prompts.py — Layer-specific prompts for each level of the summarization hierarchy.

Each layer gets progressively more architectural context and is asked
progressively higher-level questions.
"""

from __future__ import annotations
from typing import Optional
from manifest import ManifestDiff


# ---------------------------------------------------------------------------
# L1: Function / Class / Struct level
# ---------------------------------------------------------------------------

def l1_unit_prompt(
    *,
    language: str,
    file_path: str,
    unit_kind: str,
    unit_name: str,
    parent_name: Optional[str],
    source: str,
    docstring: Optional[str],
    doc_comment: Optional[str] = None,
) -> str:
    parent_ctx = f" (inside `{parent_name}`)" if parent_name else ""

    # Build documentation context — prefer doc_comment (more common across languages),
    # fall back to Python docstring. Both present? Show both.
    doc_parts = []
    if doc_comment:
        doc_parts.append(f"Doc comment: {doc_comment[:400]}")
    if docstring:
        doc_parts.append(f"Docstring: {docstring[:400]}")
    doc_ctx = ("\n" + "\n".join(doc_parts)) if doc_parts else ""

    if len(source) > 8000:
        source = source[:8000] + "\n... [truncated for brevity]"

    return f"""Language: {language}
File: {file_path}
Unit: {unit_kind} `{unit_name}`{parent_ctx}{doc_ctx}

```{language}
{source}
```"""


# ---------------------------------------------------------------------------
# L2: File level
# ---------------------------------------------------------------------------

def l2_file_prompt(
    *,
    file_path: str,
    language: str,
    unit_summaries: list[dict],  # [{name, kind, parent, summary}]
    file_line_count: int,
) -> str:
    units_block = "\n".join(
        f"- `{u['name']}` ({u['kind']}"
        + (f", in `{u['parent']}`" if u.get('parent') else "")
        + f"): {u['summary']}"
        for u in unit_summaries
    )

    return f"""You are a senior engineer writing a file-level summary for a repository index.

File: {file_path}
Language: {language}
Lines: {file_line_count}

Individual unit summaries:
{units_block}

Write 1 focused paragraph (5-8 sentences) that synthesizes what this FILE does as a whole:
- Its primary responsibility in the codebase
- Key abstractions or types it defines
- Important functions/methods and how they relate
- Any patterns (e.g. factory, middleware, handler, data model, utility)

Do not list every function — synthesize them into a coherent description of the file's role.
Output only the paragraph, no headers."""


# ---------------------------------------------------------------------------
# L3: Directory level
# ---------------------------------------------------------------------------

def l3_directory_prompt(
    *,
    dir_path: str,
    file_summaries: list[dict],  # [{file_name, summary}]
    subdirectory_summaries: list[dict],  # [{dir_name, summary}]
) -> str:
    files_block = "\n".join(
        f"- `{f['name']}`: {f['summary']}"
        for f in file_summaries
    )
    subdirs_block = ""
    if subdirectory_summaries:
        subdirs_block = "\nSubdirectories:\n" + "\n".join(
            f"- `{d['name']}/`: {d['summary']}"
            for d in subdirectory_summaries
        )

    return f"""You are a senior engineer writing a directory-level summary for a repository index.

Directory: {dir_path}

Files:
{files_block}
{subdirs_block}

Write 3-5 sentences describing what this directory/module is responsible for:
- What domain or layer does it represent (e.g. API handlers, data models, utilities, config)?
- How do the files in it relate to each other?
- What is the primary public interface or entry point if there is one?

Output only the summary, no headers or bullet points."""


# ---------------------------------------------------------------------------
# L4: Module/Package level (top-level directories)
# ---------------------------------------------------------------------------

def l4_module_prompt(
    *,
    module_name: str,
    directory_summaries: list[dict],  # [{dir_name, summary}]
) -> str:
    dirs_block = "\n".join(
        f"- `{d['name']}`: {d['summary']}"
        for d in directory_summaries
    )

    return f"""You are a senior engineer writing a module-level summary for a repository index.

Module/Package: {module_name}

Component directories:
{dirs_block}

Write 4-6 sentences describing this module as an architectural unit:
- What problem domain or system concern does it own?
- How do the sub-components relate to each other?
- What are the key abstractions or interfaces it exposes?
- How does it fit into a larger system (if inferrable)?

Output only the summary, no headers."""


# ---------------------------------------------------------------------------
# L5: Final synthesizer
# ---------------------------------------------------------------------------

def l5_final_prompt(
    *,
    repo_name: str,
    repo_structure: str,
    module_summaries: list[dict],
    languages: list[str],
    file_count: int,
    readme_excerpt: Optional[str],
    manifest_excerpt: Optional[str],
    doc_summaries: list[dict] | None = None,
) -> str:
    modules_block = "\n\n".join(
        f"### {m['name']}\n{m['summary']}"
        for m in module_summaries
    )

    readme_section = ""
    if readme_excerpt:
        readme_section = f"\nREADME excerpt:\n{readme_excerpt[:1500]}\n"

    manifest_section = ""
    if manifest_excerpt:
        manifest_section = f"\nManifest/dependencies excerpt:\n{manifest_excerpt[:800]}\n"

    docs_section = ""
    if doc_summaries:
        from pathlib import Path as _Path
        docs_block = "\n".join(
            f"- `{_Path(d['path']).name}`: {d['summary']}"
            for d in doc_summaries
        )
        docs_section = f"\nDocumentation & config files:\n{docs_block}\n"

    lang_str = ", ".join(languages)

    return f"""You are a principal engineer writing a comprehensive technical summary of a code repository.
This summary will be read by developers who are new to the codebase and need to understand it quickly.

Repository: {repo_name}
Languages: {lang_str}
Total files analyzed: {file_count}
{readme_section}{manifest_section}{docs_section}
Directory structure:
{repo_structure}

Module summaries from hierarchical analysis:
{modules_block}

Write 700-900 words in Markdown. Output only the report body.
Do not include any H1 title. Start directly with `## Executive Summary`.
Use exactly these H2 sections in this order:

## Executive Summary
- Provide 5-10 bullet points focused on architecture, purpose, and operational behavior.

## System Architecture
- Explain L1-L5 flow, subsystem boundaries, and component interactions.

## Key Components
- Cover only the most important 6-8 modules/files and why they matter.

## Operational Model
- Explain runtime behavior, caching, checkpoint/resume, cost/concurrency controls, and diff/incremental flow.

## Risks & Limitations
- Call out uncertainty, likely failure modes, tradeoffs, and current technical debt.

## Testing Coverage Snapshot
- Summarize what test areas are well-covered and what gaps remain.

Use concrete evidence from the summaries. If something is unclear, explicitly say so instead of guessing."""


# ---------------------------------------------------------------------------
# Deep-mode prompt additions
# ---------------------------------------------------------------------------

def l3_chunk_prompt(
    *,
    dir_path: str,
    file_summaries_subset: list[dict],  # [{name, summary}]
    chunk_index: int,
    total_chunks: int,
) -> str:
    files_block = "\n".join(
        f"- `{f['name']}`: {f['summary']}"
        for f in file_summaries_subset
    )
    return f"""You are a senior engineer writing a directory-level summary for a repository index.

Directory: {dir_path}
Chunk: {chunk_index} of {total_chunks}

Files in this chunk:
{files_block}

Write 3-5 sentences summarizing only this chunk of files. Focus on:
- The responsibilities represented by this subset
- Any strong relationships among files in this chunk
- How this chunk likely contributes to the parent directory's overall role

Do not assume files not shown here. Output only the summary text."""


def l3_merge_prompt(
    *,
    dir_path: str,
    chunk_summaries: list[str],
) -> str:
    chunks_block = "\n".join(
        f"- Chunk {i + 1}: {summary}"
        for i, summary in enumerate(chunk_summaries)
    )
    return f"""You are a senior engineer merging chunk-level summaries into one directory summary.

Directory: {dir_path}

Chunk summaries:
{chunks_block}

Synthesize these into one coherent 3-5 sentence directory summary.
Resolve apparent conflicts and remove repetition.
Output only the final merged summary."""


def l4_cluster_prompt(
    *,
    cluster_modules: list[dict],  # [{name, summary}]
) -> str:
    modules_block = "\n".join(
        f"- `{m['name']}`: {m['summary']}"
        for m in cluster_modules
    )
    return f"""You are a senior engineer summarizing a cluster of top-level modules.

Cluster modules:
{modules_block}

Write 4-6 sentences describing the shared concern this module cluster appears to own,
how these modules relate, and what architectural boundaries are visible.
Output only the summary text."""


def l4_final_merge_prompt(
    *,
    cluster_summaries: list[str],
) -> str:
    clusters_block = "\n".join(
        f"- Cluster {i + 1}: {summary}"
        for i, summary in enumerate(cluster_summaries)
    )
    return f"""You are a senior engineer consolidating cluster-level architectural summaries.

Cluster summaries:
{clusters_block}

Produce a concise architectural synthesis in 5-8 sentences that captures:
- The major subsystems
- Their relationships and boundaries
- The dominant architectural patterns

Output only plain summary text, no headers."""


def l5_tool_use_prompt(
    *,
    repo_name: str,
    languages: list[str],
    file_count: int,
    repo_structure: str,
    readme_excerpt: Optional[str],
) -> str:
    lang_str = ", ".join(languages)
    readme_block = f"\nREADME excerpt:\n{readme_excerpt[:1500]}\n" if readme_excerpt else ""

    return f"""You are a principal engineer producing a technical summary via tool use.

Repository: {repo_name}
Languages: {lang_str}
Files analyzed: {file_count}
{readme_block}
Repository structure:
{repo_structure}

Use tools to gather only the module and file summaries needed to understand the architecture.
Start with `list_modules`, then selectively call `get_module_summary` for the most important
modules. Do not fetch every module by default.

When you have enough context, produce a 700-900 word report.
Output only markdown body text (no H1 title, no preamble).
Start with `## Executive Summary` and use exactly these H2 sections in order:
Executive Summary, System Architecture, Key Components, Operational Model,
Risks & Limitations, Testing Coverage Snapshot."""


# ---------------------------------------------------------------------------
# Diff digest prompt
# ---------------------------------------------------------------------------

def diff_digest_prompt(
    *,
    old_run_id: str | None,
    new_run_id: str,
    old_summary: str | None,
    new_summary: str,
    diff: ManifestDiff,
) -> str:
    modified_block = "\n".join(
        f"- `{m.path}` ({m.language}): unit_delta={m.unit_count_delta}\n"
        f"  old: {m.old_summary}\n"
        f"  new: {m.new_summary}"
        for m in diff.modified[:100]
    ) or "- [none]"
    added_block = "\n".join(f"- `{p}`" for p in diff.added[:100]) or "- [none]"
    deleted_block = "\n".join(f"- `{p}`" for p in diff.deleted[:100]) or "- [none]"
    churn_block = "\n".join(f"- `{p}`" for p in diff.churn_hotspots[:50]) or "- [none]"

    old_summary_block = old_summary[:2000] if old_summary else "[unavailable]"
    new_summary_block = new_summary[:2000] if new_summary else "[unavailable]"

    return f"""You are writing an engineering change digest for a technical manager.

Previous run: {old_run_id or "none"}
Current run: {new_run_id}

Added files:
{added_block}

Deleted files:
{deleted_block}

Modified files:
{modified_block}

Churn hotspots:
{churn_block}

Previous repository summary excerpt:
{old_summary_block}

Current repository summary excerpt:
{new_summary_block}

Write a 400-600 word digest with sections:
1) Engineering Pulse
2) What Shipped
3) What Was Removed (only if any deleted files)
4) Complexity Signals
5) Architecture Notes
6) By The Numbers

Use subsystem-level language. Avoid line-level implementation detail."""


# ---------------------------------------------------------------------------
# Doc file summarization
# ---------------------------------------------------------------------------

def doc_file_prompt(*, file_path: str, extension: str, content: str) -> str:
    """Prompt for summarizing a documentation or config file."""
    if len(content) > 6000:
        content = content[:6000] + "\n... [truncated]"

    file_type_hints = {
        ".md":   "Markdown documentation",
        ".mdx":  "MDX documentation",
        ".rst":  "reStructuredText documentation",
        ".yaml": "YAML configuration or spec",
        ".yml":  "YAML configuration or spec",
        ".toml": "TOML configuration",
        ".json": "JSON configuration or schema",
        ".txt":  "text documentation",
    }
    file_type = file_type_hints.get(extension, "documentation file")

    return f"""You are a senior engineer summarizing a {file_type} file for a repository index.

File: {file_path}

```
{content}
```

Write 2-4 sentences describing:
- What this file contains or configures
- Any key values, sections, or decisions that are architecturally significant
- How it relates to the rest of the project (if inferrable)

Skip boilerplate sections (license headers, standard config noise).
Be concise and technically specific. Output only the summary, no headers."""
