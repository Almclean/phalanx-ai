"""
prompts.py — Layer-specific prompts for each level of the summarization hierarchy.

Each layer gets progressively more architectural context and is asked
progressively higher-level questions.
"""

from __future__ import annotations
from typing import Optional


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

    return f"""You are a senior engineer summarizing a single code unit for a repository index.

Language: {language}
File: {file_path}
Unit: {unit_kind} `{unit_name}`{parent_ctx}{doc_ctx}

```{language}
{source}
```

Write a concise 2-4 sentence technical summary covering:
1. What this {unit_kind} does (its purpose and responsibility)
2. Key inputs, outputs, or side effects (if meaningful)
3. Any important patterns, algorithms, or gotchas worth noting

If a doc comment or docstring is provided above, treat it as authoritative — use it to \
inform your summary but don't just repeat it verbatim.
Be specific and technical. Do not start with "This {unit_kind}..." — vary your phrasing.
Output only the summary text, no headers or bullet points."""


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

Write a technical summary of approximately 1,200 words. Structure it as flowing prose with the following sections
(use markdown H2 headers for each section):

## Overview
What this repository is, what problem it solves, and who/what uses it. Include the technology stack.

## Architecture
The high-level architectural design: layers, major components, how they fit together. Discuss patterns
(microservices, monolith, plugin architecture, MVC, etc.) with evidence from the code.

## Core Components
Walk through the most important modules/packages, what each owns, and their key abstractions.
Be specific — name important types, interfaces, or functions where they clarify the design.

## Data Flow & Key Interactions
How data moves through the system: entry points, processing pipeline, persistence, outputs.
Describe the most important call paths or workflows.

## Technical Patterns & Design Decisions
Notable engineering choices: error handling strategy, concurrency model, abstraction patterns,
testing approach, configuration management. Point out anything architecturally interesting.

## Developer Notes
What a new developer needs to know to start contributing: where to find entry points,
key conventions, anything that's non-obvious or could trip someone up.

Write in confident, precise technical prose. Be specific — generic descriptions are useless.
If something is unclear from the summaries, say so rather than fabricating.
Target: exactly 1,200 words."""


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
