# Phalanx

A recursive, hierarchical LLM agent system for generating technical summaries of code repositories.

## Architecture

Implements a **bottom-up hierarchical summarization** approach based on:
- [Hierarchical Repository-Level Code Summarization (ICCSA 2025)](https://arxiv.org/abs/2501.07857)
- [Recursive Language Models — Zhang & Khattab, MIT/arXiv 2512.24601](https://arxiv.org/abs/2512.24601)
- [RepoSummary / HMCS (Sun et al., 2025)](https://arxiv.org/html/2510.11039)

```
REPO
 └── L1: Function/Class agents     → 2-4 sentence unit summaries    [Haiku, parallel]
 └── L2: File agents               → 1 paragraph file summaries     [Haiku, parallel]
 └── L3: Directory agents          → 3-5 sentence dir summaries     [Sonnet, bottom-up]
 └── L4: Module agents             → 4-6 sentence module summaries  [Sonnet]
 └── L5: Final synthesizer         → 1,200 word technical summary   [Sonnet]
```

Each layer only sees the outputs of the layer below it — never raw source code above L2.
This keeps context clean and avoids context rot at higher levels.

## Supported Languages

| Language | Extension(s) | AST Parser |
|----------|-------------|------------|
| Python | `.py` | `tree-sitter-python` |
| TypeScript | `.ts`, `.tsx`, `.mts` | `tree-sitter-typescript` |
| JavaScript | `.js`, `.mjs`, `.cjs` | `tree-sitter-typescript` |
| Rust | `.rs` | `tree-sitter-rust` |
| Go | `.go` | `tree-sitter-go` |

C/C++ support: install `tree-sitter-c` and `tree-sitter-cpp`, then add to `_EXT_MAP` in `parser.py`.

## Installation (uv)

```bash
uv sync
```

## Usage

```bash
# Basic — prints to stdout
export ANTHROPIC_API_KEY=sk-ant-...
uv run phalanx /path/to/your/repo

# Save to file
uv run phalanx /path/to/your/repo --output summary.md

# Also save full JSON (all layer summaries)
uv run phalanx /path/to/your/repo --output summary.md --json-output full.json

# Verbose mode (see each agent working)
uv run phalanx /path/to/your/repo --verbose

# Summary only (no appendices)
uv run phalanx /path/to/your/repo --summary-only

# Exclude extra directories
uv run phalanx /path/to/your/repo --exclude-dir migrations --exclude-dir fixtures

# Tune concurrency (default 20, increase for large repos)
uv run phalanx /path/to/your/repo --max-concurrent 40
```

## Cost Estimates

| Repo size | Files | Est. API calls | Est. cost |
|-----------|-------|----------------|-----------|
| Small | ~50 files | ~300 | ~$0.05 |
| Medium | ~200 files | ~1,200 | ~$0.20 |
| Large | ~1,000 files | ~6,000 | ~$1.00 |

Caching means re-runs on unchanged repos are nearly free.
Cache lives at `~/.repo_summarizer_cache/summaries.json`.

## Model Strategy

- **L1 + L2** (leaves): `claude-haiku-4-5` — fast, cheap, heavily parallelized
- **L3 + L4** (directories/modules): `claude-sonnet-4-6` — better synthesis
- **L5** (final): `claude-sonnet-4-6` — the 1,200-word technical summary

## Output

The tool generates:
1. **Final summary** (~1,200 words): Overview, Architecture, Core Components, Data Flow, Technical Patterns, Developer Notes
2. **Module summaries**: One per top-level directory
3. **File summaries**: One per source file
4. **Stats**: Token counts, API calls, cache hits, estimated cost

## Extending to C/C++

```bash
uv add tree-sitter-c tree-sitter-cpp
```

Then in `parser.py`:
```python
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp

_LANGUAGES["c"] = Language(tsc.language())
_LANGUAGES["cpp"] = Language(tscpp.language())

_EXT_MAP.update({
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
})

_TOP_LEVEL_NODES["c"] = {"function_definition", "struct_specifier", "enum_specifier"}
_TOP_LEVEL_NODES["cpp"] = {
    "function_definition", "class_specifier", "struct_specifier",
    "enum_specifier", "namespace_definition", "template_declaration",
}
```

## Files

```
repo_summarizer/
├── pyproject.toml       # Project metadata + dependencies (uv)
├── repo_summarizer.py   # CLI entry point
├── orchestrator.py      # DAG execution engine
├── agents.py            # Async LLM callers (one per layer)
├── prompts.py           # Layer-specific prompts
├── parser.py            # AST extraction (tree-sitter)
├── cache.py             # Content-addressed summary cache
└── README.md
```
