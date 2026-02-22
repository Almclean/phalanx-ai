# Phalanx

Phalanx is a recursive, hierarchical codebase summarizer for local and remote repositories.
It uses AST parsing plus multi-layer LLM synthesis to produce architecture-level summaries that stay useful on large repos.

## What It Does

Phalanx runs a layered pipeline:

1. Parse source files into code units (functions, classes, impl blocks, etc.).
2. Summarize code units into file summaries.
3. Summarize files into directory summaries.
4. Summarize directories into module summaries.
5. Produce a final technical report.

It also supports:

- Deep mode for large repositories
- Resume from checkpoints
- Diff reports between runs
- Diff-only fast path (summarize changed files only)
- Manifest retention pruning

## Supported Languages

| Language | Extensions | Parser |
|---|---|---|
| Python | `.py` | `tree-sitter-python` |
| TypeScript | `.ts`, `.tsx`, `.mts`, `.cts` | `tree-sitter-typescript` |
| JavaScript | `.js`, `.mjs`, `.cjs` | `tree-sitter-typescript` |
| Rust | `.rs` | `tree-sitter-rust` |
| Go | `.go` | `tree-sitter-go` |
| C | `.c`, `.h` | `tree-sitter-c` |
| C++ | `.cpp`, `.cc`, `.cxx`, `.hpp` | `tree-sitter-cpp` |

## Architecture

```text
REPO
  -> L1: Unit summaries        (Haiku, parallel, batched for small units)
  -> L2: File summaries        (Haiku)
  -> L3: Directory summaries   (Sonnet; chunked in deep mode)
  -> L4: Module summaries      (Sonnet; clustered in deep mode)
  -> L5: Final synthesis       (Sonnet tool-use loop)
```

Key design points:

- Bottom-up summarization to preserve structure
- Tiered model concurrency (Haiku and Sonnet limits)
- Content-addressed summary cache
- Checkpointed phases for resumable long runs

## Cost Planning (Read First)

Before running on a large repo, estimate first:

```bash
uv run phalanx /path/to/repo --dry-run --summary-only
```

How to reason about cost:

- Dry run gives estimated token volume and estimated cost before any API calls.
- Final run prints actual token totals (`Tokens in`, `Tokens out`) and estimated blended spend.
- Exclude bundled/vendor trees to control cost: `--exclude-dir vendor --exclude-dir third_party --exclude-dir src/external`
- Use incremental mode after first run: `--diff-only` summarizes only changed files.

Worked example (raylib run):

| Metric | Value |
|---|---:|
| Files analyzed | 357 |
| API calls | 14,238 |
| Cache hits | 3,965 |
| Tokens in | 6,236,536 |
| Tokens out | 1,480,547 |
| Estimated blended spend | ~$19.91 |

Same token volume against common list-price tiers (illustrative, as of 2026-02-22):

| Provider/model tier | Approx cost |
|---|---:|
| Anthropic Haiku-class | ~$13.64 |
| Anthropic Sonnet-class | ~$40.92 |
| Anthropic Opus-class | ~$68.20 |
| OpenAI mini-class | ~$4.52 |
| OpenAI flagship-class | ~$31.64 |
| Google Gemini Flash-class | ~$7.56 |
| Google Gemini Pro-class | ~$30.24 to ~$51.60 |

Notes:

- These comparisons are list-price estimates from token totals, not invoice totals.
- Model pricing changes over time; always re-check provider pricing pages before budget sign-off.

## Installation

### Prerequisites

- Python 3.10+
- `uv`
- `git` (for remote repo URLs)

### Install

```bash
uv sync
```

This creates `.venv`, installs dependencies, and installs the local `phalanx` CLI entrypoint.

## Environment Setup

Set your Anthropic key before running non-dry workflows.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Optional for private GitHub repos:

```bash
export GITHUB_TOKEN=ghp_...
```

If you use a `.env` file:

```bash
set -a
source .env
set +a
```

## Quick Start

### Analyze a local repo

```bash
uv run phalanx /path/to/repo
```

### Analyze a GitHub repo URL

```bash
uv run phalanx https://github.com/owner/repo
```

### Save markdown + JSON outputs

```bash
uv run phalanx /path/to/repo --output summary.md --json-output full.json
```

### Dry-run estimate (no API calls)

```bash
uv run phalanx /path/to/repo --dry-run --summary-only
```

## Large Repository Workflow

Recommended sequence:

1. Estimate:
```bash
uv run phalanx /path/to/big-repo --dry-run --summary-only
```
2. Full run with artifacts:
```bash
uv run phalanx /path/to/big-repo --output big_summary.md --json-output big_full.json --verbose
```
3. Subsequent incremental check:
```bash
uv run phalanx /path/to/big-repo --diff-only --diff-digest --diff-output big_diff.json
```

## Diff and Manifest Workflow

Phalanx stores run manifests and can compare consecutive runs.

- Manifest store default: `~/.repo_summarizer_cache/manifests`
- Diff JSON includes added/deleted/modified/unchanged + churn hotspots
- Optional prose digest for technical manager consumption

### Normal run with diff output

```bash
uv run phalanx /path/to/repo --diff --diff-output run_diff.json
```

### Diff-only fast path

```bash
uv run phalanx /path/to/repo --diff-only --diff-digest --diff-output run_diff.json
```

Diff-only behavior:

- Uses previous manifest baseline
- Summarizes only added/modified source files
- Reuses prior summaries for unchanged files
- Writes a new manifest and diff artifacts

### Keep only the latest N manifests

```bash
uv run phalanx /path/to/repo --keep-manifests 20
```

## Checkpoint and Resume

Long runs are checkpointed by phase.

- Checkpoint default: `~/.repo_summarizer_cache/checkpoints`
- Resume is enabled by default

Examples:

```bash
# Explicit checkpoint location
uv run phalanx /path/to/repo --checkpoint-dir ~/.repo_summarizer_cache/checkpoints

# Disable resume
uv run phalanx /path/to/repo --no-resume
```

## Important CLI Flags

### Core output

- `--output PATH`: write markdown report
- `--json-output PATH`: write structured JSON report
- `--summary-only`: print final summary only

### Scope and limits

- `--max-files N`: source file guard (default `10000`)
- `--exclude-dir DIR`: repeatable exclude list (directory name or relative path; excluded paths are listed in run stats/report)
- `--skip-docs`: skip doc/config file summarization

### Performance and scale

- `--max-concurrent N`: baseline concurrency
- `--haiku-concurrency N`: Haiku call limit
- `--sonnet-concurrency N`: Sonnet call limit
- `--deep-mode-threshold N`: enable deep mode at file-count threshold
- `--l3-chunk-size N`: max files per L3 chunk
- `--l4-cluster-size N`: max modules per L4 cluster
- `--l1-batch-size N`: L1 small-unit batch size
- `--l1-batch-threshold N`: max source chars for L1 batch eligibility
- `--progress-heartbeat-secs N`: heartbeat interval for long-running phase logs in `--verbose` mode

### Checkpointing

- `--checkpoint-dir PATH`
- `--resume` / `--no-resume`

### Diff and manifests

- `--diff` / `--no-diff`
- `--diff-output PATH`
- `--diff-digest` / `--no-diff-digest`
- `--diff-only`
- `--since RUN_ID_PREFIX`
- `--manifest-dir PATH`
- `--keep-manifests N`

### GitHub source selection

- `--branch NAME`
- `--ref SHA_OR_TAG`
- `--github-token TOKEN`

## Output Artifacts

Depending on flags, you can get:

- Markdown report (`--output`)
- Full JSON summary (`--json-output`)
- Diff JSON (`--diff-output`)
- Optional prose diff digest (in report output / JSON field)

JSON includes:

- `final_summary`
- `module_summaries`
- `directory_summaries`
- `file_summaries`
- `doc_summaries`
- `stats`
- Optional `diff` and `diff_digest`

## Caching

Summary cache default: `~/.repo_summarizer_cache/summaries.json`

- Repeated runs with unchanged content should get high cache hit rates.
- `--no-cache` forces fresh summarization.

## Troubleshooting

### `ANTHROPIC_API_KEY` missing

Set environment variable or pass `--api-key`.

### GitHub clone failures

- Ensure network/DNS access
- Ensure `git` is installed
- For private repos, set `GITHUB_TOKEN` or pass `--github-token`

### Permission errors on default cache paths

Override locations:

```bash
uv run phalanx /path/to/repo \
  --cache-dir /tmp/phalanx-cache \
  --checkpoint-dir /tmp/phalanx-checkpoints \
  --manifest-dir /tmp/phalanx-manifests
```

### No previous manifest for diff-only

Run a normal diff-enabled analysis once first:

```bash
uv run phalanx /path/to/repo --diff
```

## Development

### Run tests

```bash
uv run --with pytest pytest -q
```

### Type/syntax sanity check

```bash
uv run python -m py_compile repo_summarizer.py orchestrator.py agents.py parser.py prompts.py
```

## Repository Layout

```text
phalanx/
  repo_summarizer.py   # CLI entry point
  orchestrator.py      # Pipeline orchestration, deep mode, checkpoints
  agents.py            # LLM calls, batching, tool-use loop
  parser.py            # AST extraction
  prompts.py           # Prompt templates
  checkpoint.py        # Checkpoint persistence
  manifest.py          # Manifest and diff models/store
  diff_report.py       # Diff JSON + digest generation
  cache.py             # Content-addressed summary cache
  github.py            # Git clone helpers
  tests/               # Unit tests
```
