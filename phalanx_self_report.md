# Repository Summary: `phalanx`

*Analyzed 20 files across 2 modules | Languages: python | 233 code units parsed*

---

Here is the complete technical summary:

---

# Phalanx: Technical Architecture Summary

## Overview

Phalanx is a recursive, hierarchical codebase summarizer that transforms raw source repositories into structured, multi-granularity documentation using a layered LLM pipeline. Written entirely in Python, it ingests local directories or remote GitHub repositories, parses source files across seven languages via `tree-sitter`, and progressively synthesizes code understanding through five ascending abstraction levels — from individual functions and classes all the way to a holistic repository overview. The system is designed for large, real-world codebases: it is async-native, parallelizes aggressively, caches aggressively, and is fully resumable via checkpoint-driven fault tolerance. Its primary output is a Markdown report accompanied by JSON artifacts, with optional diff reports between successive runs.

---

## Architecture

Phalanx is organized as a five-stage summarization pipeline, each stage feeding its outputs upward as inputs to the next:

- **L1 – Unit Summaries:** Individual functions, classes, and other code units are summarized by Claude Haiku in parallel batches.
- **L2 – File Summaries:** Per-file summaries are synthesized from their constituent unit summaries, also using Haiku.
- **L3 – Directory Summaries:** File summaries are aggregated bottom-up into directory-level narratives using Claude Sonnet, chunked to manage context limits.
- **L4 – Module Summaries:** Directories are clustered and synthesized into module-level overviews, with smart grouping applied to large repositories ("deep mode").
- **L5 – Final Synthesis:** An agentic LLM loop dynamically retrieves module and file summaries via tool invocation before producing a final ~1,200-word repository report.

The architecture cleanly separates **parsing** (tree-sitter ASTs → `CodeUnit` data models), **prompting** (prompt construction per level), **execution** (concurrent Claude API calls with caching), **orchestration** (phase sequencing and checkpointing), and **persistence** (manifests, diffs, checkpoints, and disk cache).

---

## Core Components

**`parser.py`** — The ingestion and discovery layer. Defines the foundational data models (`CodeUnit`, `FileUnits`, `DocFile`) and provides tree-sitter-based parsers for seven languages. Extracts functions, classes, docstrings, and preceding documentation comments into normalized, language-agnostic representations. Discovery utilities (`discover_files`, `discover_doc_files`) traverse directory trees while respecting configurable exclusion patterns.

**`agents.py`** — The LLM execution engine. The `Summarizer` class orchestrates concurrent API calls to Claude Haiku (fast, cheap, L1/L2) and Claude Sonnet (powerful, L3–L5) via per-model semaphores for concurrency control. Implements a two-tier cache (in-memory + disk) for individual units and composite summaries. Adaptive batching groups small code units into single prompts to reduce API round-trips. The `CostTracker` dataclass accumulates token usage and estimates blended costs across models. The `synthesize_final()` method implements a true agentic loop where Claude calls back into the system via tool use to fetch module/file summaries on demand.

**`orchestrator.py`** — The pipeline coordinator. The `RepoOrchestrator` class sequences the five summarization phases, dispatching to `Summarizer` for API work and aggregating results bottom-up. Implements `_discover_and_parse()`, `_summarize_files()`, `_summarize_directories()`, `_summarize_modules_clustered()`, and `_final_synthesis()`. Supports dry-run mode for cost estimation, verbose heartbeat logging, and checkpoint-driven resumability so any phase can be restarted without replaying prior work.

**`repo_summarizer.py`** — The CLI entry point and user-facing coordinator. Parses arguments, resolves local vs. GitHub sources, manages credential validation, and drives the orchestrator. Post-run formatting functions (`build_markdown_report()`, `build_diff_only_summary()`, `summarize_changed_files()`) transform structured `SummaryResult` objects into human-readable Markdown and JSON. Bridges synchronous CLI invocation to the async runtime via `run_cli()`.

**`checkpoint.py`** — Fault-tolerant state persistence. `CheckpointState` tracks completed phases and accumulated summaries for a run. `CheckpointStore` serializes checkpoint state to disk with atomic file-swap writes to prevent corruption and provides `find_latest()` for transparent resume. Phase transitions are validated to prevent invalid state progressions.

**`manifest.py`** — Cross-run change tracking. Defines `RunManifest`, `FileRecord`, `ManifestDiff`, and `RepoChangeSet` dataclasses. `ManifestStore` saves, loads, and diffs manifests between runs, computing per-file content hashes to categorize files as added, modified, deleted, or unchanged. Enables incremental re-indexing by identifying exactly which files changed and supports `compute_churn_hotspots()` for temporal analysis. A `prune()` method enforces manifest retention policies.

**`cache.py`** — A lightweight two-level summary cache. `SummaryCache` pairs a fast in-memory dictionary with JSON file persistence, keyed by SHA-256 hashes of (layer name + content). Survives process restarts and provides basic utilization statistics.

**`diff_report.py`** — Diff serialization and narration. Converts `ManifestDiff` objects to JSON-serializable dictionaries with aggregated statistics, writes them to disk, and optionally generates AI-powered natural language diff digests via an injected `Summarizer` instance.

**`github.py`** — Remote repository acquisition. Handles cloning or fetching GitHub-hosted repositories for local pipeline processing.

**`prompts.py`** — Prompt construction. Assembles LLM-ready prompts appropriate for each summarization level (L1–L5), keeping prompt engineering concerns cleanly separated from execution logic.

---

## Data Flow & Key Interactions

1. `repo_summarizer.py` receives CLI arguments and constructs a `RepoOrchestrator`, injecting a configured `Summarizer` and `SummaryCache`.
2. `RepoOrchestrator` calls `parser.discover_files()` to enumerate eligible source files, then `parser.parse_file()` on each to produce `CodeUnit` lists.
3. `Summarizer.summarize_unit()` (or batched `_summarize_small_batch()`) sends L1 prompts to Claude Haiku concurrently, results are cached by SHA-256 hash.
4. File-level summaries (L2) are composed from unit summaries and fed into directory aggregation (L3, Sonnet), then module clustering (L4, Sonnet).
5. `synthesize_final()` initiates an agentic Claude Sonnet loop: Claude calls registered tools to fetch specific module/file summaries, iterates, then produces the final report.
6. `CheckpointStore` is written after each phase; on resume, completed phases are skipped by loading the existing `CheckpointState`.
7. `ManifestStore` computes a diff between the previous and current run, which `diff_report.py` serializes and optionally narrates.

---

## Technical Patterns & Design Decisions

- **Layered LLM model strategy:** Haiku handles high-volume, low-complexity tasks (L1/L2); Sonnet handles reasoning-intensive synthesis (L3–L5). This two-tier approach balances cost against quality.
- **Agentic tool use at L5:** Rather than stuffing all module summaries into a single massive prompt, Claude is given retrieval tools and autonomously decides what context to pull. This avoids context-window overflow and produces more focused final reports.
- **Adaptive batching:** Small code units (short functions, simple classes) are grouped into single batch prompts to amortize API overhead; large units are summarized individually. This significantly reduces total API call count on typical repositories.
- **Atomic checkpoint writes:** Checkpoints are written via file-swap (write to temp → rename) to guarantee no partial-write corruption, a critical property for long-running jobs.
- **Content-hash caching:** Cache keys incorporate both the summarization layer and the raw content hash, so the same function summarized at L1 vs. L2 gets distinct entries, and any code change automatically invalidates stale summ

---

## Module Summaries

### `phalanx [root files]`

This directory represents the complete implementation of Phalanx, an end-to-end repository analysis and documentation system that transforms raw source code into multi-granularity summaries using Claude LLM APIs. The files form a layered pipeline: `parser.py` discovers and parses source files into structured `CodeUnit` representations, `prompts.py` constructs LLM-ready prompts at each abstraction level, `agents.py` executes concurrent LLM calls with caching and cost tracking, and `orchestrator.py` coordinates these components across five progressive summarization phases (L1–L5) with checkpoint-driven fault tolerance. Supporting infrastructure is provided by `cache.py` and `checkpoint.py` for persistence and resumability, `manifest.py` for tracking cross-run change history and incremental re-indexing, `github.py` for remote repository acquisition, and `diff_report.py` for serializing and narrating changes between runs. The primary public entry point is `repo_summarizer.py`, which exposes a CLI interface that accepts local or GitHub-hosted repositories and orchestrates the full pipeline to produce markdown reports, JSON artifacts, and diff-based summaries. The `tests/` subdirectory provides comprehensive behavioral coverage of all subsystems using mock-based isolation, ensuring correctness without incurring real API costs.

### `tests`

The `tests` directory constitutes the automated test suite for the Phalanx repository analysis and summarization system, collectively validating the correctness of every major subsystem across the application. The files share a common testing philosophy—constructing lightweight mock doubles (`_Dummy*`, `_Fake*`) and factory functions (`_unit`, `_mk_manifest`, `_manifest`) to isolate individual components from external dependencies like LLM APIs, file systems, and orchestration infrastructure, enabling deterministic and fast unit tests. The suite spans the full vertical stack of the application: source code parsing and file discovery (`test_parser.py`), manifest persistence and diffing (`test_manifest.py`, `test_diff_report.py`), checkpoint-based recovery (`test_checkpoint.py`, `test_orchestrator_checkpoint.py`), API batching optimization (`test_agents_batching.py`), agent tool-use orchestration (`test_agents_tool_use.py`), and high-level orchestrator behaviors including deep mode, progress reporting, and diff-only summarization (`test_orchestrator_deep_mode.py`, `test_orchestrator_progress.py`, `test_repo_summarizer_diff_only.py`). There is no single entry point; the directory is consumed as a whole by a test runner such as `pytest`, with each file independently exercising a distinct module or feature boundary. Together, these tests establish confidence that the system's core responsibilities—traversing repositories, extracting structure, summarizing code incrementally, and persisting progress—function correctly in isolation and across integration boundaries.

---

## File Summaries

### `agents.py`
*/home/almcl/repos/phalanx/agents.py*

This file implements a hierarchical code summarization system for repository indexing, centered around the `Summarizer` class which orchestrates concurrent LLM API calls to Claude models with intelligent caching and batching strategies across five abstraction layers (L1–L5). The `CostTracker` dataclass provides cost estimation and metrics reporting for API usage, tracking token consumption across Haiku and Sonnet models with blended pricing calculations. The `Summarizer` manages per-model concurrency control via semaphores, implements a two-tier caching system for individual units and composite summaries, and coordinates adaptive batching—processing small code units in groups via `_summarize_small_batch()` while handling large units individually through `summarize_unit()`. The system progressively synthesizes information from code units → files → directories → modules → repository clusters, with each level delegating to the next via `_call()`, a core middleware method that handles rate limiting, prompt caching, and error recovery. At the top level, `synthesize_final()` implements an agentic loop where Claude dynamically retrieves module and file summaries via tool invocation before generating a comprehensive 1,200-word repository overview, exemplifying a hierarchical abstraction pattern where lower-level summaries feed into progressively higher-level synthesis tasks.

### `cache.py`
*/home/almcl/repos/phalanx/cache.py*

The `cache.py` file implements a persistent two-level caching layer for storing code summaries in the phalanx repository, with `SummaryCache` as the central abstraction that manages both in-memory and disk-backed storage. The class uses a hybrid strategy where fast in-memory access via a dictionary is paired with JSON file persistence to survive process restarts, keyed by deterministic SHA256 hashes that combine layer name and content to ensure uniqueness across different code contexts. Core methods like `get` and `set` provide the primary cache interface, while internal methods `_load` and `_save` handle synchronization between the two storage layers, gracefully handling missing or corrupted files during initialization. The `hash` static method serves as a utility for generating consistent cache keys, and `stats` provides basic observability into cache utilization. Overall, this file implements a simple but pragmatic cache utility pattern designed to avoid redundant code analysis by memoizing summaries across tool invocations.

### `checkpoint.py`
*/home/almcl/repos/phalanx/checkpoint.py*

This file implements a checkpoint-and-resume system for repository indexing operations, enabling long-running summarization tasks to persist and recover their progress across multiple phases. It defines two core abstractions: `CheckpointState`, a dataclass that captures the state of a single indexing run (phases completed, summaries accumulated, metadata), and `CheckpointStore`, a persistence layer that manages serialization, atomic writes, and retrieval of checkpoint files from a cache directory. The `CheckpointState` class provides phase tracking with validation through `mark_phase_complete()`, preventing invalid state transitions, while `CheckpointStore` handles the I/O operations—`save()` uses atomic file swaps to prevent corruption, `load()` gracefully degrades on malformed data, and `find_latest()` enables resuming from the most recent checkpoint for a given repository. Supporting utilities like `_default_phase_state()` and `now_iso_utc()` ensure consistent initialization and standardized timestamp formatting throughout the checkpoint lifecycle, making this file a foundational data model and persistence layer for fault-tolerant, resumable repository analysis workflows.

### `diff_report.py`
*/home/almcl/repos/phalanx/diff_report.py*

This file provides reporting and serialization utilities for manifest diffing operations, serving as a bridge between diff analysis and external output/summarization systems. It defines three complementary functions that handle the complete pipeline of diff representation: `manifest_diff_to_dict` transforms internal diff dataclass objects into JSON-serializable dictionaries with aggregated statistics, `write_diff_json` persists these dictionaries to disk with formatted output for human readability, and `generate_diff_digest` leverages the serialized diff to produce AI-powered natural language summaries via an integrated LLM. The functions follow a utility pattern where each operates on standardized data structures (dictionaries and ManifestDiff objects) without maintaining state, enabling modular composition into larger diff reporting workflows. Together, they enable downstream systems to consume diff information in multiple formats—structured JSON for dashboards, files for archival, and natural language for human review—while delegating LLM concerns to a separate summarizer component that handles tokenization and retry logic.

### `github.py`
*/home/almcl/repos/phalanx/github.py*

This file provides git repository cloning and URL validation utilities, serving as the core abstraction layer for interacting with remote git repositories in the codebase. It defines a custom `CloneError` exception for distinguishing clone-related failures and exports three primary functions that form a validation-and-clone pipeline: `is_github_url` validates whether a string represents a git repository URL across major platforms (GitHub, GitLab, Bitbucket), `normalise_github_url` transforms URLs into a consistent HTTPS format with optional token-based authentication injection, and `clone_repo` orchestrates the actual shallow clone operation by composing validation, normalization, and secure git execution with timeout protection and error scrubbing. The file follows a utility pattern, with each function building on the previous—URL validation precedes normalization, which precedes cloning—enabling callers to safely and consistently fetch remote repositories with authentication while maintaining security through token obfuscation in error messages and temporary directory cleanup on failure.

### `manifest.py`
*/home/almcl/repos/phalanx/manifest.py*

`/home/almcl/repos/phalanx/manifest.py` is a data persistence and analysis layer that records and tracks code summarization runs across repository scans. It defines a hierarchy of dataclasses (`RunManifest`, `FileRecord`, `ModifiedFile`, `ManifestDiff`, `RepoChangeSet`) that model the structure of indexing operations, from individual file metadata to repository-wide change summaries, and provides the `ManifestStore` class as a factory and repository for saving, loading, and querying these manifests from disk. Core utilities like `build_run_manifest` construct manifest objects from summarization results and filesystem state, while `compute_repo_changes` detects file-level deltas between runs by computing content hashes and categorizing files as added/modified/deleted/unchanged. The `ManifestStore` methods (`save`, `load`, `diff`, `compute_churn_hotspots`, `prune`) enable temporal analysis of repositories, allowing consumers to track how codebases evolve across multiple indexing operations and identify high-churn files that warrant special attention. Overall, the file implements a manifest pattern that couples operational audit trails (run timing, costs, counts) with detailed change tracking, enabling efficient incremental re-indexing and historical diff analysis without re-processing unchanged files.

### `orchestrator.py`
*/home/almcl/repos/phalanx/orchestrator.py*

The `orchestrator.py` file implements a multi-level repository summarization pipeline that progressively analyzes source code from granular (individual files) to holistic (entire repository) levels using Claude APIs with checkpointing and concurrency management. Its core abstraction is the `RepoOrchestrator` class, which orchestrates five hierarchical summarization phases (L1–L5) by discovering and parsing source files, batching them through a delegated `Summarizer` instance, and aggregating results bottom-up through directory and module levels before final synthesis into a `SummaryResult` dataclass. Key operations include `_discover_and_parse()` for file collection, `_summarize_files()` and `_summarize_directories()` for hierarchical AST-based summaries, `_summarize_modules_clustered()` for smart grouping of large repositories, and `_final_synthesis()` for cross-cutting repository-level insights, with helper utilities like `_chunked()` and `_new_run_id()` supporting scalable batch processing and resumable checkpoint tracking. The orchestrator follows a checkpoint-driven execution pattern where each phase can be resumed independently, includes a dry-run mode for cost estimation before committing to expensive LLM calls, and provides verbose logging and heartbeat progress updates throughout. Together, these components enable the orchestrator to transform raw repository code into structured, multi-granularity documentation artifacts while gracefully handling large codebases and providing cost visibility and fault tolerance.

### `parser.py`
*/home/almcl/repos/phalanx/parser.py*

This file implements the core parsing and discovery infrastructure for a multi-language code indexing system, defining the fundamental data models (`CodeUnit`, `FileUnits`, `DocFile`) that represent extracted code entities and their metadata. It provides a tree-sitter-based parser (`parse_file`) that recursively traverses syntax trees to extract functions, classes, and other code units with their docstrings and documentation comments, along with a complementary document parser (`parse_doc_file`) for configuration and documentation files. The file also includes utility functions like `_get_node_name`, `_get_node_kind`, and `_extract_preceding_comment` that abstract away language-specific syntax tree details into normalized, human-readable metadata. Supporting this core functionality are discovery functions (`discover_files`, `discover_doc_files`, `_discover_paths`) that efficiently traverse directory trees while respecting exclusion patterns, serving as the file enumeration layer that feeds inputs to the parsers. Overall, this module acts as a factory and data model layer that transforms raw source files on disk into structured, queryable representations of code elements for downstream indexing and analysis.

### `prompts.py`
*/home/almcl/repos/phalanx/prompts.py*

This file implements a prompt-generation utility layer for a hierarchical repository indexing and summarization system, serving as the interface between raw code analysis and language model consumption. It defines a family of specialized prompt-construction functions organized across five abstraction levels—from `l1_unit_prompt` (individual code units) through `l5_final_prompt` (full repository analysis)—each tailored to synthesize summaries at increasing scales of granularity while managing token budgets through selective truncation. The file establishes a clear factory-like pattern where each function accepts structured metadata and prior-level summaries, formats them into human-readable markdown-based prompts with explicit instructions, and returns LLM-ready strings; supplementary functions like `l3_chunk_prompt`, `l3_merge_prompt`, and `l4_cluster_prompt` support alternative hierarchical decomposition strategies for handling large repositories. Beyond core summarization, it provides specialized variants including `diff_digest_prompt` for change analysis, `doc_file_prompt` for configuration/documentation artifacts, and `l5_tool_use_prompt` for agent-based selective exploration, collectively enabling flexible multi-strategy repository understanding. The underlying pattern treats prompt construction as a composition of metadata, prior results, and instruction templates, with consistent emphasis on evidence-based analysis, controlled output length, and semantic coherence across hierarchical boundaries.

### `repo_summarizer.py`
*/home/almcl/repos/phalanx/repo_summarizer.py*

This file serves as the CLI entry point and orchestration layer for a repository analysis system, translating command-line arguments into structured summarization workflows. It implements a comprehensive pipeline that accepts local or GitHub-hosted repositories, delegates deep analysis to a `RepoOrchestrator` instance, and generates multiple output formats (markdown reports, JSON, and diff-based summaries) that synthesize file-level analysis, performance metrics, and documentation into human-readable artifacts. The core pattern is a **coordinator/bridge pattern**, where `main()` handles configuration, credential validation, and checkpoint management, while specialized formatting functions—`build_markdown_report()`, `build_diff_only_summary()`, and `summarize_changed_files()`—transform the orchestrator's structured results into consumable outputs for different use cases (full reports, incremental diffs, and individual file summaries). The file manages both synchronous concerns (argument parsing, file I/O, directory handling) and asynchronous operations (parallel file summarization via `asyncio.gather`), with `run_cli()` bridging the synchronous CLI invocation to the async runtime. Through its integration of caching, resumable checkpoints, and cost-estimation modes, this file enables flexible, efficient repository analysis while abstracting away the complexity of API orchestration from end users.

### `test_agents_batching.py`
*/home/almcl/repos/phalanx/tests/test_agents_batching.py*

This test file validates the batching optimization layer in a code summarization system, specifically verifying that the `summarize_units_batched` method correctly groups small CodeUnit objects for efficient API processing while gracefully handling edge cases. The file employs a factory function (`_unit`) to construct consistent test fixtures and three complementary test cases that cover the happy path (small units batched together), error resilience (fallback to individual processing on JSON parse failures), and threshold enforcement (large units bypassed from batching). Together, these tests ensure the batching pipeline's core responsibility—reducing API overhead by consolidating small summarization requests—functions correctly while maintaining correctness when batching is inappropriate or fails, establishing confidence in the optimization's transparency to downstream consumers.

### `test_agents_tool_use.py`
*/home/almcl/repos/phalanx/tests/test_agents_tool_use.py*

This test file validates agent tool-use functionality in the Phalanx codebase by providing a comprehensive mock infrastructure for simulating Anthropic API interactions without making real requests. It defines a family of mock classes—`_FakeResponse`, `_FakeMessagesAPI`, and `_FakeClient`—that collectively implement a FIFO response queue pattern, allowing tests to control API behavior and inspect call arguments for assertion-based verification. Supporting factory functions (`_tool_use_block`, `_text_block`, `_usage`) construct realistic mock response objects that mimic the structure of actual API responses. The test suite then leverages this infrastructure to verify three critical agent behaviors: correct extraction of final responses from multi-turn tool-use loops, proper filtering of intermediate planning text while preserving final summaries, and accurate partial-match file lookups during tool execution. Together, these components demonstrate a testing pattern that isolates agent orchestration logic from external API dependencies while maintaining high fidelity to real response structures.

### `test_checkpoint.py`
*/home/almcl/repos/phalanx/tests/test_checkpoint.py*

This file provides comprehensive unit tests for the `CheckpointStore` class, validating its core responsibility of persisting and retrieving checkpoint state across the application lifecycle. It employs a test fixture helper (`_state`) that constructs fully populated `CheckpointState` objects to standardize test data across multiple scenarios, establishing a consistent pattern for checkpoint creation. The test suite exercises the complete checkpoint workflow—from atomic save operations with proper cleanup to load operations that handle both valid and corrupted JSON gracefully—ensuring data integrity through serialization boundaries. Additional tests validate the `find_latest()` query method's ability to locate the most recent checkpoint for a specific repository while filtering out unrelated checkpoints, demonstrating that the `CheckpointStore` functions as a robust persistence layer for tracking analysis progress across runs. Together, these tests confirm that the store implements safe concurrent writes (via atomic replace semantics), defensive deserialization, and precise retrieval logic, making it a reliable foundation for checkpoint-based recovery and state management.

### `test_diff_report.py`
*/home/almcl/repos/phalanx/tests/test_diff_report.py*

This test module validates the diff reporting functionality in the phalanx repository by providing comprehensive test coverage for manifest comparison and digest generation workflows. It defines a `_FakeSummarizer` test double that intercepts and records summarization prompts while returning fixed responses, enabling verification of summarization behavior without external dependencies. The file uses factory functions (`_manifest` and `_run_digest`) to construct minimal but realistic test fixtures—including `RunManifest` objects and `ManifestDiff` comparisons—that feed into the core functionality being tested. Two primary test functions verify that the diff reporting system correctly serializes manifest changes into structured payloads with accurate file operation counts and that it properly orchestrates the summarizer to generate digests containing relevant run identifiers from previous and current builds. The overall pattern demonstrates a typical unit testing approach using mocks and factories to isolate the diff reporting logic and assert both its data transformation accuracy and correct interaction with external dependencies.

### `test_manifest.py`
*/home/almcl/repos/phalanx/tests/test_manifest.py*

This test file validates the core functionality of the manifest management system used to track repository state across sequential runs. It exercises key abstractions like `RunManifest` and `ManifestStore`, testing their ability to persist repository snapshots (files, metadata, costs), detect changes between runs, and implement maintenance policies. The tests follow a factory pattern (via `_mk_manifest`) to construct test fixtures efficiently, then validate critical workflows: round-trip serialization and diffing of manifests, file-level change detection (additions/deletions/modifications), churn analysis to identify frequently-modified hotspots, and pruning policies to retain only recent manifests. Together, these tests ensure that the manifest system reliably captures repository evolution, enables incremental analysis by comparing consecutive states, and maintains a manageable historical record through configurable retention strategies.

### `test_orchestrator_checkpoint.py`
*/home/almcl/repos/phalanx/tests/test_orchestrator_checkpoint.py*

This file provides comprehensive test infrastructure for validating the checkpoint and resume functionality of `RepoOrchestrator` by defining two lightweight mock classes—`_DummyTracker` and `_DummySummarizer`—that instrument method invocation counts and return deterministic outputs without performing actual summarization work. The `_DummyTracker` simulates token and API call tracking with a stub `report()` method, while `_DummySummarizer` implements the full summarizer interface with five instrumented async methods (`summarize_file`, `summarize_doc_files_parallel`, `summarize_directory`, `summarize_module`, `synthesize_final`) that each increment call counters and generate formatted string outputs based on input metadata, enabling tests to verify both the orchestrator's behavior and the order of method invocations. The two test cases—`test_orchestrator_resume_skips_completed_phases` and `test_orchestrator_resume_skips_individual_file_nodes`—use these mocks alongside a simple `_write_file` utility to create fixture repositories and validate that the orchestrator correctly persists checkpoints, skips already-processed phases and files on resume, and produces identical results across runs, establishing that the checkpoint mechanism properly restores state without redundant processing.

### `test_orchestrator_deep_mode.py`
*/home/almcl/repos/phalanx/tests/test_orchestrator_deep_mode.py*

This test file validates the orchestrator's deep mode functionality by providing a comprehensive mock instrumentation suite for testing hierarchical repository summarization workflows. It defines two key test doubles—`_DummyTracker` (a minimal progress tracker stub) and `_DummySummarizer` (a fully-instrumented mock summarizer with call counters for every summarization stage)—that enable fine-grained assertion on orchestrator behavior without invoking actual LLM operations. The `_DummySummarizer` class implements the complete summarizer interface across file, document, directory, module, and cluster summarization stages, with each method incrementing dedicated call counters and returning predictable formatted strings, allowing tests to verify both invocation frequency and call sequencing. Three test cases leverage these mocks to validate distinct orchestrator scenarios: `test_deep_mode_uses_chunked_l3` confirms that chunking parameters correctly partition summarization work across the hierarchy, `test_dry_run_skips_all_summarizer_calls` ensures dry-run mode provides cost estimates without backend invocation, and `test_dry_run_reports_excluded_directories` verifies that directory exclusion logic properly integrates with dry-run reporting. Together, these components form a test harness that decouples orchestrator logic validation from the underlying summarization implementation, following the test double pattern to enable isolated, deterministic testing of orchestration control flow and configuration parameter handling.

### `test_orchestrator_progress.py`
*/home/almcl/repos/phalanx/tests/test_orchestrator_progress.py*

This file provides comprehensive unit tests for the RepoOrchestrator's Phase 3 file summarization workflow, defining mock helper classes (`_DummyTracker` and `_SlowSummarizer`) that simulate progress tracking and variable-latency file processing to enable isolated testing of orchestration behavior. The `_DummyTracker` dataclass implements a minimal progress-tracking interface with token/API counters and a stub report method, while `_SlowSummarizer` acts as a configurable test double that injects artificial delays and optional failures into the summarization pipeline to exercise both success and error paths. The two test functions (`test_phase3_logs_heartbeat_and_progress` and `test_phase3_failure_includes_file_path`) use these fixtures to verify critical orchestrator responsibilities: that heartbeat logging occurs at configured intervals during long-running concurrent operations, and that failure context (file path) is properly preserved and reported when summarization errors occur. This pattern of minimal mock objects combined with async test scenarios follows a common testing strategy for validating complex concurrent workflows without requiring real language model calls or heavyweight instrumentation.

### `test_parser.py`
*/home/almcl/repos/phalanx/tests/test_parser.py*

This test file comprehensively validates the parsing and file discovery infrastructure for the Phalanx repository indexer across multiple programming languages (C, C++, and Rust). It exercises the core `parse_file()` function to verify that language-specific code units (functions, structs, namespaces, templates, and test modules) are correctly extracted with appropriate metadata, while also testing the file discovery system's ability to classify files by type (code vs. documentation) and apply exclusion filters to dependency directories. The tests follow a consistent pattern of creating temporary files or directories with representative source code, invoking the discovery or parsing functions, and asserting that results contain expected units with correct kinds and names—establishing the baseline correctness of the indexer's ability to traverse repositories and extract structural information from source code.

### `test_repo_summarizer_diff_only.py`
*/home/almcl/repos/phalanx/tests/test_repo_summarizer_diff_only.py*

This test file validates the diff-only summarization functionality of a repository summarizer by employing mock test doubles that isolate the summarization logic from external dependencies. It defines two lightweight fixture classes—`_DummySummarizer` and `_DummyOrchestrator`—that simulate the real summarizer and orchestrator components, enabling deterministic testing without requiring full infrastructure setup. The `_DummySummarizer` tracks invocation counts and returns synthetic summaries keyed by filename, acting as a spy object to verify that the summarizer is called with correct inputs and respects file type filtering. The test suite then exercises two critical behaviors: verifying that `summarize_changed_files` correctly filters to only parseable source files (excluding non-code files like `.txt`), and confirming that `build_diff_only_summary` properly formats and includes file change counts in its output. Together, these tests follow a mock object pattern to ensure the diff-only summarization pathway correctly identifies and summarizes only relevant code files while accurately representing change statistics.

---

## Analysis Stats

- Files analyzed: 20
- Code units parsed: 233
- Excluded directories: 5
- Doc files summarized: 0
- Directories summarized: 2
- Modules summarized: 2
- Languages: python
- API calls: 7
- Cache hits: 278
- Input tokens: 16,449
- Output tokens: 3,535

### Excluded Directory Paths

- `.git`
- `.pytest_cache`
- `.venv`
- `__pycache__`
- `tests/__pycache__`