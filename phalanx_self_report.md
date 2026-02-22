# Repository Summary: `phalanx`

*Analyzed 20 files across 2 modules | Languages: python | 234 code units parsed*

---

## Executive Summary

Phalanx is a recursive, hierarchical codebase summarizer written in Python that converts raw source repositories into structured, multi-granularity documentation using a layered LLM synthesis pipeline. It ingests code from local directories or GitHub URLs, parses it with tree-sitter into structured code units, and progressively distills those units through five abstraction levels—from individual functions up to a final repository-wide narrative—using Anthropic's Claude models. The system is architected for production-grade reliability, with built-in checkpointing, content-addressed caching, incremental (diff-only) re-indexing, and cost estimation, making it practical for large real-world codebases.

## System Architecture

Phalanx's architecture is a strict bottom-up pipeline organized in five layers (L1–L5):

- **L1 – Unit summaries**: Individual functions, classes, and code blocks extracted by `parser.py` are batch-summarized in parallel using Claude Haiku for throughput and cost efficiency.
- **L2 – File summaries**: Unit summaries are aggregated into per-file digests, also via Haiku.
- **L3 – Directory summaries**: File summaries are chunked and composed into directory-level overviews using Claude Sonnet for higher reasoning quality.
- **L4 – Module summaries**: Directory summaries are clustered and synthesized into module-level documentation, again with Sonnet.
- **L5 – Final synthesis**: An agentic loop allows the model to dynamically query module and file metadata via tool use before generating the final cross-cutting repository report.

A two-tier caching system keyed on source content prevents redundant API calls across runs. Concurrency is governed by per-model semaphores, and exponential backoff handles API rate limiting. This layered design cleanly separates concerns: parsing, summarization, orchestration, and persistence are each isolated into their own modules.

## Key Components

**`parser.py`** is the ingestion layer. Using tree-sitter grammars for seven languages (Python, TypeScript/JavaScript, Rust, Go, C, C++), it traverses syntax trees to extract `CodeUnit` and `FileUnits` data structures annotated with docstrings, comments, and normalized metadata. Discovery functions traverse directory trees while respecting exclusion patterns.

**`agents.py`** is the summarization engine. The `Summarizer` class manages the full L1–L5 abstraction chain, delegating to layer-specific prompt templates in `prompts.py`. A `CostTracker` dataclass accumulates token usage and estimated spend across both Haiku and Sonnet variants, providing cost transparency throughout a run.

**`orchestrator.py`** is the coordination hub. `RepoOrchestrator` drives file discovery, feeds batches to `Summarizer`, aggregates results up through directory and module levels, and triggers final synthesis. It supports deep mode for exceptionally large repositories, dry-run mode for pre-run cost estimation, and checkpoint-driven resumability so any phase can restart without reprocessing completed work.

**`manifest.py`** provides change tracking and audit persistence. `ManifestStore` saves and loads `RunManifest` objects that record per-file content hashes, run timing, and cost metadata. `compute_repo_changes` diffs successive manifests to classify files as added, modified, deleted, or unchanged, enabling incremental re-indexing that skips unmodified files entirely.

**`repo_summarizer.py`** is the public CLI entry point. It parses arguments, handles GitHub credential resolution, and coordinates `RepoOrchestrator` for both full and diff-only analysis modes. It also assembles the final Markdown and JSON output reports, normalizing LLM-generated headings to ensure consistent formatting.

**`diff_report.py`** bridges manifest diffs to human-readable output, serializing `ManifestDiff` objects to JSON and optionally generating natural-language change digests via an LLM call.

## Operational Model

A typical full run proceeds as: `repo_summarizer.py` → `RepoOrchestrator` discovers files → `parser.py` extracts code units → `agents.py` summarizes units in parallel batches → results are aggregated up through directories and modules → `Summarizer.synthesize_final()` runs an agentic tool-use loop for the repository overview → Markdown and JSON reports are emitted and a `RunManifest` is persisted. On subsequent runs, `ManifestStore` detects only changed files, and diff-only mode routes just those files through the pipeline, merging new summaries with cached prior context. Checkpoints allow interrupted runs to resume from the last completed phase rather than restarting from scratch.

## Risks & Limitations

The system carries several inherent risks. **LLM non-determinism** means summaries can vary between runs even for identical inputs, complicating diff reliability. **Cost unpredictability** is a concern for very large repositories; while dry-run mode and `CostTracker` mitigate this, multi-layer Sonnet calls on huge codebases can be expensive. **Language coverage** is fixed to seven languages via static tree-sitter grammars; unsupported languages are silently skipped. **Context window pressure** at L3–L5 is managed by chunking, but unusually large directories or modules could still exceed model context limits. Finally, **GitHub rate limits and API availability** represent external failure modes for remote ingestion that the retry logic only partially absorbs.

## Testing Coverage Snapshot

The `tests/` directory provides comprehensive mock-based coverage across all major subsystems. Dedicated test files target: source parsing and file discovery (`test_parser.py`), manifest persistence and diffing (`test_manifest.py`, `test_diff_report.py`), checkpoint recovery (`test_checkpoint.py`, `test_orchestrator_checkpoint.py`), LLM batching optimization (`test_agents_batching.py`), agentic tool-use loops (`test_agents_tool_use.py`), deep mode behavior (`test_orchestrator_deep_mode.py`), progress reporting (`test_orchestrator_progress.py`), and diff-only summarization (`test_repo_summarizer_diff_only.py`). All tests use lightweight fakes and mock doubles to remain independent of live LLM APIs and filesystems, ensuring deterministic, fast execution. The breadth of coverage is strong; the primary gap is the absence of end-to-end integration tests exercising real API calls or full repository ingestion.

---

## Appendix A: Module Summaries

### `phalanx [root files]`

This root directory implements a complete repository analysis and summarization system called Phalanx, spanning the full application stack from source code parsing through LLM-powered multi-level summarization to CLI output. The files form a layered dependency chain: `parser.py` discovers and extracts structured code units from source files, `agents.py` drives concurrent LLM calls using prompts from `prompts.py` to summarize those units at progressively higher abstraction levels (L1–L5), and `orchestrator.py` coordinates the full pipeline with checkpointing (`checkpoint.py`), caching (`cache.py`), and manifest tracking (`manifest.py`) to enable fault-tolerant, incremental re-indexing across runs. Supporting infrastructure includes `github.py` for remote repository ingestion, `diff_report.py` for serializing and narrating change reports, and `manifest.py` for detecting file-level deltas between indexing runs to avoid redundant reprocessing. The primary public interface is `repo_summarizer.py`, which serves as the CLI entry point and top-level workflow orchestrator, accepting GitHub URLs or local paths and emitting structured markdown and JSON summarization reports. The `tests/` subdirectory provides comprehensive unit and integration coverage of every subsystem using mock-based isolation, validating the correctness of parsing, caching, checkpointing, batching, and orchestration behaviors without requiring live LLM or filesystem dependencies.

### `tests`

The `tests` directory constitutes the automated test suite for the Phalanx repository analysis and summarization system, collectively validating the correctness of every major subsystem across the application. The files share a common testing philosophy—constructing lightweight mock doubles (`_Dummy*`, `_Fake*`) and factory functions (`_unit`, `_mk_manifest`, `_manifest`) to isolate individual components from external dependencies like LLM APIs, file systems, and orchestration infrastructure, enabling deterministic and fast unit tests. The suite spans the full vertical stack of the application: source code parsing and file discovery (`test_parser.py`), manifest persistence and diffing (`test_manifest.py`, `test_diff_report.py`), checkpoint-based recovery (`test_checkpoint.py`, `test_orchestrator_checkpoint.py`), API batching optimization (`test_agents_batching.py`), agent tool-use orchestration (`test_agents_tool_use.py`), and high-level orchestrator behaviors including deep mode, progress reporting, and diff-only summarization (`test_orchestrator_deep_mode.py`, `test_orchestrator_progress.py`, `test_repo_summarizer_diff_only.py`). There is no single entry point; the directory is consumed as a whole by a test runner such as `pytest`, with each file independently exercising a distinct module or feature boundary. Together, these tests establish confidence that the system's core responsibilities—traversing repositories, extracting structure, summarizing code incrementally, and persisting progress—function correctly in isolation and across integration boundaries.

---

## Appendix C: File Summaries

### `agents.py`
*/home/almcl/repos/phalanx/agents.py*

The file implements a hierarchical code summarization system centered around the `Summarizer` class, which orchestrates concurrent LLM API calls to Anthropic's Claude models for generating multi-level repository documentation. It defines the `CostTracker` dataclass to monitor token usage and estimate expenses across Haiku/Sonnet variants, and the `Summarizer` class which manages a five-layer abstraction pipeline: individual code units (L1) are batch-processed and cached, combined into file summaries (L2), aggregated into directory summaries (L3), composed into module summaries (L4), and finally synthesized into a repository overview (L5) using an agentic loop with tool use. The core pattern involves constructing layer-specific prompts, leveraging a two-tier caching system keyed on source content to reduce API calls, enforcing per-model concurrency limits via semaphores, and implementing retry logic with exponential backoff for rate limiting. Key methods like `summarize_units_batched()`, `summarize_file()`, and `synthesize_final()` form a dependency chain where smaller units are combined and re-summarized at progressively higher levels of abstraction, with the final synthesis step using tool invocation to allow the model to dynamically query module and file metadata before generating the output. The entire design optimizes for throughput and cost efficiency while maintaining cache coherence across multiple summarization requests.

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

This file implements a **prompt template factory** for a hierarchical code summarization system, constructing structured LLM prompts at five escalating levels of abstraction—from individual code units (L1) through files (L2), directories (L3), modules (L4), and finally repository-wide analysis (L5). Each function accepts metadata and pre-computed summaries from the level below, formats them according to domain-specific instructions, and returns a complete prompt string that guides an LLM to synthesize that layer's summary with consistent structure, word count targets, and formatting constraints. The file encodes the core abstraction hierarchy of a repository indexing system, where prompts act as the interface between hierarchical aggregation logic and LLM intelligence, enabling progressive summarization from source code up to architectural overviews. Beyond the main pipeline, it provides specialized prompts for auxiliary tasks like diff digestion, documentation file summaries, and deep-mode chunked analysis, each following the same pattern of metadata formatting and instruction templating. This utility design—where all functions are pure template generators with no external dependencies—makes the prompts portable, testable, and easily modifiable as LLM-based summarization strategies evolve.

### `repo_summarizer.py`
*/home/almcl/repos/phalanx/repo_summarizer.py*

`/home/almcl/repos/phalanx/repo_summarizer.py` is the CLI entry point and orchestration layer for repository summarization, responsible for parsing command-line arguments, managing the end-to-end workflow from repository input (GitHub URLs or local paths) through analysis to formatted output. The file implements a pipeline pattern where `main()` coordinates the `RepoOrchestrator` to analyze repositories and generate summaries, while supporting both full and diff-only modes for resumable/incremental analysis. Key functions like `summarize_changed_files()` parallelize file-level summarization via async gathering, `build_markdown_report()` structures the final output by hierarchically composing metadata, module summaries, and statistics, and `build_diff_only_summary()` creates lightweight updates by merging change metrics with prior context. The file also provides utility functions such as `_normalize_final_summary_markdown()` that clean up LLM-generated markdown to enforce consistent heading levels and remove boilerplate, ensuring output integrates cleanly into the report structure. Overall, this file acts as the public interface and workflow orchestrator, translating user intent into repository analysis while managing credentials, concurrency, caching, and multiple output formats (markdown, JSON, diff reports).

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
- Code units parsed: 234
- Excluded directories: 5
- Doc files summarized: 0
- Directories summarized: 2
- Modules summarized: 2
- Languages: python
- API calls: 12
- Cache hits: 274
- Input tokens: 22,813
- Output tokens: 3,728

### Excluded Directory Paths

- `.git`
- `.pytest_cache`
- `.venv`
- `__pycache__`
- `tests/__pycache__`