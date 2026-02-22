# Phalanx v2 Progress

## Current Slice (P0 foundations)
- [x] Add C/C++ parser support in `parser.py`
- [x] Exclude Rust `mod tests` wrapper units while preserving individual test functions
- [x] Add parser unit tests for C/C++ extraction and Rust test-wrapper exclusion
- [x] Run tests and sanity checks locally

## Next Slice (queued)
- [x] Add `checkpoint.py` with atomic save/load/find_latest/phase tracking
- [x] Add prompt additions for chunk/merge/tool-use (`prompts.py`)
- [x] Add tiered semaphores + L1 prompt extraction + small-unit batching (`agents.py`)

## Upcoming
- [x] Wire checkpoint save/resume into `orchestrator.py`
- [x] Add deep-mode orchestration (`l3` chunking + `l4` clustering + dry-run)
- [x] Add CLI flags for deep mode / batching / resume

## Remaining PRD Items
- [x] Implement L5 tool-use agent loop (`list_modules`, `get_module_summary`, `get_file_summary`)
- [ ] Add run manifest support (`manifest.py`) with churn tracking helpers
- [ ] Add diff report generation (`diff_report.py`) and digest prompt flow
- [ ] Add CLI diff flags (`--diff`, `--diff-only`, `--diff-output`, `--diff-digest`, `--since`, `--manifest-dir`)
- [ ] Update README for v2 flags and deep-mode behavior
