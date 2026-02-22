# Phalanx Roadmap

Last updated: 2026-02-22

## Goals

- Make Phalanx reliable for large-repo, repeatable analysis.
- Improve report quality and consistency for technical readers.
- Prepare enterprise-ready deployment options.

## Current Status

- Core hierarchical pipeline (L1-L5) is implemented and stable.
- Diff/checkpoint/cache workflows are implemented.
- Self-report structure has been tightened (executive-first layout + appendices).

## Near-Term Priorities (Next 2-4 Weeks)

### 1) Provider Expansion (Proposed)

- [ ] Add provider abstraction for LLM backend selection.
- [ ] Support Anthropic direct + Amazon Bedrock + Google Vertex AI.
- [ ] Add model mapping per provider for Haiku/Sonnet tiers.
- [ ] Add provider-specific auth/env validation with clear CLI errors.
- [ ] Add docs for provider setup examples.

### 2) Report Quality

- [ ] Enforce strict report section contract in all L5 paths.
- [ ] Reduce redundancy between main report and per-file appendices.
- [ ] Add optional "brief report" mode for faster executive consumption.

### 3) Operational Hardening

- [ ] Improve retry classification (rate limit vs transient vs fatal errors).
- [ ] Add run-level metadata for easier audit/debug.
- [ ] Add configurable hard limits for spend/tokens/calls per run.

## Mid-Term Backlog (1-2 Months)

- [ ] Enterprise auth patterns (AWS role assumption, GCP workload identity).
- [ ] Policy controls by tenant/workspace (allowed providers/models).
- [ ] Integration test matrix across provider backends.
- [ ] Optional web UI for report browsing and run history.
- [ ] Packaging/versioning cleanup for first public beta.

## Done Recently

- [x] Generated and linked `phalanx_self_report.md` from `README.md`.
- [x] Removed local artifact noise from self-analysis runs with `--skip-docs`.
- [x] Updated report template to executive-first structure:
  - `Executive Summary`
  - `System Architecture`
  - `Key Components`
  - `Operational Model`
  - `Risks & Limitations`
  - `Testing Coverage Snapshot`

## Open Decisions

- [ ] Do we ship Bedrock + Vertex in one release, or Bedrock first then Vertex?
- [ ] Do we keep one global model mapping or expose per-layer model overrides in CLI?
- [ ] What is the minimum enterprise scope for "v1" (provider support only vs policy controls too)?
