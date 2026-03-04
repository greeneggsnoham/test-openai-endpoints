# API Endpoint Test Plan

## Summary
Expand `test-endpoints.py` into a structured, capability‑gated audit suite that exercises every endpoint family listed in the API Reference Overview. Each test will make a real API call, with best‑effort cleanup and minimal, low‑cost payloads. Realtime endpoints will be included but gated behind explicit env flags/credentials.

## Assumptions & Defaults
- Scope: all endpoints in the API Reference Overview.
- Cost/Time: full lifecycle where feasible (create + retrieve + list + cancel/delete); but long‑running jobs (video, fine‑tune, evals) use short polling windows and then cancel/skip if still running.
- Cleanup: best‑effort cleanup in `finally` blocks; log cleanup failures without failing the run.
- Realtime endpoints included, gated by `OPENAI_REALTIME=1`.

## Public API / Interface Changes
- Add configuration via environment variables:
  - `OPENAI_API_KEY` (required)
  - `OPENAI_ORG` (optional)
  - `OPENAI_REALTIME` (`0/1`)
  - `OPENAI_RUN_EXPENSIVE` (`0/1`) to allow video/fine‑tune/evals to proceed
  - `OPENAI_CLEANUP_STRICT` (`0/1`) to optionally fail on cleanup failures
  - `OPENAI_POLL_SECONDS` (e.g., `10`) and `OPENAI_POLL_INTERVAL` (e.g., `2`)
- Add CLI args for selecting suites (optional): `--only responses,files` and `--skip admin,realtime`.

## Implementation Plan
1. Refactor structure
- Split tests into endpoint families (e.g., `test_responses`, `test_files`, `test_batches`, …).
- Add a lightweight runner that:
- prints a table header
- reads env flags
- executes suites in order
- collects failures and prints a final summary.

2. Add common helpers
- `log_result(name, status, note)`
- `safe_call(label, fn)` to standardize try/except and error truncation
- `poll_until(get_fn, done_pred, timeout_s, interval_s)` for long‑running tasks
- `best_effort_cleanup(fn)` which logs cleanup failures, optionally raises if `OPENAI_CLEANUP_STRICT=1`

3. Endpoint family coverage (real API calls)

Responses
- `create` (non‑streaming and background)
- `retrieve`, `delete`
- `input items list`, `input items count`, `cancel`, `compact`

Streaming events
- Create a streaming response and consume a small token stream

Conversations & Items
- `conversations.create`, `retrieve`, `update`, `delete`
- `items.create`, `retrieve`, `list`, `delete`

Chat Completions (legacy)
- `chat.completions.create` (already present)

Audio
- speech create (TTS) + transcription create
- translation create
- voice create + voice consents CRUD (skip if not supported by account, but make a real call)

Images
- generate via `image_generation` tool
- edit and variation (use temporary files)
- image streaming events if supported

Videos
- `videos.create`, `retrieve`, `list`, `delete`, `content`, `remix` (if available)
- Short polling window, then cancel/delete if still processing

Embeddings
- `embeddings.create` with a minimal string

Evals
- `evals.create`, `retrieve`, `list`, `delete`
- `eval runs.create`, `retrieve`, `list`, `cancel`
- `eval output items` (if run completes quickly; otherwise cancel)

Fine‑tuning
- `fine_tuning.jobs.create` with a tiny training file (JSONL)
- `jobs.retrieve`, `jobs.list`, `jobs.cancel`
- `checkpoints.list`, `events.list`, `permissions` if applicable
- Use `OPENAI_RUN_EXPENSIVE=1` gating

Files
- `files.create`, `retrieve`, `list`, `content`, `delete`

Uploads
- `uploads.create`, `parts.create`, `uploads.complete`
- `uploads.cancel` for cleanup path

Batches
- `batches.create`, `retrieve`, `list`, `cancel`

Models
- `models.list`, `models.retrieve` (delete only if you own the model; otherwise skip)

Moderations
- `moderations.create` with a minimal test input

Vector stores
- `vector_stores.create`, `retrieve`, `list`, `delete`, `search`
- `vector_store.files.create`, `retrieve`, `list`, `delete`, `content`
- `vector_store.file_batches.create`, `retrieve`, `list`, `cancel`

ChatKit
- `chatkit.sessions.create`, `retrieve`, `list`, `delete`
- `chatkit.threads.create`, `retrieve`, `list`, `delete`
- `chatkit.items.create`, `retrieve`, `list`, `delete`

Containers
- `containers.create`, `retrieve`, `list`, `delete`
- `container.files.create`, `retrieve`, `list`, `delete`, `content`

Skills
- `skills.create`, `retrieve`, `list`, `delete`
- `skills.versions.create`, `retrieve`, `list`, `delete`, `content`

Realtime (gated by `OPENAI_REALTIME=1`)
- `realtime.client_secrets.create` (if available)
- `realtime.calls.*` minimal create/accept/hangup flow
- consume at least one client/server event on a short session


4. Test data and cleanup strategy
- Generate temporary files with `tempfile`; delete in `finally`.
- Use short prompts/inputs to control cost.
- Cancel/cleanup background jobs (videos, evals, fine‑tunes, batches) after a short polling window.

5. Output and reporting
- Keep the existing table format.
- Add summary: total tests, passed, failed, skipped (with reasons: missing credentials/feature flags).

## Test Cases & Scenarios
- Run with only `OPENAI_API_KEY`:
- All non‑admin tests should execute or skip cleanly if unsupported.
- Run with `OPENAI_REALTIME=1`:
- Realtime tests execute; if websocket/session cannot be established, report failure.
- Run with `OPENAI_RUN_EXPENSIVE=0`:
- Expensive endpoints are skipped with a clear note.
- Run with `OPENAI_CLEANUP_STRICT=1`:
- Cleanup failures are treated as test failures.

## Open Items (for you to confirm later)
- Which models to use for each endpoint if account access is restricted (defaults will be conservative).
- Whether to persist generated artifacts for inspection (default: no).
