# Changelog

This file tracks changes made to the `aether` repository by the assistant during
interactive sessions. It is a running log, not a versioned release history.

## 2026-09-18 — Test fixes, server/client removal, full English translation

### Test and lint fixes

- `src/aether/evaluation.py`: fixed `write_comparison()`, which round-tripped its
  own already-validated `EvaluationRun` through `model_dump(mode="json")` and
  then `EvaluationRun.model_validate(...)`. Under the model's `strict=True`
  config, re-validating a JSON-dumped `tuple[SessionMeasurement, ...]` field
  (which becomes a `list` after JSON dumping) raised a `ValidationError`. Fixed
  by re-validating that specific, already-trusted round trip with
  `strict=False`.
- `src/aether/server.py` (later removed, see below): fixed
  `test_oversized_control_is_protocol_error`. The WebSocket's
  `max_msg_size` was set to exactly the transport ceiling, so aiohttp itself
  closed oversized control messages with a raw close code `1009` before the
  application ever saw them, instead of returning the intended
  `session.error` / `PROTOCOL_ERROR` JSON reply. Fixed by introducing an
  explicit application-level `MAX_CONTROL_BYTES` check (performed by our own
  code before `json.loads`) and raising the transport-level ceiling
  (`_MAX_FRAME_BYTES`) enough that oversized control messages reach the
  application instead of being cut off by the transport.
- Ruff cleanup: import ordering in `src/aether/cli.py`, a lambda-assignment
  rewritten as a `def` in `tests/unit/test_evaluation.py`, and an over-long
  line in `notebooks/aether_colab.ipynb` split into a local variable. Ran
  `ruff format .` across the repository (7 previously-unformatted files, all
  new/untracked at the time).
- Full local check suite (`pytest`, `ruff check`, `ruff format --check`,
  `mypy`) passes after these fixes.

### Removed the live server, local client, and wire protocol

Per a scope decision to focus exclusively on the dialogue model/generator for
this research phase rather than a live serving product, removed the live
WebSocket serving path entirely:

- Deleted `src/aether/server.py`, `src/aether/client.py`,
  `src/aether/protocol.py`, and their tests
  (`tests/unit/test_server.py`, `tests/unit/test_client.py`,
  `tests/unit/test_protocol.py`).
- Removed the `serve` and `talk` CLI subcommands and their argument parsers
  from `src/aether/cli.py`, along with the now-unused `asyncio`/`os` imports.
- Removed the `aiohttp` runtime dependency and the `audio` dependency-group
  (`sounddevice` + `soundfile`) from `pyproject.toml`; `soundfile` remains
  available where still needed (dataset/backend/trainer) under the `model`
  group. Regenerated `uv.lock` accordingly.
- Verified no other module imports the removed files (`protocol.py` was only
  used by the deleted server/client and their tests); full test suite passes
  at 202 tests (down from 235, consistent with removing the deleted
  server/client/protocol test files).

### Full English translation and tone pass across documentation

All prose in the repository was in Russian, much of it phrased as
second-person imperative "user instructions" (e.g. "select this," "run
that"). Since this is internal research documentation describing a system's
design and decisions rather than a product manual for an end user, every file
was translated to English and rewritten in a descriptive/academic register
(facts and decisions stated in the third person, not commands). Work was
parallelized across subagents, one per file or small file group; each
translation was faithful to the original technical content, numbers, tables,
and structure.

Files translated (pure translation + tone, no scope changes):
`docs/artifacts.md`, `docs/audio.md`, `docs/data.md` (no changes needed —
contained no Russian text), `docs/configuration.md`, `docs/model.md`,
`docs/training.md`, `src/aether/cli.py` (the handful of remaining Russian
argparse help/error strings, translated directly rather than via subagent).

Files translated **and** edited to remove/rework content describing the
now-deleted server/client/protocol:

- **`docs/architecture.md`** — removed Section 8 ("Client and server":
  browser client, WebSocket transport/event table, service separation)
  entirely; simplified the Section 3 architecture diagram to remove
  client/server/microphone/speaker nodes (it now runs directly from a raw
  audio input node to the streaming encoder, and from the streaming decoder
  to a raw audio output node); removed the "Stage B: live conversation"
  roadmap stage (renumbering the following stages); removed `server/` and
  `client/` entries from the proposed directory layout; removed scattered
  server/client mentions in the introduction and scope sections.
- **`docs/runtime.md`** — renamed to "aether — generation runtime"; deleted
  the entire "Protocol version 1" section (the WebSocket wire format);
  removed references to the `/health/live`/`/health/ready` HTTP endpoints and
  the "second session gets `BUSY`" behavior, reframing the lifecycle
  description around the generation engine's own readiness state instead of
  a network server's.
- **`docs/specification.md`** — removed "Python client" and "server with one
  active conversation" from the first-version scope list; removed "a live
  client" from the team-deliverables list; reworded the non-functional
  requirement about server readiness to describe the generation engine's own
  readiness instead.
- **`docs/development.md`** — removed the `serve`/`talk` CLI references, the
  "Server and WebSocket | aiohttp" dependency-table row, the `sounddevice`
  mention, the `server/`/`client/` directory-tree entries, the `audio`
  dependency-group description, and the corresponding example commands.
- **`docs/colab.md`** — reworded the sentence claiming a live WebSocket
  server/client was "not yet implemented" (it was deliberately removed, not
  pending) to state that live serving is out of scope for the current
  model-research phase.
- **`docs/validation.md`** — milestone M3 ("live full-duplex") kept, with an
  added "Status: on hold" note explaining that it cannot be attempted until a
  serving/client component is reintroduced.
- **`docs/README.md`** — removed the sentence noting that `serve`/`talk` were
  "not yet implemented."
- **`docs/tasks.md`** (the project's single task-status registry, ~516
  lines) — translated in full, including the operating-policy section
  (rewritten from imperative to descriptive policy statements). Task **A14**
  ("Server and Python client") is marked **Removed — out of scope** in the
  registry table, with its detailed section keeping the historical
  implementation record intact and a new closing note explaining the
  removal. Task **A20** ("Live full-duplex verification"), which depends on
  A14, is marked **On hold / blocked**, with a note that downstream tasks
  A24 and A27 inherit this blocked dependency (their own dependency lists
  were left unchanged to avoid silently rewriting the recorded dependency
  graph). A stale cross-reference to "A14 live server/client... remain open"
  in the 2026-09-18 notebook-readiness summary was subsequently corrected by
  the assistant directly to point to A14's new removed status, for
  consistency.
- **`notebooks/aether_colab.ipynb`** — translated all 10 Russian markdown
  cells to English with the same tone conversion (no server/client content
  to remove here, since the notebook never invoked `serve`/`talk`); all code
  cells were already in English. Verified via the notebook's own structural
  tests (`tests/unit/test_notebook.py`, `tests/unit/test_notebook_git.py`).

### Verification

After all edits: `grep -rP '[\x{0400}-\x{04FF}]'` across the repository
(excluding caches/venv) returns no matches — no Russian text remains. Full
local check suite passes: 202 tests, `ruff check`, `ruff format --check`, and
`mypy` all clean.

No commit was made as part of this work; changes remain in the working tree
pending review.
