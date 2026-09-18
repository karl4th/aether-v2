# aether — Unified Task Plan

Status date: 2026-09-18. This file is the single registry of tasks and their completion status. Other documents describe technical contracts but do not maintain a parallel status list.

## 1. Operating Policy and Constraints

**Autonomous continuation (user instruction):** after a task is completed, work continues without waiting for a "go ahead" command. Available tasks are executed sequentially, tested, fixed, and this registry is kept up to date. When blocked, independent parts are advanced; work stops only once a coherent result has been reached for the available work, or when a specific user action is required. Necessary data is requested in a timely manner. Paid resources and training are not launched automatically.

**Only development, documentation preparation, and limited tests are performed on the current machine. Training is prohibited here. All training is performed in a remote GPU runtime of paid Google Colab via a Jupyter notebook.** Connecting Colab to this machine's local runtime is not permitted, since that would move the computation back onto the local machine.

Locally permitted: schema checks, synthetic data, small forward passes, audio buffers, a mock server, and integration tests. Not run locally: optimizer steps, single-example overfitting, LoRA, full training, long hyperparameter searches, and full GPU evaluation. Even a short training smoke test is run in Colab.

Model, training, and evaluation code lives in the Python package; dependencies and processes are managed through uv. The notebook orchestrates runs, displays results, and connects to storage; it does not contain a second, independent implementation of the trainer.

Paid Colab does not guarantee a specific GPU or fixed session limits. Because of this, every run begins with a resource check and supports recovery after disconnection. Choosing a paid tier by itself does not prove that a large model will fit in memory. These constraints are confirmed by the [official Colab FAQ](https://research.google.com/colaboratory/faq.html).

This plan itself does not run computations, purchase resources, or create a working notebook: the corresponding work items are listed below.

## 2. How Completion Is Marked

- `[x]` — the result exists and the acceptance criteria are met.
- `[ ]` — the task is not complete. The table additionally notes "planned," "in progress," or a specific blocker.
- A task is closed after verification, not after the code is written.
- On closure, a date, paths to the results, the verification command or report, and any remaining limitations are added.
- Links in the "Result" field below denote future artifacts when a task is not yet marked done.

Order: documentation → scaffolding and contracts → small local tests → Colab preparation → weights and data verification → baseline evaluation → training pilot → recovery → adaptation → quality comparison. Independent tasks can be executed in parallel, but launching training is prohibited until its dependencies are satisfied.

## 3. Full Registry

| Done | ID | Task | Environment | Dependencies | Status |
|---|---|---|---|---|---|
| [x] | A01 | Architecture and technical contracts | Documents | — | Done |
| [x] | A02 | Python and uv specification | Documents | A01 | Done |
| [x] | A03 | Unified plan and environment separation | Documents | A01, A02 | Done |
| [ ] | A04 | Input decisions and resources | Locally / Colab preflight | A03 | Bundle/data selected; T4 measured, A100 profile still needs to be run |
| [x] | A05 | Python package and uv environment | Locally | A02 | Done |
| [x] | A06 | Configuration schemas and local-training lockout | Local tests | A05 | Done |
| [ ] | A07 | Artifact contract and loader | Local tests | A05, A06 | Pinned loader implemented; GPU verification pending |
| [x] | A08 | Audio buffers and the time grid | Local tests | A05, A06 | Done |
| [x] | A09 | Tokens, masks, and the delay schedule | Local tests | A06 | Done |
| [ ] | A10 | Streaming audio codec | Code locally, weights in Colab | A07, A08, A04 | Initial codec wired in; quality not verified |
| [ ] | A11 | Temporal transformer | Small local tests | A07, A09, A04 | Temporal backend wired in; model tests pending |
| [ ] | A12 | Text head and depth transformer | Small local tests | A11 | Text/depth backend wired in; model tests pending |
| [ ] | A13 | Generator and session isolation | Small local tests | A09, A10, A12 | Offline streaming loop and mocks ready; full run pending |
| [ ] | A14 | Server and Python client | Locally with a test engine | A08, A13 | **Removed — out of scope:** the live server, Python client, and wire protocol were deleted from the codebase; current work is model-only |
| [ ] | A15 | Evaluation tools and scenario set | Local tests | A06, A13 | Audio CE/perplexity implemented; conversational evaluation still open |
| [ ] | A16 | Colab orchestration notebook | Locally → Colab | A04, A05, A06 | Full notebook ready locally; GPU operations not yet tested |
| [ ] | A17 | Persistent storage and artifact delivery | Colab | A07, A16 | Checkpoint/Drive path implemented; remote recovery pending |
| [ ] | A18 | Dataset, annotation, and split creation | Small checks locally, processing in Colab | A04, A09, A17 | Reading corpus and splits implemented; preparation in Colab pending |
| [ ] | A19 | Baseline model: memory and offline quality | Colab | A13, A15, A16, A17 | Planned |
| [ ] | A20 | Live full-duplex verification | Local client + remote test | A14, A19 | **On hold / blocked:** depends on A14, whose server/client were removed; blocked pending a decision to reintroduce a live serving component |
| [ ] | A21 | Training loop implementation | Code and checks locally | A06, A12, A18 | Limited trainer/LoRA/resume implemented; extended scope still open |
| [ ] | A22 | Short training pilot | Colab only | A19, A21, A17 | Planned |
| [ ] | A23 | Training recovery proof | Colab only | A22 | Planned |
| [ ] | A24 | First targeted fine-tuning | Colab only | A18, A20, A22, A23 | Planned (inherits A20's blocked dependency — see the note under A20) |
| [ ] | A25 | Evaluating the adaptation and exporting the model | Colab | A24, A15 | Planned |
| [ ] | A26 | Automated checks and team instructions | Locally / CI without training | A05, A06, A09, A13, A16, A21 | Planned |
| [ ] | A27 | First-version acceptance | Reports and a repeat Colab run | A20, A25, A26 | Planned (inherits A20's blocked dependency — see the note under A20) |
| [ ] | A28 | Planning the next quality iteration | Analysis | A27 | Planned |

## 4. Completed Foundation

### A01 — Architecture and technical contracts

**Goal:** describe the system before implementation.

**What was done:** the audio codec, temporal model, text, and audio generation were separated out; initial tensor shapes, channel order, runtime, data format, and verification criteria were defined. Unconfirmed parameters are flagged separately.

**Result:** [architecture.md](architecture.md), [model.md](model.md), [runtime.md](runtime.md), [training.md](training.md), [validation.md](validation.md), [specification.md](specification.md).

**Acceptance:** the documents are present, internal links have been checked. Done 2026-09-16. This reflects documentation readiness, not proof of a trained model or compatibility with specific weights.

### A02 — Python and uv specification

**What was done:** Python/uv was selected; the `src/aether` package, dependency groups, CLI, checks, and reproducibility are described.

**Result:** [development.md](development.md).

**Acceptance:** the stack is agreed in the documentation. Done 2026-09-16. Creating the package, lock file, and environment was done separately in A05.

### A03 — Unified plan and environment separation

**What was done:** the documentation was broken down into tasks; for each one, actions, results, and verification are defined. Training and all tests that update parameters are assigned to the remote Colab runtime.

**Result:** this document and the updated environment rules in the documentation.

**Acceptance:** a full registry with dependencies and criteria exists; there are no markings for implementations that do not exist. Done 2026-09-16.

## 5. Implementation Plan

### A04 — Input decisions and resources

**Partial decision, 2026-09-16:** the initial codec remains in place for the first working result. The Manifestro codec is planned for later; its technical parameters are not yet known. The implementation assignment is kept separate from the versioned token contract. A swap will require checking frame rate, codebooks, vocabularies, dimensionality, and token semantics; it may require adapting/retraining the voice model and recreating tokenized data. This does not close A04: weights, data, and actual GPU resources remain open. Language, scenario, storage, and budget are specified below.

**Goal:** eliminate the unknowns that weights, memory, and training depend on.

**Actions:**
1. The language of the first result and the task class used to evaluate responses are recorded.
2. A compatible weights bundle, tokenizer, and configurations are selected; missing parameters are listed.
3. In Colab, the actually allocated GPU, VRAM, RAM, and disk are checked without starting training.
4. The available data volume, the experiment's step/time limit, and persistent storage are recorded.
5. A first adaptation mode is selected; LoRA is treated as a starting candidate, not a guarantee of fitting in memory.

**What it looks like:** `configs/experiment/base.json` and a filled-in resource table in the preflight report. Each unknown has an explicit status, with no invented VRAM figure or promise of a specific GPU.

**Done when:** the team can determine a viable configuration, or a specific reason why a run is not yet possible. Decisions are recorded before a large corpus is loaded.

**User decisions, 2026-09-16:** the first baseline's language is English; the scenario is a conversational chatbot for a demo; persistent storage is Google Drive; the first-stage budget is 500 units. The budget does not authorize automatically launching paid compute. Actual unit availability, consumption on the allocated GPU, time limits, and memory are checked in a future Colab preflight. A compatible weights/tokenizer bundle, a full architecture config, and data remain unknown. `configs/experiment/base.json` is not yet created as an executable config with invented weights.

### A05 — Python package and uv environment

**Actions:**
1. `pyproject.toml`, `.python-version`, `src/aether`, the CLI, and `.gitignore` are created.
2. The `dev`, `audio`, and `train` groups are defined; dependencies are resolved and `uv.lock` is saved.
3. Installation into a clean environment and the output of `uv run --locked aether --help` are checked.
4. A small test configuration that does not download large weights is created.

**What it looks like:** an installable package with the `inspect`, `infer`, `serve`, `talk`, `evaluate`, `train` commands; incomplete commands terminate with a clear message rather than simulating a working model.

**Done when:** import, CLI, formatting, and minimal tests pass through uv. No training is launched as part of this task.

**Done 2026-09-16:** `pyproject.toml`, `uv.lock`, `.python-version` (3.12.14), `.gitignore`, `src/aether`, `configs/model/tiny.json`, and CLI integration tests were created. The package and dev tools were installed into a fresh `.venv`. The installed command, help output, the refusal of incomplete operations without writing files, and the training lockout were checked: 6 tests pass. Ruff and mypy were checked. An independent read-only review by a subagent found no blocking defects in A05.

**Limitations:** this is scaffolding, not a voice model. The `train` group is deliberately empty until the Colab preflight. Working operations are not yet implemented; `train` is unconditionally blocked. `tiny.json` is a synthetic placeholder; the schema is refined in A06.

### A06 — Configurations and environment protection

**Actions:**
1. Validated schemas for the model, runtime, data, and training are introduced.
2. The `local_test` and `colab_train` profiles are separated; only the former is allowed by default.
3. Before optimizer creation, the profile, an explicit training authorization, and signs of a remote Colab runtime are checked. A single flag is not considered sufficient protection.
4. `train --validate-only` is implemented, checking the configuration without a trainer, optimizer, or training loop.
5. A mock test verifies that the local refusal happens before large weights are loaded and before the optimizer is accessed.

**What it looks like:** local `train` raises `TRAINING_ENVIRONMENT_REQUIRED`, pointing to the notebook. Configurations are versioned and reject unknown fields.

**Done when:** all trainer entry points use a single check; `--validate-only` works locally; the test performs zero weight updates. This check is a safeguard against mistakes, not a mechanism against a deliberately spoofed environment.

**Prepared during the A05 review:** strict types, a ban on unknown nested fields, finiteness of numbers, and width/heads/delays checks. The tiny model's BOS is determined by its vocabularies. Explicit tiny-size limits and a context reserve are needed. Validation-only does not import the backend. Until real signs of a managed Colab runtime are checked, the training path remains blocked. These requirements are verified in A06 (see below).

**Done 2026-09-16:** `src/aether/config/__init__.py`, `src/aether/training_gate.py`, the CLI validation-only path, and an updated `configs/model/tiny.json`. Pydantic strict/extra-forbid, version 1, finite numbers, sizes, heads, 17 channels, vocabulary-based BOS, explicit tiny limits, and context reserves. A versioned codec contract and the `src/aether/codec.py` interface were added. The JSON check does not import the backend, does not write files, and does not touch the network (audited in a separate process).

**Checks:** `uv run --locked pytest` — 37 tests together with A09; `uv run --locked ruff check .`, `uv run --locked ruff format --check .`, `uv run --locked mypy src/aether`. A mock test confirms the refusal happens before weights/optimizer under both the local and the claimed-Colab profile. No parameter updates were performed.

**Historical limitations as of 2026-09-16 (superseded by the implementation above):** there was no backend/trainer. The overall environment check deliberately always refuses: real signs of a managed remote Colab and a paid tier had not yet been checked. Validation-only confirms the schema, not the ability to train. Future trainer entry points are required to call this check before the backend. Details: [configuration.md](configuration.md).

### A07 — Artifacts and loader

**Actions:**
1. The manifest is specified: file names, SHA-256, schema version, dtype, vocabularies, provenance, and a versioned codec contract matching the configuration.
2. Checks for presence, integrity, and configuration match are implemented.
3. Strict parameter loading and a separate, versioned layout-conversion path are implemented where needed.
4. Small synthetic weights, a corrupted file, and an incompatible shape are checked locally.

**What it looks like:** `aether inspect` prints a table of checks and exits with a non-zero code on incompatibility.

**Done when:** an incomplete bundle is not silently accepted; large real weights are checked later in Colab.

**Partially implemented, 2026-09-16:** `src/aether/artifacts.py`, `aether inspect --config ... --manifest ...`, `tests/unit/test_artifacts.py`; the contract is described in [artifacts.md](artifacts.md). Version, provenance, sizes/SHA-256, paths, full configuration and codec match, and exact names/shapes/dtypes of synthetic parameters are checked. The loader calls the mock adapter only after all checks pass. Corruption, a missing file, an incompatible shape, extra parameters, NaN, an unsupported layout, and the CLI were checked.

**A07 remains open:** there is no full backend/state_dict or semantic verification of a real tokenizer. A full weights/architecture-parameter bundle has not been chosen (A04). A converter for unknown layouts is not implemented: they are rejected. Real weights are not loaded locally.

### A08 — Audio buffers

**Actions:**
1. Mono float32 PCM, value validation, and sample-rate conversion are implemented.
2. Packet accumulation is kept separate from 1920-sample frames.
3. Counters, bounded queues, reset, and rules for an incomplete tail are introduced.
4. Random packet boundaries and sample-count preservation are checked on a synthetic signal.

**What it looks like:** `audio/` produces identical frames regardless of network packet boundaries; tests show correct overflow handling and cleanup.

**Done when:** there is no loss, reordering, or unbounded memory growth; silence is distinguished from the absence of packets.

**Done 2026-09-16:** `src/aether/audio/__init__.py`, `tests/unit/test_audio.py`, [audio.md](audio.md). Little-endian float32 PCM, value validation, 1920-sample frames, a bounded queue, atomic failure on overflow, counters, reset, and a tail with explicit padding/discard. Offline sample-rate conversion uses windowed-sinc with anti-aliasing; a streaming resampler is not claimed. Random packet boundaries, absence of loss/reordering, buffer independence, silence versus absence, length, and the frequency response of synthetic signals were checked.

**A07/A08 checks:** a combined `uv run --locked pytest` run — 70 tests; `uv run --locked ruff check .`, `uv run --locked ruff format --check .`, `uv run --locked mypy src/aether`. All passed; the backend, real weights, a microphone, and the optimizer were not run.

### A09 — Tokens and temporal alignment

**Actions:**
1. The 17 channels, special values, masks, and delays are specified in a single schema.
2. The shift, initial positions, and reassembly of the physical frame are implemented.
3. Unique synthetic frame indices are used to check for off-by-one errors.
4. It is checked that sentinel values never reach the decoder and never enter the loss.

**What it looks like:** a shared schedule module for training and inference; a test table mapping input to output timing.

**Done when:** the inverse transform recovers all valid positions and behaves identically on a full sequence and on individual steps.

**Done 2026-09-16:** `src/aether/tokens.py`, `tests/unit/test_tokens.py`. A shared delay schedule, BOS, distinct missing/pending sentinels, lookup/loss masks, batched shifting, bounded state for the step-by-step shift and reassembly, and an explicit flush/reset. Sentinel values are rejected before the decoder; user channels do not enter the loss for the model's own voice prediction. Unique frame numbers prove the absence of off-by-one errors and the match between batch and stream modes. Verified in the combined run of 37 tests, Ruff, and mypy. This verification covers the discrete schedule, not a future tensor backend or model quality.

### A10 — Streaming codec

**Actions:**
1. The encoder, quantizer, and decoder are implemented per the fixed configuration.
2. Weights are kept separate from each session's encoder/decoder state.
3. Shapes and chunk boundaries are checked on small local fixtures.
4. In Colab, real weights are loaded and reconstruction of speech, silence, and noise is compared.

**What it looks like:** `AudioCodec.encode/decode/reset`, a comparison report between original and reconstructed audio, and audio samples from the approved test set.

**Done when:** the codec passes streaming checks and produces intelligible speech without regular seams between frames. Training the codec from scratch is out of scope for this task.

### A11 — Temporal transformer

**Actions:**
1. Stream embeddings, causal attention, positions, normalization, and the FFN are assembled per the full configuration.
2. A KV cache, a context limit, and reset are implemented.
3. Causality is checked by perturbing future inputs.
4. Batch mode and teacher-forced streaming are compared on a small, untrained network.

**What it looks like:** `TemporalModel` with documented shapes, state, and numerical-comparison tolerances.

**Done when:** future data does not change past outputs, the cache does not mix sessions, and the forward comparison passes.

### A12 — Text and audio codes

**Actions:**
1. A text output head and special-token masks are added.
2. Eight depth steps are implemented with the required projections and per-codebook weights.
3. Independent sampling settings for text and audio are introduced.
4. Resetting the depth cache between time steps and deterministic greedy mode are checked.

**What it looks like:** a single time step produces text and eight audio codes in the shifted space; the output logits have the shapes given in `model.md`.

**Done when:** shapes, conditioning order, token ranges, and handling of non-finite logits are all correct.

### A13 — Generator and sessions

**Actions:**
1. The codec, the schedule, temporal, depth, and reassembly are combined.
2. An explicit state object is created: caches, RNG, offsets, and queues.
3. Start, step, waiting for the first complete frame, and shutdown are implemented.
4. Two small test sessions are interleaved and compared against running them independently.

**What it looks like:** `DialogueEngine.step` returns `None` while delays are filling, then an aligned result; reconnecting starts from clean state.

**Done when:** the streaming loop passes a bounded synthetic test with no leaks and no cross-session context. Full quality is verified in A19.

### A14 — Server and client

**Actions:**
1. A versioned WebSocket protocol from `runtime.md` is implemented.
2. Ready/live states, a single active conversation, packet limits, and error messages are added.
3. A Python client is created with independent capture and playback queues; the callback does not call the model.
4. A mock engine, disconnect, BUSY, overload, and stop are checked locally.

**What it looks like:** a terminal client shows connection state and text; synthetic audio is transmitted bidirectionally. The microphone is used only in an explicitly launched test.

**Done when:** the network and audio loop works without the full model, handlers never hang, and all queues are bounded.

**Independently implemented part, 2026-09-16:** `src/aether/protocol.py` — the v1 binary packet, exact header/payload lengths, counter ranges, PCM validation, and separate per-direction/per-session sequence numbers. 10 tests verify the wire layout, loss/duplication/offset handling, corruption, and the maximum packet size. The WebSocket server, ready/live states, the client, and the mock engine were not implemented; A14 remained open.

**Closing note (this documentation revision):** the `server.py`, `client.py`, and `protocol.py` source files described above, their tests, the `serve`/`talk` CLI commands, and the `aiohttp`/`sounddevice` dependencies were removed from the codebase in this documentation pass. This was a deliberate decision to narrow the project's scope to the dialogue model/generator itself rather than a live serving product; the historical record above is kept for reference, but none of that code currently exists in the repository. **Status: Removed — out of scope.**

### A15 — Evaluation

**Actions:**
1. A scenario format with expected events and time annotations is created.
2. At least 100 scenarios are composed across the categories in `validation.md`.
3. Timing, queueing, content, and audio metrics are implemented; automated evaluation is supplemented with manual review.
4. A unified report is produced with configuration, seed, sample size, and errors.

**What it looks like:** JSON with measurements and a Markdown report; locally the format is checked against fixtures, real results are obtained in Colab.

**Done when:** the baseline and a candidate can be compared on the same inputs and multiple seeds, and unknown results are never substituted with zeros.

### A16 — Colab orchestration notebook

**Actions:**
1. `notebooks/aether_colab.ipynb` is created with no secrets and no saved heavy outputs.
2. It is split into cells: settings → preflight → fetching the exact code version → uv sync → storage → artifact verification → baseline → pilot → resume → train → evaluate → export.
3. Training sections are disabled by default via `RUN_TRAINING = False`; running all cells with this value set to False never triggers the optimizer.
4. The package's Python processes are run through `uv run --locked` in a separate `.venv`. The notebook kernel is not assumed to use this environment automatically.
5. The system kernel is used for widgets, storage authorization, and displaying results; numerical computation happens in the package's own process.
6. In a fresh Colab runtime, setup and validation-only are checked, and versions and output are recorded.

**What it looks like:** a notebook with clear sections, a GPU/VRAM table, a chosen run ID, and links to artifacts. Rerunning setup is safe and does not overwrite a completed run.

**Done when:** the notebook can be opened in paid Colab and sequentially prepare a reproducible environment. Locally, only the JSON/structure and the compilation of ordinary Python cells are checked; GPU cells are not executed here.

**Prepared locally, 2026-09-16:** `notebooks/aether_colab.ipynb`, `src/aether/preflight.py`, `scripts/build_colab_bundle.py`, [colab.md](colab.md). The notebook fetches an exact SHA-256-addressed archive of uncommitted sources, runs uv sync/validation-only, gathers resource information in a separate process, and saves a report into a new run on Drive. Environment observations are not presented as proof of remoteness or of a paid tier. RUN_TRAINING=False; the model and training sections are explicitly blocked. A16 remains open until verified in a fresh Colab; GPU cells were not executed locally.

### A17 — Persistent storage

**Actions:**
1. Google Drive is used as the proposed persistent storage to start; the path is set via a notebook parameter.
2. `datasets/`, `bundles/`, `runs/<run_id>/`, and `exports/` are kept separate.
3. Working files are copied to the Colab VM's disk and hashes are checked. During training, thousands of small examples are not read directly from mounted Drive.
4. A checkpoint is first fully written to the VM, then copied into a new directory in persistent storage; size and hash are checked, and only then is a completion marker published.
5. Resume selects only confirmed checkpoints; the last known-good one is not deleted until the new one is verified.

**What it looks like:** after the Colab runtime is deleted, the manifest, metrics, and completed checkpoints remain; a new runtime can restore them.

**Done when:** a copy failure never turns an incomplete file into an available checkpoint; quota/disk-full conditions lead to a clear stop. The `/content` disk is not considered persistent storage.

**Independently implemented part, 2026-09-16:** `src/aether/storage.py`, tests for publication, integrity checks, copy-failure handling, and selecting only confirmed checkpoints. A copy receives the COMPLETE marker only after files are verified; the previous one is not overwritten. A17 remains open: real Drive, runtime deletion, and repeated recovery have not been verified.

### A18 — Data and annotation

**Actions:**
1. An approved corpus is selected, and sources, language, and channel roles are recorded.
2. JSONL, audio, and time-boundary checks are implemented locally on small fixtures.
3. Data is split by conversation and by speaker before training.
4. In Colab, resampling, encoding, and alignment are performed; versions of the transformations are recorded.
5. Examples with pauses, overlaps, and transcription errors are manually reviewed.

**What it looks like:** versioned manifests for train/validation/test, a quality report, and tokenized shards in persistent storage.

**Done when:** there is no conversation leakage between splits, channels are synchronized, conflicting annotations are explicitly accounted for, and the cache is tied to the codec and tokenizer hashes.

### A19 — Baseline in Colab

**Actions:**
1. Preflight and strict loading of real weights are performed in the remote environment.
2. Memory is measured before and after warm-up and at the target context length.
3. Offline dialogues are run on fixed recordings without training.
4. Audio, text, latencies, configuration, and errors are saved.
5. On an out-of-memory condition, the failure is recorded; precision or context length are changed only via a separate configuration, followed by a re-check.

**What it looks like:** a `runs/<baseline_id>/` directory with a report on baseline quality and the actual GPU used.

**Done when:** speech is intelligible, measurements are reproducible, and the memory and duration limits are known. This is the reference point for all subsequent improvements.

### A20 — Live full-duplex

**Actions:**
1. A supported way to interactively test against the remote runtime is chosen: either a secure channel to the client, or a small browser-based audio adapter for the notebook.
2. `sounddevice` is not used in Colab to access the local machine's microphone, since these are different machines.
3. If the browser adapter is chosen, JavaScript is limited to audio capture and transport; the model and logic remain in Python.
4. Testing is done in headphones first, with echo cancellation tested separately afterward.
5. Barge-in, acknowledgment, correction, stop, and reconnect are checked; network latency is separated from compute latency.

**What it looks like:** an interactive test session in which the user's speech reaches the remote model while the response is being played back, with a report on the scenarios saved.

**Done when:** genuinely simultaneous listening and speaking is confirmed. Two-channel offline replay does not substitute for live acceptance. If the environment does not allow the necessary channel, the task remains open; Colab is not turned into a promised, persistent production server.

**Status update (this documentation revision): on hold / blocked.** This task depends on A14, whose live server, Python client, and wire protocol have been deleted from the codebase (see A14). There is currently no live-serving component to test full-duplex behavior against. This task is blocked pending a future decision to reintroduce a live serving component; it is not being actively worked and cannot proceed as originally scoped until that decision is made.

**Downstream note:** A24 and A27 both list A20 as a dependency. That dependency chain is now stalled for the same reason: neither task can be considered unblocked until A20's blocker (A14) is resolved. Their entries in Section 3 and below are left unchanged to avoid silently rewriting the recorded dependency graph, but a future reader should treat A24 and A27 as inheriting A20's blocked status until this note is updated.

### A21 — Training code

**Actions:**
1. The dataset/collator, forward pass, masks, and separate loss components are implemented.
2. Selection of trainable parameters and a LoRA configuration are added.
3. Accumulation, clipping, a scheduler, checkpointing, and a manifest are implemented.
4. The A06 safeguard is applied to every path that creates or launches the trainer.
5. Numerical functions are checked locally on fixed tensors, along with serialization and mock-optimizer calls; a real `optimizer.step()` is never executed.

**What it looks like:** a single trainer that the notebook invokes through the CLI, with no duplicated training code inside cells.

**Done when:** the contracts and validation-only pass locally, while an actual training run awaits A22 in Colab. The code is ready to be tried, but is not yet marked as verified training.

### A22 — Short pilot

**Actions:**
1. In Colab, a new run is created with a hard, small `max_steps`, a time limit, and frequent checkpointing.
2. A single batch, backward pass, finiteness of loss/gradients, and that only authorized parameters change are checked.
3. A short overfit on a small set is run, followed by free generation.
4. Peak VRAM and step time are measured, and viable batch size/length/accumulation are tuned.
5. The run stops at the limit regardless of the loss value.

**What it looks like:** a small, completed run with a loss curve, a list of trainable parameters, a checkpoint, and a memory profile.

**Done when:** the training loop actually works and fits on the allocated GPU. The pilot is not considered proof of usefulness on new data.

### A23 — Recovery after disconnection

**Actions:**
1. A full checkpoint of the pilot is saved: parameters, optimizer, scheduler, RNG, scaler if present, and the data position.
2. The process is terminated and a clean Colab session is opened.
3. The environment is restored from the commit/lock, and the checkpoint is loaded from persistent storage.
4. The next step is compared against an uninterrupted control run within a set tolerance.
5. Failure handling is checked for an incomplete file and an incompatible configuration.

**What it looks like:** a `resume-check` report with the recovered step number, hashes, and the comparison result.

**Done when:** recovery does not restart training from scratch, does not lose optimizer state, and does not use an unconfirmed copy. Both training runs are performed only in Colab.

### A24 — First targeted fine-tuning

**Actions:**
1. One category of weak responses from the baseline is selected.
2. Data, parameters, the target metric, and acceptable regressions are recorded before the run starts.
3. GPU, the step/time budget, free space, and checkpointing are checked.
4. Training is launched from the notebook, saving checkpoints on both time and step intervals.
5. Validation and limited free generation are run; the experiment is stopped on a numerical error or when the limit is reached.

**What it looks like:** a new run with an immutable config, metrics, intermediate results, and a completed checkpoint.

**Done when:** the run is either completed or correctly stopped, and is fully documented. Completion by itself does not imply improvement — that is decided in A25.

### A25 — Comparison and export

**Actions:**
1. The baseline and the candidate are evaluated on the same held-out set and seed.
2. Usefulness, voice, barge-in behavior, latency, and robustness are compared.
3. Regression cases are separately listened to.
4. For an accepted candidate, an inference bundle is created with hashes and configuration.
5. Loading the exported bundle is checked in a fresh Colab process.

**What it looks like:** a comparison table, an accept/reject decision, and, only upon acceptance, a model export. A rejected run is kept as a research result.

**Done when:** the decision is based on criteria defined in advance; the export does not depend on the state of the old notebook.

### A26 — Checks and team process

**Actions:**
1. Ruff, mypy, pytest, and notebook schema checks are set up in CI.
2. Training, large-weight loading, and GPU tests are excluded from the regular local/CI suite.
3. A check for the local-training safeguard is added.
4. Lock updates, versioning/release, running the notebook, and reproducing a run are documented.
5. The notebook is checked for cleanliness: no access tokens, no private outputs, and no accidental training auto-runs.

**What it looks like:** a short mandatory check suite, reproduction instructions, and separately marked Colab-only checks.

**Done when:** another team member can set up the environment and find the needed experiment without verbal instructions.

### A27 — First-version acceptance

**Actions:**
1. The scenario is run through from a clean checkout and a new Colab runtime.
2. The bundle, offline inference, live dialogue, saving, and reloading are checked.
3. Functional requirements F01–F10 are checked off against links to their verifications.
4. A list of known limitations is compiled: language, GPU, duration, and the measured cost of the run.

**What it looks like:** a first-version report with a requirements table, links to artifacts, and an honest status for each criterion.

**Done when:** the mandatory criteria are met; open limitations are not hidden behind a blanket "done" status. The baseline voice result and the improvement result are assessed separately.

### A28 — Next iteration

**Actions:**
1. Remaining errors are grouped by audio understanding, facts, logic, context, and conversational dynamics.
2. The most significant direction is chosen: data, language, the text backbone, context, or an external planner.
3. Feasibility of the experiment in the available Colab runtime is assessed before allocating a budget.
4. New tasks are added to this same registry with dependencies and a measurable result.

**What it looks like:** a concrete next hypothesis and a set of tasks, instead of a vague intention to "make it smarter."

**Done when:** there is a testable goal for the next iteration and a saved comparison point. Training the entire foundation model from scratch is not automatically assumed to be feasible within paid Colab.

## 6. Layout of the Finished Notebook

```text
aether_colab.ipynb
  01  Parameters: commit, run ID, config, paths, RUN_TRAINING=False
  02  Checking the remote runtime, GPU, memory, and disk
  03  Fetching the code and preparing the uv environment
  04  Connecting persistent storage
  05  Verifying weights, tokenizer, and data
  06  Baseline and profiling without training
  07  Short training pilot — only when explicitly enabled
  08  Selecting and verifying a checkpoint for resume
  09  Limited training via the Python CLI
  10  Evaluation, tables, and listening to samples
  11  Export and verification of the copy in persistent storage
  12  Run summary and releasing the runtime
```

The notebook is expected to show the chosen configuration before any costly step, the reason for skipping disabled sections, and the location of the last confirmed checkpoint. Rerunning cells must not silently duplicate a run or overwrite its results.

## 7. Completion Log

| Date | ID | Confirmation | Limitation |
|---|---|---|---|
| 2026-09-16 | A01 | Architecture and contract documents in docs | No implementation or verified weights yet |
| 2026-09-16 | A02 | development.md and agreed Python/uv | Environment not yet created |
| 2026-09-16 | A03 | tasks.md, the registry, and the local/Colab rules | Notebook and training not yet run |
| 2026-09-16 | A05 | Python 3.12.14, uv.lock, CLI; 6 tests, Ruff, mypy | Backend and configuration validation not yet implemented |

The current next step is given in the full-notebook readiness section below. Historical entries preserve their date and are not a current description of the CLI.

**Future codec gate (user decision, 2026-09-16):** aether is first fully implemented and verified with the initial codec. The switch to the Manifestro codec is deferred until the second stage of its development, even if the aether baseline succeeds. For now, dependencies, adapters, and migration logic for the future codec are not added. The shared Protocol fixes the interface of the initial codec, not the integration of the future one.

## Full Notebook Readiness — 2026-09-18

The following sequence has been implemented: Git fetch/detached checkout → uv/model dependencies → runtime and Drive resolution → data → real weights → inference with listening → baseline CE → a 5-step pilot → a new resume process up to 20 steps → repeat CE/inference → export. The utility preflight is not presented as the final result. The training path only executes backward/optimizer.step when RUN_TRAINING=True is set manually and remotely.

**Artifacts:** backend.py, dataset.py, trainer.py, remote.py, the strict config/adaptation.py and config/operations.py schemas, the CLI, and configurations. Dependencies are pinned in uv.lock separately from dev. Legal attribution is in THIRD_PARTY_NOTICES.md, which is included in the checkpoint. The initial codec is kept; the future one is not wired in.

**Preflight obtained:** Tesla T4 15360 MiB, driver 580.82.07, RAM 53467192 KiB, disk 201975767040 bytes, Python 3.12.14/Linux. The absence of google.colab under uv is expected. BF16 inference requires 22 GiB of free VRAM; adaptation requires 38 GiB (A100 40 GB+). The 500 units are not converted into an invented number of hours; consumption is tracked in the Colab UI.

**Pilot scope:** a legally sourced English reading corpus, 16 train/4 evaluation recordings of 2–5 seconds, with no speaker overlap. Two rank-8 LoRA matrices on the last temporal FFN are trained; the base model and codec are frozen. Accumulation, clipping, a scheduler, a step/time limit, and an optimizer/RNG checkpoint are all in place. Text without timestamps is excluded from the loss. This is audio-generation adaptation, not dialogue instruction tuning.

**Checks:** 184 tests (`uv run --locked pytest`), Ruff check/format, and mypy; CLI dispatch, schemas, model mocks, masks, SHA256/identity, resume, notebook compilation, and a Git checkout against a small repository. Local checks exclude optimizer steps, full weights, and a GPU forward pass.

**Open:** A07/A10–A13 require the full model; A16/A17 require remote execution; A19/A22/A23 require real baseline/pilot/resume reports. A14 (live server/client) has since been removed from scope (see A14), the full A15 conversational scenarios remain open, and the A18 dialogue data remain open. This does not block the prepared offline notebook with limited adaptation, but it does not substitute for project acceptance. Readiness for a manual run means the code is ready, not that training has already succeeded. Instructions: [colab.md](colab.md). The agent did not launch any paid resources.
