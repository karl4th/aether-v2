# aether — verification and implementation plan

Statuses and task order are tracked only in [tasks.md](tasks.md). The stages below describe technical acceptance, not a separate execution registry.

Only limited tests without training are run locally: schemas, synthetic tensors, small forward passes, mock optimizer, and protocol. Real optimizer steps, training smoke tests, and training-resumption checks are run only in a remote paid Colab. Full weights, GPU profiling, and model quality are also verified in Colab. "Local test bench" in the latency thresholds means a controlled test alongside the compute process, not permission to run the large model or training on the current machine; network conditions are accounted for separately for a live remote conversation.

## 1. Principle

Tests distinguish program correctness, weight compatibility, and trained-model quality. A successful tensor-shape test does not prove speech intelligibility; a pleasant voice does not prove that the answer is correct.

## 2. Tests without large weights

Small network sizes and synthetic data are used.

| Test | What it detects |
|---|---|
| Splitting PCM into random packets | Sample loss and incorrect frame accumulation |
| Delay → undelay on unique indices | Swapped channels, off-by-one errors, and tail errors |
| Modifying future input | Violation of causality in past outputs |
| Teacher-forced streaming vs. batch forward | Divergence of caches and masks |
| Reset and new session | State carryover |
| Two interleaved sessions | State mixing between users |
| Protocol round trip | Endianness, length, and sequence errors |
| Overflow and disconnect | Unbounded queues, leaks, and stuck workers |
| Loss masks | Training on padding and invalid positions |

The streaming vs. batch comparison is performed under identical initial conditions and without random sampling. Tolerances are determined by dtype. The causality check excludes changes that legitimately affect the output due to declared alignment.

## 3. Tests with weights

1. Strict loading of all parameters.
2. Verification of tokenizer and special-ID consistency.
3. Audio encode/decode on speech, silence, and noise.
4. Reference logits on a fixed input, when a reference artifact is available.
5. Comparison of frame-by-frame and continuous codec execution, accounting for padding and tail.
6. Intelligible generation on offline scenarios.
7. Verification of free generation after any fine-tuning.

Bit-exact matching is not required across different GPUs and kernel implementations. A significant discrepancy must be explained, not hidden by widening the tolerance to an arbitrary value.

## 4. Timing measurement

Four quantities are distinguished:

- **Compute step:** encode + temporal + depth + decode of a single step.
- **Pipeline latency:** the path of a known input frame through transport, buffers, and processing until the associated output frame becomes available. The association is established by instrumenting indices, not by guessing from speech.
- **Response gap:** from the marked end of the user's utterance to the start of a meaningful response.
- **Yield latency:** from the start of an explicit interruption to the cessation of the audible response.

Latency is not measured by subtracting unsynchronized clocks between two machines. On the local test bench, a shared monotonic clock or an external recording of input and output is used. GPU durations are measured with correct GPU timers.

For each test bench, the following are recorded at minimum: GPU, CPU, RAM, driver, versions, dtype, number of sessions, network mode, packet size, and run duration. p50/p95/p99 and the number of observations are reported, with cold start reported separately from warmed-up operation.

Initial engineering targets: compute p95 < 80 ms; pipeline p50 ≤ 300 ms and p95 ≤ 500 ms on a consistent local test bench. These are project targets, not yet confirmed by measurement. For response gap and yield latency, distributions are collected first; acceptance thresholds are established before candidates are compared.

## 5. Dialogue test set

At least 100 scenarios, divided into categories:

| Category | Expected behavior |
|---|---|
| Short question | Relevant, complete answer |
| Pause within a question | No systematic premature response |
| Correction of a number or name | Use of the corrected value |
| Explicit interruption | Yielding the floor and accounting for the new utterance |
| Short "uh-huh" | No mandatory cutoff of the response |
| Two speakers at once | Input preservation and generation stability |
| Silence | No uncontrolled infinite monologue |
| Noise | Measured degradation with no failures |
| Instruction given at the start of the conversation | Verification of retention within the context window |
| Disconnect and reconnect | New, clean state |

For each scenario, duration, expected features, event annotation, and set version are stored. Stochastic trials are repeated on at least three seeds. The same scenarios and seeds are used when comparing variants.

## 6. Quality

Content metrics: correctness of fact or decision, adherence to explicit constraints, handling of corrections, context retention. Audio metrics: intelligibility, artifacts, timbre, pace, and naturalness of pauses. Dialogue metrics: premature responses, successful interruptions, erroneous yields, and hangs.

Automatic transcription can be used to count errors, but it is itself error-prone. A portion of recordings is reviewed by humans; evaluation does not rely on a single automated judge alone. Categories with a small number of examples are not combined into a convincing overall percentage without stating the sample size.

## 7. Milestones and transition criteria

### M0 — Python scaffold

`pyproject.toml`, `uv.lock`, the package and CLI, validatable configurations, and CI are in place. `uv sync --locked` and CPU checks pass on a clean environment. The model bundle and target hardware are either fixed or explicitly block M2.

### M1 — computational contracts

A small model passes shape, causality, cache, delay/undelay, reset, and loss-mask checks. The protocol passes packets of arbitrary length and malformed messages.

### M2 — offline voice

The weight bundle loads strictly. The codec and generator produce verifiable audio output. A run manifest, memory profile, and baseline-scenario report exist. The absence of intelligible speech blocks progression to a product demonstration.

### M3 — live full-duplex

**Status: on hold.** The live WebSocket server and the local Python microphone/speaker client (`server.py`, `client.py`, `protocol.py`, and the `serve`/`talk` CLI commands) have been removed from the codebase in this revision so the project can focus on the dialogue model and generator rather than on a live serving product. M3 cannot be attempted until a serving/client component is reintroduced; the criteria below describe the target validation for that future component, not current work.

The (Python) client transmits and plays back audio simultaneously. A warmed-up session does not accumulate a queue. Reconnection clears state. Pauses, interruptions, and the context limit are verified. Measurements are preserved.

### M4 — controlled fine-tuning

Training resumes from a checkpoint, preserves frozen parameters, and exports the result correctly. Meaningful improvement is confirmed on independent data, and voice regressions are checked.

### M5 — further development

A new language, a stronger text backbone, long sessions, and scaling each receive their own configurations and criteria. Performance optimization is not combined with a change in training data within a single, non-comparable experiment.

## 8. Report format

Each report contains the objective, configuration, environment, artifact hashes, scenario set, sample sizes, summary metrics, links to permitted diagnostic recordings, errors, and the decision for the next stage. On failure to meet a target threshold, the actual result is preserved; the status is not replaced with a "mostly works" wording.
