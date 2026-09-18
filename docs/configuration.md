# aether — version 1 configurations

`uv run --locked aether train --config configs/model/tiny.json --validate-only`
reads the JSON and validates the schema without a backend, weight loading, file writes, or network access.
Code 0 means the schema is correct; 2 indicates a file/JSON/contract error.
A regular `train` invocation returns 4 (`TRAINING_ENVIRONMENT_REQUIRED`) before the file is read.

All nested schemas carry `schema_version=1` (it can be omitted, default 1),
forbid unknown fields, and forbid coercion of strings/booleans to integers.
Numbers are finite; sizes are positive; dimension is a multiple of heads.
Unknown versions and duplicate JSON keys are rejected.

The default `profile` is `local_test`: CPU, float32, a synthetic codec, and training
disabled. Limits: temporal/depth dimension ≤64, layers ≤2, context ≤64,
audio vocabulary ≤64, text vocabulary ≤128, batch ≤2, queue ≤8 frames.
These are schema constraints, not full memory control for a future backend.

Runtime: 24000 Hz, 1920-sample frame, positive queue/timeout limits,
session frame count, and separate initialization/shutdown reserves (minimum 1 each).
Session length and data sequence length together with reserves do not exceed context.
Data configuration sets batch, length, and seed; training configuration sets resolution, max_steps, and learning_rate.
They do not yet describe the dataset manifest or the full trainer.

The model has exactly 17 channels with delays `[0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1]`.
BOS equals the size of the corresponding vocabulary and is allowed only on input.
Text padding=3 is excluded from loss; end-of-padding=0 remains a trainable marker.
Sentinels missing=-1 and pending=-2 are not used for embedding or loss.

The working remote path uses `configs/model/remote.json` and
`configs/training/colab_lora.json`: separate strict, backend-free schemas for inference
and for limited adaptation. `train --validate-only` checks them without a trainer.

Real operations invoke `require_remote_runtime` before the backend. This requires Linux,
two Colab environment observations, an NVIDIA driver, and a short-lived permission
with the current boot ID, a live kernel PID, source revision, and explicit
user confirmation. Training additionally requires a separate allow_training flag. The
google.colab package inside the isolated .venv is not required. A paid tier
cannot be proven programmatically; the safeguard prevents accidental launches, not
deliberate environment spoofing.

The codec implementation is specified separately from the token contract version. For local
fixtures it is `synthetic`; for the first result it is `initial` (the initial codec).
The `unverified` status does not mean compatibility with real weights has been proven.
The Manifestro codec has not yet been added as a selectable implementation: its parameters are unknown.
A future bundle must verify this contract together with the weights and tokenizer.

The full notebook is ready for manual remote testing. It performs real
inference, a limited pilot run, and resume when RUN_TRAINING=True. Instructions are in
[colab.md](colab.md). A successful GPU run has not yet been claimed.

**Future codec gate (user decision, 2026-09-16):** aether is to be fully
implemented and verified with the initial codec first. The transition to the Manifestro codec
is deferred to the second stage of its development, even given a successful aether baseline.
Dependencies, adapters, and migration for the future codec are not being added at this time.
The shared Protocol fixes the interface of the initial codec, not integration of the future one.
