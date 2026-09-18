# aether — data and training

## Required environment

All training under this document is performed only in the remote GPU runtime of paid Google Colab via `notebooks/aether_colab.ipynb`. This includes the pilot run, backward/optimizer checks, single-example overfitting, and resume verification. The current machine is used for development and limited tests without parameter updates. Exact tasks and statuses are tracked in [tasks.md](tasks.md).

GPU and available memory are checked anew before each run. The notebook invokes the Python package via uv. Data for active work is placed on the Colab VM disk; completed checkpoints and reports are copied to and verified in persistent storage. The runtime's temporary disk is not a backup. Training from scratch remains a research program whose feasibility on available resources has not been confirmed.

## 1. Two modes of work

**Adaptation of compatible pretrained components** is the first practical path. Quality is first established without changes, then limited fine-tuning and re-evaluation are performed.

**Training foundational components from scratch** is a separate research program. It requires a text corpus, an audio corpus, two-channel dialogues, a teacher model for semantic features, and a sufficient compute budget. Timeline and cost cannot be estimated from a single architecture table alone.

The team's initial task is adaptation and reproducibility. Newly trained own weights are not considered ready until intelligible speech is produced and the dialogue evaluation is passed.

## 2. Model artifact

Each bundle contains:

```text
bundle/
  manifest.json
  model_config.json
  codec_config.json
  tokenizer.model
  dialogue.safetensors
  codec.safetensors
  generation_config.json
```

The manifest specifies the schema version, model identifier, SHA-256 of the files, evaluation language, dtype, and provenance of each artifact. If a component carries mandatory usage or attribution terms, the bundle must comply with them. A custom model name does not substitute for this metadata.

Full loading is verified before an adapter is added. Name and layout conversion is performed by a separate, tested script; `strict=False` is not a way to prove compatibility.

## 3. Dataset

The core unit is a synchronized two-channel dialogue. Channel 0 is the target aether speech; channel 1 is the user's speech. Both sides share the same time axis and sample rate. A mono recording with multiple voices is not automatically considered a suitable two-channel example.

JSONL manifest, one record per conversation:

```json
{"id":"dialogue-0001","audio":"audio/dialogue-0001.wav","annotation":"annotations/dialogue-0001.json","sample_rate":24000,"channels":2,"duration_samples":240000,"language":"en","split":"train","speaker_ids":["s01","s02"],"source_id":"collection-a"}
```

Annotation example:

```json
{"schema_version":1,"segments":[{"speaker":1,"start_sample":12000,"end_sample":36000,"text":"Hello","words":[{"text":"Hello","start_sample":12000,"end_sample":36000}]}]}
```

Boundaries are half-open: `[start_sample, end_sample)`. Overlap between different voices is allowed. Invalid negative boundaries, an end past the recording duration, an unknown speaker, and channel mismatch are rejected. Participant identifiers are pseudonymized.

## 4. Preparation

1. Usability and provenance of the recording are verified.
2. Decoding, channels, duration, NaNs, clipping, and empty files are checked.
3. Both channels are resampled to a common rate together, preserving synchrony.
4. A transcript with time boundaries is obtained, and a sample is manually reviewed.
5. Deduplication and splitting by speaker and conversation are performed.
6. Audio codes are obtained with the frozen codec.
7. Text subtokens are mapped to the time grid and masks are formed.
8. The hash of the source recording, and the codec, tokenizer, and transform versions, are saved.

Cached codes are invalidated when the codec or its parameters change. Trimming of a recording is done identically for both channels. Chunk boundaries must not lose overlaps or turn pauses into time gaps.

## 5. Text alignment

When compatible with pretrained weights, the alignment scheme matching those weights is used. It cannot be replaced with an approximate word placement without verification.

For in-house training, a separate versioned algorithm is proposed: tokenize the word, distribute its subtokens across the allowed frames within the time interval, and fill the remaining positions with padding. Conflicts, where there are more subtokens than available frames, are flagged for special handling rather than silently overwritten.

Such an algorithm is a research decision, not a proven equivalence to any pretrained model. The conflict rate, time shift, and impact on intelligibility are measured. A saved alignment configuration is mandatory for every experiment.

## 6. Training forward pass

Input: `[B,17,T]`. The delays from `model.md` are applied, followed by the initial step. Temporal receives past shifted values; the text head predicts the next text token.

Depth is trained with teacher forcing: it receives the target text and the previous target audio codes of the current shifted step. The text and audio heads are aligned back with the targets; invalid initial and trailing positions are excluded.

In the base conditional setting, loss is computed on aether's text and its eight audio books. User codes serve as conditioning. Additional prediction of the user's speech is a possible separate objective with its own head configuration and loss weights.

```text
L = λtext × mean_valid(CEtext)
  + Σq λq × mean_valid(CEaudio,q)
```

Normalization is performed by the number of valid targets for each component, not by the padded batch length. When there are no valid targets, the component is skipped correctly, without division by zero.

Teacher forcing can produce low loss and poor live dialogue. Therefore, every saved candidate undergoes free-running generation on held-out scenarios.

## 7. Adaptation stages

### T0. Unmodified model

A report on speech, content, latency, and memory is obtained. Reference audio and configuration are saved. Without this, subsequent improvement cannot be separated from changes to the test rig.

### T1. Training loop verification

On a small set, it is confirmed that loss is finite, gradients reach the selected parameters, frozen weights do not change, the checkpoint restores correctly, and a single example can be overfit. This is a technical check, not an evaluation of generalization.

### T2. Limited fine-tuning

The codec is frozen. Either adapters or a limited parameter set are selected. Target modules, rank, scaling, dropout, optimizer, learning rate, scheduler, clipping, batch size, accumulation, and audio segment length are fixed. Numbers are chosen after a memory profile and a small pilot run.

### T3. Usefulness improvement

Data is added targeting a specific identified error: corrections, instruction following, conciseness, facts, context retention. Language, voice, architecture, and dialogue policy are not changed simultaneously — otherwise the cause of the result cannot be established.

### T4. New language and architectural changes

Tokenizer coverage, codec quality for the language, and the volume of aligned dialogues are checked separately. A stronger text backbone requires new alignment with the audio network and is not a routine replacement of the weights path.

## 8. Training the codec from scratch

If this direction is chosen, an independent audio-reconstruction experiment is conducted first. The semantic objective, quantization, adversarial loss, and feature matching are considered separately. Training discriminators are not included in inference.

Codebook collapse, code usage frequencies, intelligibility, timbre, latency, and streaming parity are checked. The codec is fixed before mass caching of tokens for the dialogue model. Jointly changing the codec and the dialogue model requires regenerating the data and re-running the baseline evaluation.

## 9. Checkpoint and recovery

A training checkpoint contains the model or adapter parameters, optimizer, scheduler, scaler (if used), RNG state for all devices, global step, data position, and configuration. An inference export contains only the artifacts necessary for inference and a manifest.

Saving is atomic: written to a temporary location, verified, then the completed checkpoint is published. A write failure must not destroy the last valid checkpoint. Recovery is verified by comparing the next step against a continuous run within a specified tolerance.

## 10. Result selection

Low validation loss alone is not a release criterion. A candidate is accepted if it improves a pre-selected content metric and does not violate constraints on intelligibility, interruption handling, latency, and robustness. The report includes failure scenarios and differences from the baseline model.
