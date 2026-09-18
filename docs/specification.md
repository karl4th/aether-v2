# aether — project specification

## 1. Purpose

aether is a streaming speech-to-speech system for live dialogue. The user speaks freely, and the system continuously listens and can speak simultaneously with them. The first milestone is reproducible voice interaction; the next stage is improving the quality of understanding and responses.

The development language is Python. The interpreter, environment, dependencies, and execution are managed via uv. PyTorch is used for the neural networks. A new, separate runtime in another language is out of scope for the first version.

The current machine is intended only for development and limited tests without training. All training runs, including pilots and optimizer checks, are performed via a notebook on a remote GPU runtime of paid Google Colab. Full model checks are likewise carried out on Colab. Tasks, dependencies, and actual execution are tracked in [tasks.md](tasks.md).

## 2. Functional requirements

| ID | Requirement | Verifiable outcome |
|---|---|---|
| F01 | Accept a mono PCM stream | Arbitrary packet boundaries do not change the sequence of model frames |
| F02 | Generate speech until the session ends | Output is available frame by frame, without accumulating a whole response |
| F03 | Listen while responding | Input frames continue to be fed into the model |
| F04 | Model pauses and overlaps | Silence and simultaneous speech preserve the timeline |
| F05 | Emit the text of its own speech | Special tokens are hidden, text order is preserved |
| F06 | End and start a conversation | A new conversation receives a clean state |
| F07 | Handle explicit cancellation | The output queue is cleared, the session ends |
| F08 | Replay recorded input | The same configuration yields comparable results |
| F09 | Save a run manifest | The result is linked to weights, code, environment, and seed |
| F10 | Support fine-tuning | The base model and the adapted model can be compared on an independent set |

F07 means cancellation of the entire session in the first version. Preserving the conversation after a forced removal of already generated speech is a separate experiment. Ordinary voice interruption (barge-in) is handled by the model without an administrative reset.

## 3. Non-functional requirements

1. A session does not use another session's mutable state.
2. Queues and session duration are bounded by configuration.
3. The generation engine does not report a ready state until the model has been loaded, validated, and warmed up.
4. CPU is acceptable for small test configurations; real-time operation of the full model on CPU is not guaranteed.
5. Linux with CUDA is the primary profile for performance research. macOS is the development profile; acceleration on it requires separate confirmation of operator compatibility.
6. Microphone recordings are not saved by default. Diagnostic audio collection is enabled explicitly.
7. Shape errors, token range errors, and configuration errors are detected before a live session starts.
8. Weight loading is strict: missing and unexpected parameters are not ignored without a versioned conversion.
9. A change in numerical precision is accompanied by a quality and latency check.
10. Mandatory checks are run via uv from a pinned environment.

## 4. First version

The scope includes the model, the streaming codec, the generator, offline evaluation, and preparation of the first fine-tuning experiment.

An external text LLM is not a required part of the voice loop. The full pipeline of recognition, text generation, and speech synthesis can be used as a separate comparison experiment, but it does not replace validation of the joint two-stream model.

Out of scope for the first acceptance: multi-user batching, telephony, a mobile client, action execution, long-term memory, arbitrary voice cloning, guaranteed multilinguality, and training the entire system from scratch.

## 5. Team deliverables

- An installable Python package `aether` and its CLI commands.
- Versioned configuration schemas for the model and the session.
- A validated set of weights, tokenizer, and compatibility manifest, kept outside Git.
- A working offline pipeline.
- A test suite for temporal alignment and isolation.
- A report on latency, memory, duration, and dialogue quality.
- Data and a protocol for the first controlled fine-tuning experiment.

## 6. What counts as equivalent behavior

A similar timbre or a brief demonstration is not sufficient. Comparison is carried out under identical scenarios, hardware, audio format, and network conditions. Evaluation covers not only initial responses but also corrections, interruptions, pauses, long context, and reconnections.

A claim of architecture reproduction requires matching computational contracts. A claim of quality reproduction requires trained weights and measurements. These are two distinct readiness criteria.

## 7. Decisions before computational experiments

| Decision | What it blocks |
|---|---|
| Weight set versus training from scratch | Validation of intelligible generation |
| GPU and available memory | Performance profile and training scale |
| First evaluation language | Data selection and substantive criteria |
| Available two-channel recordings | Training for natural interruptions |
| Budget and experiment duration | Choice between full adaptation or LoRA |

While these decisions remain open, work can proceed on the protocol, small test models, contracts, evaluation, and environment management. This work does not require presenting a random model as a finished conversational partner.
