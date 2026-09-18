# aether — technical documentation

Status: a remote backend with real weights, an offline audio demo, and limited adaptation have been implemented. `notebooks/aether_colab.ipynb` contains Git checkout, installation, data, inference with listening, evaluation, pilot/resume, and export. This is readiness for manual testing in Colab, not the result of completed training. Locally, the model and optimizer have not been run.

Work begins with the [unified task plan](tasks.md): it contains the full registry, completed items, dependencies, and acceptance criteria. The current machine is used only for development and limited tests. All training, including short training checks, is performed in the remote runtime of paid Google Colab via notebook; the local runtime is not used.

## How to read

| Document | Contents |
|---|---|
| [tasks.md](tasks.md) | Unified plan, statuses, step-by-step tasks, and work via Colab |
| [specification.md](specification.md) | Requirements, boundaries, mandatory decisions, and result criteria |
| [architecture.md](architecture.md) | Overall architecture and component explanation |
| [model.md](model.md) | Mathematics, model configuration, tensors, codec, and temporal alignment |
| [runtime.md](runtime.md) | Execution order, states, protocol, buffers, and errors |
| [training.md](training.md) | Data, training, loss functions, and experiment saving |
| [development.md](development.md) | Python package structure, uv, dependencies, and workflow |
| [validation.md](validation.md) | Implementation checks, measurements, and acceptance stages |
| [data.md](data.md) | Brief project identification |

## Requirement statuses

- **Mandatory** — a condition for implementation and acceptance.
- **Initial configuration** — the chosen starting point; changing it requires a configuration version and re-evaluation.
- **Experiment** — a hypothesis that must not be presented as a finished capability.
- **Open decision** — a missing choice required before the corresponding stage.

When the overview text and the detailed contract diverge, the following apply: `tasks.md` for statuses and the division of work between the local machine and Colab, `specification.md` for project scope, `model.md` for computations, `runtime.md` for transport, `development.md` for tooling, `validation.md` for acceptance. The divergence is resolved before the affected component is implemented.

## Current readiness

The path from tensor and audio-codec verification to a live full-duplex conversation and controlled fine-tuning is documented. English, a chatbot demo, Google Drive, and a first-stage budget of 500 units have been chosen. The weight set and reading corpus are fixed in the code. The full scenario requires an A100 with 40 GB or more; the available T4 does not fit the BF16 profile. Therefore, the existence of a specification does not mean the system is ready to be trained on a known budget or that the voice model can be launched with a single command.

The model commands in the documents represent the target interface of a future application. `pyproject.toml`, dependency groups, and `uv.lock` have already been created. Scaffold checks: `uv run --locked pytest`, `uv run --locked ruff check .`, `uv run --locked ruff format --check .`, `uv run --locked mypy src/aether`.

Schemas and constraints are described in [configuration.md](configuration.md). Token alignment is implemented without a model backend; this is not a check of speech quality. An initial codec was chosen for the first result, with the Manifestro codec planned afterward along with a separate compatibility check.

[Artifact verification](artifacts.md) and [audio buffers](audio.md) are implemented locally; the real state_dict, tokenizer, and speech quality have not yet been verified.

A locally verified [full notebook](../notebooks/aether_colab.ipynb) and [Colab instructions](colab.md) are ready. A16/A17 remain open until remote verification.
