# aether — Python and uv

## Execution Environments

Mandatory constraint: the current machine is for development and limited testing only. Training, optimizer steps, and retraining even a single example are performed exclusively in the remote GPU runtime of paid Google Colab. Local connection of the runtime to Colab is not used. Full model and GPU checks are likewise performed in Colab; locally, small forward fixtures and mock components are used.

The target notebook is `notebooks/aether_colab.ipynb`. It invokes the package via `uv run --locked` rather than duplicating the trainer in cells. Training sections are disabled by default. The plan for environment setup, run protection, and persistent storage is given in [tasks.md](tasks.md).

## 1. Required Stack

Python 3.12 is the project's initial minor version. The exact patch version is pinned when the environment is created. uv manages Python, `.venv`, dependencies, and commands. `pyproject.toml` defines the package, `uv.lock` pins the resolved dependencies; both files are tracked in Git.

The CLI scaffold is set up; `--help` and `--version` work. `train --validate-only` validates the schema, `inspect --config ... --manifest ...` validates a synthetic bundle; `infer`, `prepare-data`, `train`, `evaluate`, `serve`, `talk`, `tunnel` are available given remote runtime authorization or explicit local use; `train` without `--validate-only` returns `TRAINING_ENVIRONMENT_REQUIRED`. The command semantics below describe the target design, not functionality that is already available.

| Layer | Choice |
|---|---|
| Model, training, and tensors | PyTorch |
| Arrays outside the model | NumPy |
| Weights | safetensors |
| Text tokenizer | SentencePiece, compatible with the weights |
| Configuration validation | Pydantic |
| Server and WebSocket | aiohttp |
| Local audio device I/O | sounddevice; file reading uses soundfile |
| CLI | argparse from the standard library |
| Tests | pytest, pytest-asyncio |
| Style and static checks | Ruff, mypy |

Library versions and the PyTorch build source are chosen when the target platform is verified and are pinned by the lock file. CUDA compatibility is determined not by the presence of the word `cuda` in the configuration, but by the model actually running successfully and the driver being verified.

## 2. Package Structure

```text
src/aether/
  __init__.py
  __main__.py
  cli.py
  config.py
  audio/          capture.py, resample.py, buffers.py
  model/          codec.py, quantizer.py, temporal.py, depth.py, embeddings.py
  inference/      engine.py, scheduler.py, state.py, checkpoint.py
  server.py       protocol-v1 WebSocket service, bounded queues, single session
  client.py       microphone/speaker bridge over the same protocol
  tunnel.py       Cloudflare quick tunnel helper, since Colab has no public IP
  training/       dataset.py, alignment.py, losses.py, trainer.py
  evaluation/     runner.py, metrics.py, report.py
tests/
  unit/
  integration/
  gpu/
configs/
  model/
  runtime/
  training/
docs/
notebooks/
  aether_colab.ipynb
```

The model does not import the CLI or audio devices. The server does not implement token sampling itself; it calls the same engine used offline. Data preparation does not depend on live sessions. Shared contracts are defined once.

## 3. Environment

After the project configuration is created:

```bash
uv python install 3.12
uv python pin 3.12
uv sync --locked --group dev
uv run --locked aether --help
```

`uv sync --locked` verifies that the lock file is up to date. A plain `uv sync` may update it. Dependency updates are performed as a separate change with diff review. CI uses `--locked` and the same pinned uv version.

Adding a dependency is done via `uv add`, and development tools via `uv add --group dev`. Manual `pip install` is not used within the project environment as a means of changing the project's composition. System audio libraries and GPU drivers are installed separately and documented in the environment instructions.

## 4. Dependency Groups

- Main package: model inference output, configuration, weight loading, and the server.
- `dev`: testing, formatting, and type checking.
- `model`: pinned Python backend, PyTorch/torchaudio 2.8, and weight dependencies. Installed only in Colab.
- `train`: includes `model`; local tests do not install this group.
- `audio`: local input/output device access for the `talk` client.

These are dependency groups in `pyproject.toml`. The examples below assume they are defined. Experimental libraries are not all added to the main group.

```bash
uv sync --locked --group dev --group audio
uv run --locked --group dev ruff check .
uv run --locked --group dev ruff format --check .
uv run --locked --group dev mypy src/aether
uv run --locked --group dev pytest
```

## 5. Target CLI

```bash
uv run --locked aether inspect --config configs/model/base.json
uv run --locked aether infer --config configs/model/base.json --input sample.wav --output answer.wav
uv run --locked aether serve --config configs/runtime/local.json
uv run --locked --group audio aether talk --url ws://127.0.0.1:8998/v1/session
uv run --locked aether evaluate --config configs/evaluation.json
```

These model commands with full weights are executed in Colab; locally, test configurations are used. Only within the remote Colab runtime, after resource verification and explicit enabling of training:

```bash
uv run --locked --group train aether train --config configs/training/adapter.json
```

Locally, only validation of the training configuration via `train --validate-only` is permitted, without creating an optimizer or starting training.

`inspect` checks files, sizes, vocabularies, dtype, device, and memory without opening the microphone. `infer` also saves the text and manifest. `serve` warms the worker up to readiness. `talk` explicitly opens the microphone. `evaluate` does not modify weights. `train` writes to a new run directory without overwriting the previous experiment.

## 6. Implementation Style

Public functions receive type annotations. Documentation of tensor APIs specifies shape, dtype, device, value range, and state ownership. Configurations are validated before large weights are loaded. Unknown keys are treated as errors.

The `torch.inference_mode()` context is used for inference, and the model is set to `eval()`. Global mutable session state is prohibited. The audio callback does not perform inference, file writes, or network waiting; it only passes data through a bounded queue.

Compilation optimizations and CUDA graphs are added after correct eager inference is established. For each optimization, the ability to compare against the simple implementation is preserved.

## 7. Reproducibility

The run manifest contains the commit, the state of modified files, the `uv.lock` hash, Python and uv versions, PyTorch version, GPU, driver, dtype, configuration, seed, and hashes of all weights and the tokenizer. The same seed does not guarantee bitwise identical results across different devices; comparisons use specified tolerances.

`.venv`, weights, caches, microphone recordings, local secrets, and large reports are not included in Git. Small synthetic fixtures and data schemas are included. The project name in the user interface, package, and documents is `aether`.
