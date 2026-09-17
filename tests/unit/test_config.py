"""Schema rejection and training gate tests; no parameter updates."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from aether.config import ExperimentConfig, load_config
from aether.training_gate import TrainingBlockedError, require_training_environment

TINY = Path("configs/model/tiny.json")


@pytest.mark.parametrize(
    "path,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("extra", 1),
        ("model.temporal.dimension", 31),
        ("model.temporal.layers", 0),
        ("model.depth.heads", "2"),
        ("model.audio_codebooks_per_stream", 7),
        ("model.delays", [0] * 17),
        ("model.delays", [0]),
        ("model.audio_vocab_size", 128),
        ("model.temporal.context", 128),
        ("runtime.max_session_frames", 15),
        ("runtime.completion_reserve", 0),
        ("runtime.audio_timeout_seconds", float("nan")),
        ("training.learning_rate", float("inf")),
        ("training.allow_training", True),
        ("training.allow_training", "false"),
        ("data.batch_size", 3),
        ("model.depth.unknown", 1),
        ("codec.vocabulary_size", 31),
        ("codec.codebooks", 7),
        ("codec.token_contract_version", 2),
        ("device", "cuda"),
    ],
)
def test_reject_invalid(path: str, value: object) -> None:
    data = json.loads(TINY.read_text())
    keys = path.split(".")
    target = data
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate_json(json.dumps(data))


def test_defaults_bos_and_roundtrip() -> None:
    config = load_config(TINY)
    assert config.model.audio_bos == 32
    assert config.model.text_bos == 64
    assert config == ExperimentConfig.model_validate_json(config.model_dump_json())
    data = json.loads(TINY.read_text())
    del data["profile"]
    assert ExperimentConfig.model_validate_json(json.dumps(data)).profile == "local_test"


def test_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate"):
        load_config(path)


@pytest.mark.parametrize("colab", [False, True])
def test_gate_precedes_backend(monkeypatch: pytest.MonkeyPatch, colab: bool) -> None:
    data = json.loads(TINY.read_text())
    if colab:
        data.update(profile="colab_train", device="cuda", purpose="training")
        data["training"]["allow_training"] = True
        data["codec"].update(implementation="initial", token_semantics="unverified")
    config = ExperimentConfig.model_validate_json(json.dumps(data))
    monkeypatch.setenv("COLAB_RELEASE_TAG", "claimed-colab-runtime")
    monkeypatch.setenv("AETHER_ALLOW_TRAINING", "1")
    backend = Mock()
    with pytest.raises(TrainingBlockedError, match="notebooks/aether_colab.ipynb"):
        require_training_environment(config)
        backend.load_weights()
        backend.create_optimizer()
    assert backend.mock_calls == []
