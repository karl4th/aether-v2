"""CLI dispatch on mocks; no backend construction or optimizer steps."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from aether.cli import main


@pytest.mark.parametrize("command", ["infer", "train", "evaluate", "prepare-data"])
def test_remote_commands_reject_local_machine(command: str, tmp_path: Path) -> None:
    args = [command, "--permit", str(tmp_path / "absent")]
    if command == "prepare-data":
        args += ["--output", str(tmp_path / "data")]
    else:
        args += ["--config", "configs/training/colab_lora.json"]
    if command == "infer":
        args += ["--input", "absent.wav"]
    assert main(args) == 4
    assert list(tmp_path.iterdir()) == []


def test_inference_cli_forwards_adapter_and_sampling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from aether import backend

    monkeypatch.setattr("aether.cli.require_remote_runtime", Mock())
    infer = Mock(return_value={"audio": "reply.wav", "text": "fixture"})
    monkeypatch.setattr(backend, "infer_file", infer)
    assert (
        main(
            [
                "infer",
                "--config",
                "configs/model/remote.json",
                "--permit",
                "permit",
                "--input",
                "question.wav",
                "--output",
                str(tmp_path / "result"),
                "--checkpoint",
                "checkpoint",
            ]
        )
        == 0
    )
    assert infer.call_args.kwargs["adapter_checkpoint"] == Path("checkpoint")
    assert infer.call_args.kwargs["seed"] == 42


def test_training_cli_pilot_and_resume_dispatch_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from aether import dataset, trainer

    monkeypatch.setattr("aether.cli.require_remote_runtime", Mock())
    manifest = object()
    monkeypatch.setattr(dataset, "load_manifest", Mock(return_value=manifest))
    train = Mock(return_value={"step": 5, "checkpoint": "fixture-checkpoint"})
    monkeypatch.setattr(trainer, "train", train)
    assert (
        main(
            [
                "train",
                "--config",
                "configs/training/colab_lora.json",
                "--permit",
                "permit",
                "--dataset",
                "dataset.json",
                "--output",
                str(tmp_path),
                "--stop-after-steps",
                "5",
                "--resume",
                "previous",
            ]
        )
        == 0
    )
    assert train.call_args.args[0] is manifest
    assert train.call_args.kwargs["stop_after_steps"] == 5
    assert train.call_args.kwargs["resume"] == Path("previous")
