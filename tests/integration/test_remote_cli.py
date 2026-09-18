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


def test_serve_cli_forwards_host_port_and_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from aether import server

    served = Mock()
    monkeypatch.setattr(server, "serve", served)
    monkeypatch.setenv("AETHER_LIVE_TOKEN", "secret")
    permit = str(tmp_path / "permit.json")
    assert (
        main(
            [
                "serve",
                "--permit",
                permit,
                "--host",
                "0.0.0.0",
                "--port",
                "9090",
                "--checkpoint",
                "checkpoints/step-000020",
            ]
        )
        == 0
    )
    served.assert_called_once_with(
        permit,
        host="0.0.0.0",
        port=9090,
        token="secret",
        checkpoint="checkpoints/step-000020",
    )


def test_talk_cli_forwards_url_duration_and_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    from aether import client

    calls = []

    async def fake_talk(url: str, *, duration=None, seed=42, auth_token=None) -> None:
        calls.append(dict(url=url, duration=duration, seed=seed, auth_token=auth_token))

    monkeypatch.setattr(client, "talk", fake_talk)
    monkeypatch.delenv("AETHER_LIVE_TOKEN", raising=False)
    assert (
        main(["talk", "--url", "wss://example.invalid/live", "--duration", "5", "--seed", "7"]) == 0
    )
    assert calls == [
        {"url": "wss://example.invalid/live", "duration": 5.0, "seed": 7, "auth_token": None}
    ]


def test_tunnel_cli_forwards_port(monkeypatch: pytest.MonkeyPatch) -> None:
    from aether import tunnel

    calls = []
    monkeypatch.setattr(tunnel, "run_tunnel", calls.append)
    assert main(["tunnel", "--port", "9999"]) == 0
    assert calls == [9999]
