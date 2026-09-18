from pathlib import Path
from unittest.mock import patch

import pytest

from aether.remote import RemoteRuntimeError, require_remote_runtime


def test_local_refusal_before_reading_permit(tmp_path: Path) -> None:
    with patch("aether.remote.platform.system", return_value="Darwin"):
        with patch.object(Path, "read_text", side_effect=AssertionError("permit read")):
            with pytest.raises(RemoteRuntimeError, match="TRAINING_ENVIRONMENT_REQUIRED"):
                require_remote_runtime(training=True, permit_path=tmp_path / "permit.json")


def test_linux_alone_does_not_authorize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    with patch("aether.remote.platform.system", return_value="Linux"):
        with patch.object(Path, "is_dir", return_value=True):
            with pytest.raises(RemoteRuntimeError, match="missing Colab"):
                require_remote_runtime(permit_path="absent")


@pytest.mark.parametrize("fault", [None, "expired", "boot", "training", "kernel", "env", "driver"])
def test_full_permit_contract_is_not_a_single_flag(
    monkeypatch: pytest.MonkeyPatch, fault: str | None
) -> None:
    import json

    permit = {
        "boot_id": "vm-boot",
        "kernel_pid": 123,
        "created_at": 1000.0,
        "expires_at": 2000.0,
        "user_confirmed_remote_paid": True,
        "allow_training": fault != "training",
        "budget_units": 500,
        "source_revision": "a" * 40,
    }
    monkeypatch.setenv("COLAB_RELEASE_TAG", "test")
    monkeypatch.setenv("COLAB_BACKEND_VERSION", "test")

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        return (
            json.dumps(permit)
            if str(path) == "permit.json"
            else ("other" if fault == "boot" else "vm-boot")
        )

    def read_bytes(path: Path) -> bytes:
        if path.name == "cmdline":
            return b"python\x00-m\x00" + (b"local" if fault == "kernel" else b"ipykernel")
        return b"" if fault == "env" else b"COLAB_RELEASE_TAG=x\x00COLAB_BACKEND_VERSION=y"

    with (
        patch("aether.remote.platform.system", return_value="Linux"),
        patch("aether.remote.time.time", return_value=2500 if fault == "expired" else 1500),
        patch.object(Path, "is_dir", return_value=True),
        patch.object(Path, "is_file", return_value=fault != "driver"),
        patch.object(Path, "read_text", read_text),
        patch.object(Path, "read_bytes", read_bytes),
    ):
        if fault:
            with pytest.raises(RemoteRuntimeError):
                require_remote_runtime(training=True, permit_path="permit.json")
        else:
            assert require_remote_runtime(training=True, permit_path="permit.json").allow_training
