import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from aether.remote import RemoteRuntimeError, create_runpod_permit, require_remote_runtime


def test_local_refusal_before_reading_permit(tmp_path: Path) -> None:
    with patch("aether.remote.platform.system", return_value="Darwin"):
        with patch.object(Path, "read_text", side_effect=AssertionError("permit read")):
            with pytest.raises(RemoteRuntimeError, match="TRAINING_ENVIRONMENT_REQUIRED"):
                require_remote_runtime(training=True, permit_path=tmp_path / "permit.json")


def test_linux_alone_does_not_authorize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    with patch("aether.remote.platform.system", return_value="Linux"):
        with patch.object(Path, "is_dir", return_value=True):
            with pytest.raises(RemoteRuntimeError, match="no supported managed remote runtime"):
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


@pytest.mark.parametrize(
    "fault", [None, "expired", "boot", "training", "init", "env", "pod", "driver"]
)
def test_runpod_permit_contract_is_not_a_single_flag(
    monkeypatch: pytest.MonkeyPatch, fault: str | None
) -> None:
    import json

    permit = {
        "boot_id": "vm-boot",
        "kernel_pid": 1,
        "pod_id": "pod-abc",
        "created_at": 1000.0,
        "expires_at": 2000.0,
        "user_confirmed_remote_paid": True,
        "allow_training": fault != "training",
        "budget_units": 500,
        "source_revision": "a" * 40,
    }
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.setenv("RUNPOD_POD_ID", "other-pod" if fault == "pod" else "pod-abc")
    for key, value in (
        ("RUNPOD_GPU_COUNT", "1"),
        ("RUNPOD_DC_ID", "EU-SE-1"),
        ("RUNPOD_PUBLIC_IP", "1.2.3.4"),
    ):
        if fault == "env" and key == "RUNPOD_DC_ID":
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        return (
            json.dumps(permit)
            if str(path) == "permit.json"
            else ("other" if fault == "boot" else "vm-boot")
        )

    def read_bytes(path: Path) -> bytes:
        return b"/sbin/" + (b"local-shell" if fault == "init" else b"docker-init")

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
            result = require_remote_runtime(training=True, permit_path="permit.json")
            assert result.allow_training
            assert result.pod_id == "pod-abc"


def test_colab_permit_without_pod_id_still_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    """pod_id defaults to None; a pre-existing Colab permit JSON lacks the field."""
    import json

    permit = {
        "boot_id": "vm-boot",
        "kernel_pid": 123,
        "created_at": 1000.0,
        "expires_at": 2000.0,
        "user_confirmed_remote_paid": True,
        "allow_training": False,
        "budget_units": 500,
        "source_revision": "a" * 40,
    }
    monkeypatch.setenv("COLAB_RELEASE_TAG", "test")
    monkeypatch.setenv("COLAB_BACKEND_VERSION", "test")

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        return json.dumps(permit) if str(path) == "permit.json" else "vm-boot"

    def read_bytes(path: Path) -> bytes:
        if path.name == "cmdline":
            return b"python\x00-m\x00ipykernel"
        return b"COLAB_RELEASE_TAG=x\x00COLAB_BACKEND_VERSION=y"

    with (
        patch("aether.remote.platform.system", return_value="Linux"),
        patch("aether.remote.time.time", return_value=1500),
        patch.object(Path, "is_dir", return_value=True),
        patch.object(Path, "is_file", return_value=True),
        patch.object(Path, "read_text", read_text),
        patch.object(Path, "read_bytes", read_bytes),
    ):
        assert require_remote_runtime(permit_path="permit.json").pod_id is None


def test_create_runpod_permit_requires_explicit_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-abc")
    with pytest.raises(ValueError, match="explicitly confirmed"):
        create_runpod_permit(confirm_remote_paid=False, allow_training=False, budget_units=10)


def test_create_runpod_permit_requires_runpod_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    with pytest.raises(ValueError, match="does not look like a RunPod pod"):
        create_runpod_permit(confirm_remote_paid=True, allow_training=False, budget_units=10)


@pytest.mark.parametrize("budget_units", [0, -1, 501, 1.5])
def test_create_runpod_permit_rejects_bad_budget(
    monkeypatch: pytest.MonkeyPatch, budget_units: object
) -> None:
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-abc")
    with pytest.raises(ValueError, match="budget_units"):
        create_runpod_permit(
            confirm_remote_paid=True, allow_training=False, budget_units=budget_units
        )


@pytest.mark.parametrize("hours", [0, -1, 12.1])
def test_create_runpod_permit_rejects_bad_lifetime(
    monkeypatch: pytest.MonkeyPatch, hours: float
) -> None:
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-abc")
    with pytest.raises(ValueError, match="lifetime"):
        create_runpod_permit(
            confirm_remote_paid=True, allow_training=False, budget_units=10, hours=hours
        )


def test_create_runpod_permit_builds_a_valid_permit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-abc")
    monkeypatch.setattr(
        subprocess,
        "run",
        Mock(return_value=Mock(stdout="f" * 40 + "\n")),
    )
    with patch.object(Path, "read_text", return_value="vm-boot"):
        permit = create_runpod_permit(
            confirm_remote_paid=True, allow_training=True, budget_units=250, hours=6
        )
    assert permit.pod_id == "pod-abc"
    assert permit.kernel_pid == 1
    assert permit.allow_training is True
    assert permit.budget_units == 250
    assert permit.source_revision == "f" * 40
    assert permit.expires_at - permit.created_at == pytest.approx(6 * 3600)
