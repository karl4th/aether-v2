"""Accidental-local-execution guard, not infrastructure or billing attestation.

A short-lived permit is bound to the current VM boot and a live anchor process
(a notebook kernel on Colab, the container init on RunPod). Several independent
runtime observations are required before backend imports.
"""

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Annotated

from pydantic import Field

from aether.config import StrictConfig, _unique_object


class RemoteRuntimeError(RuntimeError):
    """The requested remote operation was not authorized in this runtime."""


class RuntimePermit(StrictConfig):
    boot_id: str
    kernel_pid: Annotated[int, Field(gt=0)]
    created_at: Annotated[float, Field(gt=0)]
    expires_at: Annotated[float, Field(gt=0)]
    user_confirmed_remote_paid: bool
    allow_training: bool
    budget_units: Annotated[int, Field(gt=0, le=500)]
    source_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    # Set only for a RunPod-issued permit; absent (None) for a Colab one.
    pod_id: Annotated[str, Field(min_length=1)] | None = None


def _detect_platform() -> str | None:
    """Return "colab", "runpod", or None; never guesses from a single signal."""
    if Path("/content").is_dir() and all(
        os.environ.get(key) for key in ("COLAB_RELEASE_TAG", "COLAB_BACKEND_VERSION")
    ):
        return "colab"
    if os.environ.get("RUNPOD_POD_ID") and all(
        os.environ.get(key) for key in ("RUNPOD_GPU_COUNT", "RUNPOD_DC_ID", "RUNPOD_PUBLIC_IP")
    ):
        return "runpod"
    return None


def require_remote_runtime(*, training: bool = False, permit_path: str | Path) -> RuntimePermit:
    """Refuse local execution before imports/downloads/optimizer construction.

    Neither Colab's nor RunPod's packages need exist inside uv's isolated Python.
    Signals do not prove payment or defeat deliberate spoofing; the user confirms
    paid remote execution when the permit is created, and actual usage remains
    visible in that provider's own billing UI.
    """
    reason = "TRAINING_ENVIRONMENT_REQUIRED" if training else "REMOTE_RUNTIME_REQUIRED"
    try:
        if platform.system() != "Linux":
            raise ValueError("requires a managed remote Linux GPU runtime")
        kind = _detect_platform()
        if kind is None:
            raise ValueError("no supported managed remote runtime observed (Colab or RunPod)")
        raw = Path(permit_path).read_text(encoding="utf-8")
        json.loads(raw, object_pairs_hook=_unique_object)
        permit = RuntimePermit.model_validate_json(raw)
        if not permit.user_confirmed_remote_paid or (training and not permit.allow_training):
            raise ValueError("explicit authorization is missing")
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if permit.boot_id != boot_id:
            raise ValueError("permit belongs to another runtime")
        now = time.time()
        if not permit.created_at <= now < permit.expires_at:
            raise ValueError("runtime permit has expired or has an invalid timestamp")
        if permit.expires_at - permit.created_at > 12 * 3600:
            raise ValueError("permit lifetime exceeds 12 hours")
        anchor = Path(f"/proc/{permit.kernel_pid}")
        if not anchor.is_dir():
            raise ValueError("authorizing anchor process is no longer alive")
        command = (anchor / "cmdline").read_bytes().replace(b"\x00", b" ")
        if kind == "colab":
            if not any(word in command for word in (b"ipykernel", b"colab_kernel_launcher")):
                raise ValueError("permit process is not a notebook kernel")
            kernel_environment = (anchor / "environ").read_bytes().split(b"\x00")
            for key in (b"COLAB_RELEASE_TAG=", b"COLAB_BACKEND_VERSION="):
                if not any(
                    item.startswith(key) and len(item) > len(key) for item in kernel_environment
                ):
                    raise ValueError("authorizing kernel lacks Colab observations")
        else:
            if permit.pod_id != os.environ.get("RUNPOD_POD_ID"):
                raise ValueError("permit belongs to another RunPod pod")
            if not any(word in command for word in (b"docker-init", b"nvidia_entrypoint")):
                raise ValueError("permit anchor is not the container init")
        if not Path("/proc/driver/nvidia/version").is_file():
            raise ValueError("no NVIDIA driver in this runtime; select a GPU runtime")
        return permit
    except (OSError, ValueError) as exc:
        raise RemoteRuntimeError(
            f"{reason}: {exc}; use notebooks/aether_colab.ipynb or `aether permit` on a RunPod pod"
        ) from exc


def create_runpod_permit(
    *,
    confirm_remote_paid: bool,
    allow_training: bool,
    budget_units: int,
    hours: float = 12.0,
    repo_path: str | Path = ".",
) -> RuntimePermit:
    """Build a permit for a plain SSH session on a RunPod pod; no notebook involved.

    Bound to the container's own init process (PID 1), which is alive for exactly
    as long as the pod exists, and to RUNPOD_POD_ID, so a copied permit does not
    authorize a different pod.
    """
    if not confirm_remote_paid:
        raise ValueError("remote paid execution must be explicitly confirmed")
    if type(budget_units) is not int or not 0 < budget_units <= 500:
        raise ValueError("budget_units must be an integer between 1 and 500")
    if not isinstance(hours, int | float) or not 0 < hours <= 12:
        raise ValueError("permit lifetime must be between 0 and 12 hours")
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id:
        raise ValueError("RUNPOD_POD_ID is not set; this does not look like a RunPod pod")
    source = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    now = time.time()
    return RuntimePermit(
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        kernel_pid=1,
        pod_id=pod_id,
        created_at=now,
        expires_at=now + hours * 3600,
        user_confirmed_remote_paid=True,
        allow_training=allow_training,
        budget_units=budget_units,
        source_revision=source,
    )
