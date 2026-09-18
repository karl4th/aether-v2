"""Accidental-local-execution guard, not infrastructure or billing attestation.

A short-lived notebook permit is bound to the current VM boot and live kernel.
Several independent runtime observations are required before backend imports.
"""

import json
import os
import platform
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


def require_remote_runtime(*, training: bool = False, permit_path: str | Path) -> RuntimePermit:
    """Refuse local execution before imports/downloads/optimizer construction.

    Colab's package need not exist inside uv's isolated Python. Signals do not
    prove payment or defeat deliberate spoofing; the user confirms paid remote
    execution in the notebook, and actual unit usage remains visible in its UI.
    """
    reason = "TRAINING_ENVIRONMENT_REQUIRED" if training else "REMOTE_RUNTIME_REQUIRED"
    try:
        if platform.system() != "Linux" or not Path("/content").is_dir():
            raise ValueError("requires a managed remote Linux Colab VM")
        if not all(os.environ.get(key) for key in ("COLAB_RELEASE_TAG", "COLAB_BACKEND_VERSION")):
            raise ValueError("missing Colab runtime environment observations")
        raw = Path(permit_path).read_text(encoding="utf-8")
        json.loads(raw, object_pairs_hook=_unique_object)
        permit = RuntimePermit.model_validate_json(raw)
        if not permit.user_confirmed_remote_paid or (training and not permit.allow_training):
            raise ValueError("explicit notebook authorization is missing")
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if permit.boot_id != boot_id:
            raise ValueError("permit belongs to another runtime")
        now = time.time()
        if not permit.created_at <= now < permit.expires_at:
            raise ValueError("runtime permit has expired or has an invalid timestamp")
        if permit.expires_at - permit.created_at > 12 * 3600:
            raise ValueError("permit lifetime exceeds 12 hours")
        kernel = Path(f"/proc/{permit.kernel_pid}")
        if not kernel.is_dir():
            raise ValueError("authorizing notebook kernel is no longer alive")
        command = (kernel / "cmdline").read_bytes().replace(b"\x00", b" ")
        if not any(word in command for word in (b"ipykernel", b"colab_kernel_launcher")):
            raise ValueError("permit process is not a notebook kernel")
        kernel_environment = (kernel / "environ").read_bytes().split(b"\x00")
        for key in (b"COLAB_RELEASE_TAG=", b"COLAB_BACKEND_VERSION="):
            if not any(
                item.startswith(key) and len(item) > len(key) for item in kernel_environment
            ):
                raise ValueError("authorizing kernel lacks Colab observations")
        if not Path("/proc/driver/nvidia/version").is_file():
            raise ValueError("no NVIDIA driver in this runtime; select a GPU runtime")
        return permit
    except (OSError, ValueError) as exc:
        raise RemoteRuntimeError(f"{reason}: {exc}; use notebooks/aether_colab.ipynb") from exc
