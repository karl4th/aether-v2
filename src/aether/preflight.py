"""Resource report only, never authorization to construct a model or trainer."""

import importlib.util
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


def collect_preflight() -> dict[str, Any]:
    """Called explicitly in the remote notebook; nvidia-smi performs no training."""
    try:
        has_colab = importlib.util.find_spec("google.colab") is not None
    except (ImportError, ModuleNotFoundError):
        has_colab = False
    memory_kib = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                memory_kib = int(line.split()[1])
    gpu: dict[str, object]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        gpu = {
            "exit_code": result.returncode,
            "output": result.stdout.strip(),
            "error": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        gpu = {"error": type(exc).__name__}
    disk = shutil.disk_usage(Path.cwd())
    return {
        "schema_version": 1,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "ram_kib": memory_kib,
        "disk_free_bytes": disk.free,
        "gpu": gpu,
        "runtime_observations": {
            "google_colab_importable": has_colab,
            "content_directory_exists": Path("/content").is_dir(),
            "colab_release_tag_present": bool(os.environ.get("COLAB_RELEASE_TAG")),
            "colab_backend_version_present": bool(os.environ.get("COLAB_BACKEND_VERSION")),
        },
        "managed_remote_runtime_verified": False,
        "paid_plan_verified": False,
        "training_authorized": False,
        "note": "Observations are not proof of remote managed execution or a paid plan.",
    }
