import subprocess
from unittest.mock import patch

from aether.preflight import collect_preflight


def test_observations_do_not_authorize_training() -> None:
    with patch(
        "aether.preflight.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, "Synthetic GPU, 100 MiB, mock-driver", ""),
    ):
        report = collect_preflight()
    assert report["training_authorized"] is False
    assert report["managed_remote_runtime_verified"] is False
    assert report["paid_plan_verified"] is False
    assert "Synthetic GPU" in report["gpu"]["output"]
