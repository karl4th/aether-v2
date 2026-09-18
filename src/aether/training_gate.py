"""Compatibility wrapper for the common guarded remote training entry."""

from pathlib import Path

from aether.config import ExperimentConfig
from aether.remote import RemoteRuntimeError, require_remote_runtime


class TrainingBlockedError(RuntimeError):
    """Training was rejected before model or optimizer construction."""


def require_training_environment(
    config: ExperimentConfig | None = None, *, permit_path: str | Path | None = None
) -> None:
    if permit_path is None or (
        config is not None
        and (config.profile != "colab_train" or not config.training.allow_training)
    ):
        raise TrainingBlockedError(
            "TRAINING_ENVIRONMENT_REQUIRED: use notebooks/aether_colab.ipynb with explicit "
            "remote paid-runtime confirmation and training permission; local runtime forbidden"
        )
    try:
        require_remote_runtime(training=True, permit_path=permit_path)
    except RemoteRuntimeError as exc:
        raise TrainingBlockedError(str(exc)) from exc
