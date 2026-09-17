"""Single fail-closed entry gate; no backend is imported by this module."""

from aether.config import ExperimentConfig


class TrainingBlockedError(RuntimeError):
    """Training was rejected before model or optimizer construction."""


def require_training_environment(config: ExperimentConfig | None = None) -> None:
    """Keep closed until managed remote runtime evidence is verified in Colab.

    Neither configuration, environment flags, /content nor google.colab proves
    remote managed execution or a paid plan. No bypass is exposed here.
    Future training entry points must call this before any backend construction.
    """
    reason = "проверка удалённого управляемого runtime ещё не реализована"
    if config is not None and (
        config.profile != "colab_train" or not config.training.allow_training
    ):
        reason = "профиль или явное разрешение обучения отсутствуют"
    raise TrainingBlockedError(
        f"TRAINING_ENVIRONMENT_REQUIRED: {reason}. Обучение закрыто. "
        "Будущий notebooks/aether_colab.ipynb: удалённый платный Google Colab, "
        "проверка ресурсов и runtime; local runtime запрещён."
    )
