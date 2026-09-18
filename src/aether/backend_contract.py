"""Validate the installed pinned backend API without loading any weights."""

import importlib
import inspect
from typing import Any


def validate_backend_api() -> dict[str, Any]:
    """Remote setup check only; imports optional modules but constructs no model."""
    loaders = importlib.import_module("moshi.models.loaders")
    model = importlib.import_module("moshi.models.lm")
    requirements = (
        (loaders.CheckpointInfo.from_hf_repo, {"hf_repo", "revision"}),
        (loaders.CheckpointInfo.get_moshi, {"device", "dtype"}),
        (loaders.CheckpointInfo.get_mimi, {"device"}),
        (loaders.get_moshi_lm, {"lm_kwargs_overrides"}),
        (model.LMModel, {"gradient_checkpointing"}),
        (model.LMGen, {"lm_model", "temp", "temp_text"}),
    )
    for function, names in requirements:
        if not names <= set(inspect.signature(function).parameters):
            raise RuntimeError(f"installed backend API is incompatible: {function.__name__}")
    configuration = loaders._lm_kwargs
    if configuration["delays"] != [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]:
        raise RuntimeError("backend delay contract changed")
    if configuration["dep_q"] != 8 or configuration["text_card"] != 32000:
        raise RuntimeError("backend channel/vocabulary contract changed")
    return {"api": "compatible", "weights_loaded": False}
