"""Backend-free validation for the supported remote inference profile."""

from typing import Annotated, Literal

from pydantic import Field

from aether.config import StrictConfig


class InferenceConfig(StrictConfig):
    task: Literal["speech_inference"] = "speech_inference"
    context: Annotated[int, Field(ge=32, le=512)] = 256
    tail_seconds: Annotated[float, Field(ge=0, le=30)] = 8.0
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 42
