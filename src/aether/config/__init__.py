"""Versioned, backend-independent configuration contracts."""

import json
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

PositiveInt = Annotated[int, Field(gt=0)]
NonnegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
DELAYS = (0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1)


class StrictConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Annotated[int, Field(ge=1, le=1)] = 1


class TransformerConfig(StrictConfig):
    dimension: PositiveInt
    layers: PositiveInt
    heads: PositiveInt

    @model_validator(mode="after")
    def divisible_heads(self) -> Self:
        if self.dimension % self.heads:
            raise ValueError("dimension must be divisible by heads")
        return self


class TemporalConfig(TransformerConfig):
    context: PositiveInt


class ModelConfig(StrictConfig):
    audio_codebooks_per_stream: Annotated[int, Field(ge=8, le=8)] = 8
    audio_vocab_size: PositiveInt
    text_vocab_size: Annotated[int, Field(ge=4)]
    temporal: TemporalConfig
    depth: TransformerConfig
    delays: tuple[NonnegativeInt, ...] = DELAYS

    @property
    def audio_bos(self) -> int:
        return self.audio_vocab_size

    @property
    def text_bos(self) -> int:
        return self.text_vocab_size

    @model_validator(mode="after")
    def channel_contract(self) -> Self:
        if self.delays != DELAYS:
            raise ValueError("schema v1 requires the exact 17-channel delay schedule")
        return self


class RuntimeConfig(StrictConfig):
    sample_rate: Annotated[int, Field(ge=24000, le=24000)] = 24000
    frame_samples: Annotated[int, Field(ge=1920, le=1920)] = 1920
    max_session_frames: PositiveInt
    initialization_reserve: PositiveInt = 1
    completion_reserve: PositiveInt = 1
    max_queue_frames: PositiveInt = 4
    audio_timeout_seconds: PositiveFloat = 5.0


class DataConfig(StrictConfig):
    batch_size: PositiveInt = 1
    sequence_frames: PositiveInt
    seed: NonnegativeInt = 0


class TrainingConfig(StrictConfig):
    allow_training: bool = False
    max_steps: PositiveInt = 1
    learning_rate: PositiveFloat = 0.0001


class CodecConfig(StrictConfig):
    """Token contract only; does not claim a compatible checkpoint exists."""

    implementation: Literal["initial", "synthetic"]
    token_contract_version: Annotated[int, Field(ge=1, le=1)] = 1
    sample_rate: Annotated[int, Field(ge=24000, le=24000)] = 24000
    frame_samples: Annotated[int, Field(ge=1920, le=1920)] = 1920
    codebooks: Annotated[int, Field(ge=8, le=8)] = 8
    vocabulary_size: PositiveInt
    token_semantics: Literal["unverified", "synthetic"]


class ExperimentConfig(StrictConfig):
    profile: Literal["local_test", "colab_train"] = "local_test"
    purpose: Literal["synthetic_forward_tests", "training"]
    device: Literal["cpu", "cuda"] = "cpu"
    dtype: Literal["float32", "bfloat16"] = "float32"
    model: ModelConfig
    codec: CodecConfig
    runtime: RuntimeConfig
    data: DataConfig
    training: TrainingConfig = TrainingConfig()

    @model_validator(mode="after")
    def resource_contract(self) -> Self:
        r, m = self.runtime, self.model
        if self.codec.vocabulary_size != m.audio_vocab_size:
            raise ValueError("codec vocabulary must match model audio vocabulary")
        if self.profile == "local_test" and (
            self.codec.implementation != "synthetic" or self.codec.token_semantics != "synthetic"
        ):
            raise ValueError("local_test requires a synthetic codec contract")
        if self.profile == "colab_train" and self.codec.implementation != "initial":
            raise ValueError("colab_train requires the initial codec contract")
        reserve = r.initialization_reserve + r.completion_reserve
        if max(r.max_session_frames, self.data.sequence_frames) + reserve > m.temporal.context:
            raise ValueError("session/data frames plus explicit reserves exceed context")
        if self.profile == "local_test":
            if (self.device, self.dtype, self.purpose) != (
                "cpu",
                "float32",
                "synthetic_forward_tests",
            ) or self.training.allow_training:
                raise ValueError("local_test requires CPU float32 synthetic tests without training")
            limits = (
                (m.temporal.dimension, 64),
                (m.depth.dimension, 64),
                (m.temporal.layers, 2),
                (m.depth.layers, 2),
                (m.temporal.context, 64),
                (m.audio_vocab_size, 64),
                (m.text_vocab_size, 128),
                (self.data.batch_size, 2),
                (r.max_queue_frames, 8),
            )
            if any(value > limit for value, limit in limits):
                raise ValueError("local_test exceeds explicit tiny resource limits")
        elif self.device != "cuda" or self.purpose != "training":
            raise ValueError("colab_train requires cuda and training purpose")
        return self


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate configuration key: {key}")
        result[key] = value
    return result


def load_config(path: str | Path) -> ExperimentConfig:
    """Read only JSON; reject duplicate keys and validate without backend imports."""
    raw = Path(path).read_text(encoding="utf-8")
    json.loads(raw, object_pairs_hook=_unique_object)
    return ExperimentConfig.model_validate_json(raw)
