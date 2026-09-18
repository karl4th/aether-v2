"""Strict bounded remote audio adaptation settings."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TrainingConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False, frozen=True)
    task: Literal["speech_adaptation"] = "speech_adaptation"
    max_seconds: int = Field(default=1800, ge=60, le=7200)
    schema_version: Literal[1] = 1
    objective: Literal["read_speech_audio"] = "read_speech_audio"
    steps: int = Field(default=20, ge=1, le=1000)
    accumulation_steps: int = Field(default=1, ge=1, le=8)
    frames: int = Field(default=64, ge=16, le=128)
    rank: int = Field(default=8, ge=1, le=32)
    alpha: float = Field(default=8.0, gt=0, le=64)
    learning_rate: float = Field(default=0.0001, gt=0, le=0.001)
    clip_norm: float = Field(default=1.0, gt=0, le=10)
    save_every: int = Field(default=5, ge=1, le=100)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)

    @field_validator("schema_version", mode="before")
    @classmethod
    def strict_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value
