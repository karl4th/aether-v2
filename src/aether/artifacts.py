"""Read-only bundle integrity and strict synthetic parameter loading.

Real model deserialization and architecture compatibility remain unavailable.
No pickle, backend import, download, or automatic layout conversion is used.
"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Protocol, Self

from pydantic import Field, model_validator

from aether.config import (
    CodecConfig,
    ExperimentConfig,
    ModelConfig,
    PositiveInt,
    StrictConfig,
    _unique_object,
)

MAX_METADATA_BYTES = 1024 * 1024
MAX_LOCAL_BUNDLE_BYTES = 16 * 1024 * 1024


class ArtifactFile(StrictConfig):
    path: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    size_bytes: PositiveInt

    @model_validator(mode="after")
    def safe_relative_path(self) -> Self:
        path = PurePosixPath(self.path)
        if (
            not self.path
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in self.path
            or str(path) != self.path
            or self.path == "."
        ):
            raise ValueError("artifact path must be a canonical relative POSIX path")
        return self


class TensorSpec(StrictConfig):
    shape: tuple[PositiveInt, ...]
    dtype: Literal["float32", "bfloat16"]


class BundleManifest(StrictConfig):
    layout_version: Annotated[int, Field(ge=1, le=1)]
    format: Literal["synthetic_json_v1", "unimplemented_model"]
    provenance: Annotated[str, Field(min_length=1)]
    dtype: Literal["float32", "bfloat16"]
    model: ModelConfig
    codec: CodecConfig
    config: ArtifactFile
    tokenizer: ArtifactFile
    weights: ArtifactFile
    parameters: dict[str, TensorSpec]

    @model_validator(mode="after")
    def unique_files(self) -> Self:
        if len({f.path for f in (self.config, self.tokenizer, self.weights)}) != 3:
            raise ValueError("config, tokenizer and weights must be distinct files")
        if not self.parameters or any(not key for key in self.parameters):
            raise ValueError("named parameters are required")
        if any(spec.dtype != self.dtype for spec in self.parameters.values()):
            raise ValueError("parameter dtype differs from bundle dtype")
        return self


def _read_bounded(path: Path, limit: int) -> bytes:
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"file exceeds local inspection limit: {path.name}")
    return data


def _json(data: bytes) -> str:
    text = data.decode("utf-8")
    json.loads(text, object_pairs_hook=_unique_object)
    return text


def _artifact(root: Path, entry: ArtifactFile) -> bytes:
    root = root.resolve()
    path = root / entry.path
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact resolves outside bundle")
    if any(part.is_symlink() for part in (path, *path.parents) if part.is_relative_to(root)):
        raise ValueError("symlink artifacts are not accepted")
    if not path.is_file():
        raise ValueError(f"missing regular artifact: {entry.path}")
    if entry.size_bytes > MAX_LOCAL_BUNDLE_BYTES:
        raise ValueError("large artifacts require a future Colab inspection path")
    data = _read_bounded(path, entry.size_bytes)
    if len(data) != entry.size_bytes or hashlib.sha256(data).hexdigest() != entry.sha256:
        raise ValueError(f"size or SHA-256 mismatch: {entry.path}")
    return data


class SyntheticTensor(StrictConfig):
    shape: tuple[PositiveInt, ...]
    dtype: Literal["float32"]
    values: tuple[float, ...]

    @model_validator(mode="after")
    def count_values(self) -> Self:
        import math

        if math.prod(self.shape) != len(self.values):
            raise ValueError("tensor value count does not match shape")
        return self


class SyntheticWeights(StrictConfig):
    parameters: dict[str, SyntheticTensor]


class ParameterTarget(Protocol):
    """Backend adapter must implement strict, atomic installation itself."""

    def parameter_specs(self) -> Mapping[str, TensorSpec]: ...

    def install_parameters(self, parameters: Mapping[str, SyntheticTensor]) -> None: ...


def inspect_bundle(manifest_path: str | Path, expected: ExperimentConfig) -> BundleManifest:
    """Validate local integrity and contracts; success does not mean model readiness."""
    path = Path(manifest_path)
    manifest = BundleManifest.model_validate_json(_json(_read_bounded(path, MAX_METADATA_BYTES)))
    if sum(f.size_bytes for f in (manifest.config, manifest.tokenizer, manifest.weights)) > (
        MAX_LOCAL_BUNDLE_BYTES
    ):
        raise ValueError("bundle exceeds local inspection budget")
    if (manifest.model, manifest.codec, manifest.dtype) != (
        expected.model,
        expected.codec,
        expected.dtype,
    ):
        raise ValueError("bundle model/codec/dtype differs from requested configuration")
    config_bytes = _artifact(path.parent, manifest.config)
    bundled = ExperimentConfig.model_validate_json(_json(config_bytes))
    if bundled != expected:
        raise ValueError("bundle experiment configuration differs from requested configuration")
    _artifact(path.parent, manifest.tokenizer)
    weights = _artifact(path.parent, manifest.weights)
    if manifest.format == "synthetic_json_v1":
        if expected.profile != "local_test":
            raise ValueError("synthetic weights require local_test")
        parameters = SyntheticWeights.model_validate_json(_json(weights)).parameters
        specs = {name: TensorSpec(shape=t.shape, dtype=t.dtype) for name, t in parameters.items()}
        if specs != manifest.parameters:
            raise ValueError("weight names, shapes or dtypes differ from manifest")
    else:
        raise ValueError("real model inspection is not implemented; backend and weights unresolved")
    return manifest


def load_synthetic_bundle(
    manifest_path: str | Path, expected: ExperimentConfig, target: ParameterTarget
) -> None:
    manifest = inspect_bundle(manifest_path, expected)
    weights = SyntheticWeights.model_validate_json(
        _json(_artifact(Path(manifest_path).parent, manifest.weights))
    )
    if dict(target.parameter_specs()) != manifest.parameters:
        raise ValueError("target parameter names, shapes or dtypes do not match exactly")
    target.install_parameters(weights.parameters)
