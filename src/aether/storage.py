"""Immutable checkpoint publication; incomplete copies are never resumable."""

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Annotated

from pydantic import Field

from aether.config import StrictConfig, _unique_object


class StoredFile(StrictConfig):
    size: Annotated[int, Field(ge=0)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class CheckpointManifest(StrictConfig):
    files: dict[str, StoredFile]


def _digest(path: Path) -> StoredFile:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return StoredFile(size=size, sha256=digest.hexdigest())


def _files(root: Path) -> dict[str, StoredFile]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("checkpoint may not contain symlinks")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in {"checkpoint.json", "COMPLETE"}:
                result[relative] = _digest(path)
        elif not path.is_dir():
            raise ValueError("checkpoint contains a nonregular file")
    return result


def _name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("identifier must use 1..80 ASCII letters, digits, underscores or hyphens")
    return value


def create_run(root: Path, run_id: str) -> Path:
    """Never overwrite an existing run. Root may be local or mounted Drive."""
    run = root / "runs" / _name(run_id)
    run.mkdir(parents=True, exist_ok=False)
    for name in ("datasets", "bundles", "exports"):
        (root / name).mkdir(exist_ok=True)
    return run


def verify_checkpoint(path: Path) -> CheckpointManifest:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("checkpoint must be a regular directory")
    marker = path / "COMPLETE"
    manifest_path = path / "checkpoint.json"
    if marker.is_symlink() or manifest_path.is_symlink():
        raise ValueError("checkpoint metadata may not be symlinks")
    if not marker.is_file():
        raise ValueError("checkpoint is incomplete")
    raw = manifest_path.read_text(encoding="utf-8")
    if marker.read_text(encoding="ascii") != hashlib.sha256(raw.encode()).hexdigest():
        raise ValueError("checkpoint completion hash mismatch")
    json.loads(raw, object_pairs_hook=_unique_object)
    manifest = CheckpointManifest.model_validate_json(raw)
    if not manifest.files or _files(path) != manifest.files:
        raise ValueError("checkpoint content mismatch")
    return manifest


def publish_checkpoint(source: Path, run: Path, checkpoint_id: str) -> Path:
    """Copy to a fresh destination, verify, then publish a completion marker.

    Any exception leaves an incomplete directory; resume ignores it. Reusing its
    name is rejected so the previous confirmed checkpoint is never overwritten.
    """
    if source.is_symlink() or not source.is_dir():
        raise ValueError("source must be a regular directory")
    if (source / "checkpoint.json").exists() or (source / "COMPLETE").exists():
        raise ValueError("source contains reserved metadata names")
    entries = _files(source)
    if not entries:
        raise ValueError("checkpoint source is empty")
    target = run / "checkpoints" / _name(checkpoint_id)
    if target.resolve().is_relative_to(source.resolve()):
        raise ValueError("destination cannot be inside checkpoint source")
    target.mkdir(parents=True, exist_ok=False)
    for name in entries:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
        with destination.open("rb") as handle:
            os.fsync(handle.fileno())
    if _files(target) != entries or _files(source) != entries:
        raise ValueError("checkpoint changed or copy verification failed")
    raw = CheckpointManifest(files=entries).model_dump_json()
    with (target / "checkpoint.json").open("x", encoding="utf-8") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    with (target / "COMPLETE").open("x", encoding="ascii") as handle:
        handle.write(hashlib.sha256(raw.encode()).hexdigest())
        handle.flush()
        os.fsync(handle.fileno())
    verify_checkpoint(target)
    return target


def resumable_checkpoints(run: Path) -> list[Path]:
    """Return verified paths only; do not guess which training step is newest."""
    valid = []
    for path in sorted((run / "checkpoints").glob("*")):
        try:
            verify_checkpoint(path)
        except (OSError, ValueError):
            continue
        valid.append(path)
    return valid
