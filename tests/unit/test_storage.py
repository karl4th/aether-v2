from pathlib import Path
from unittest.mock import patch

import pytest

from aether.storage import (
    create_run,
    publish_checkpoint,
    resumable_checkpoints,
    verify_checkpoint,
)


def test_publish_verify_and_no_overwrite(tmp_path: Path) -> None:
    run = create_run(tmp_path / "drive", "demo-1")
    source = tmp_path / "vm"
    source.mkdir()
    (source / "weights.fixture").write_bytes(b"synthetic values, no optimizer")
    path = publish_checkpoint(source, run, "step-1")
    assert verify_checkpoint(path).files["weights.fixture"].size == 30
    assert resumable_checkpoints(run) == [path]
    with pytest.raises(FileExistsError):
        create_run(tmp_path / "drive", "demo-1")
    with pytest.raises(FileExistsError):
        publish_checkpoint(source, run, "step-1")
    (path / "weights.fixture").write_bytes(b"corrupt")
    assert resumable_checkpoints(run) == []


def test_interrupted_copy_preserves_previous_checkpoint(tmp_path: Path) -> None:
    run = create_run(tmp_path / "drive", "demo")
    source = tmp_path / "vm"
    source.mkdir()
    (source / "data").write_bytes(b"synthetic")
    first = publish_checkpoint(source, run, "first")
    with patch("aether.storage.shutil.copyfile", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            publish_checkpoint(source, run, "interrupted")
    assert not (run / "checkpoints/interrupted/COMPLETE").exists()
    assert resumable_checkpoints(run) == [first]


def test_unsafe_names_and_symlinks(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        create_run(tmp_path, "../escape")
    source = tmp_path / "source"
    source.mkdir()
    (source / "link").symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError):
        publish_checkpoint(source, tmp_path / "run", "test")


def test_completion_manifest_and_extra_file_are_verified(tmp_path: Path) -> None:
    source = tmp_path / "vm"
    source.mkdir()
    (source / "data").write_bytes(b"fixture")
    run = create_run(tmp_path / "drive", "test")
    checkpoint = publish_checkpoint(source, run, "valid")
    (checkpoint / "extra").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="content mismatch"):
        verify_checkpoint(checkpoint)
    (checkpoint / "extra").unlink()
    (checkpoint / "COMPLETE").write_text("0" * 64)
    with pytest.raises(ValueError, match="completion hash"):
        verify_checkpoint(checkpoint)
