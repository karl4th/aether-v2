"""Verify installed entry points and fail-closed scaffolding in separate processes."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "aether", *args],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_installed_entry_point() -> None:
    executable = Path(sys.executable).parent / ("aether.exe" if os.name == "nt" else "aether")
    result = subprocess.run(
        [str(executable), "--version"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "aether 0.1.0"


def test_help_does_not_require_model_dependencies() -> None:
    result = run_cli("--help")
    assert result.returncode == 0
    assert "Colab" in result.stdout
    assert "inspect" in result.stdout


def test_local_inference_refuses_before_backend_or_output(tmp_path: Path) -> None:
    output = tmp_path / "answer.wav"
    result = run_cli(
        "infer", "--config", "absent.json", "--input", "absent.wav", "--output", str(output)
    )
    assert result.returncode == 4
    assert "REMOTE_RUNTIME_REQUIRED" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("validate_only", [False, True])
def test_training_never_starts(validate_only: bool) -> None:
    args = ["train", "--config", "absent.json"]
    if validate_only:
        args.append("--validate-only")
    result = run_cli(*args)
    assert result.returncode == (2 if validate_only else 4)
    expected = "INVALID_CONFIG" if validate_only else "TRAINING_ENVIRONMENT_REQUIRED"
    assert expected in result.stderr


def test_missing_arguments_fail() -> None:
    result = run_cli("infer")
    assert result.returncode == 2


@pytest.mark.parametrize(
    "config_path", ["configs/model/tiny.json", "configs/training/colab_lora.json"]
)
def test_validation_is_backend_free_and_read_only(tmp_path: Path, config_path: str) -> None:
    config = Path(config_path).resolve()
    code = """
import sys
from pathlib import Path

def audit(event, args):
    if event == 'import' and (
        args[0].split('.')[0] in {'torch', 'transformers'}
        or args[0].startswith('aether.trainer')
    ):
        raise AssertionError('backend import')
    if event.startswith('socket.') or event in {'os.mkdir', 'os.remove', 'os.rename'}:
        raise AssertionError(event)
    if event == 'open':
        mode, flags = args[1], args[2]
        if (isinstance(mode, str) and any(x in mode for x in 'wax+')) or flags & 3:
            raise AssertionError('write')
sys.addaudithook(audit)
from aether.cli import main
raise SystemExit(main(['train', '--config', sys.argv[1], '--validate-only']))
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(config)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "VALID_CONFIG" in result.stdout
    assert list(tmp_path.iterdir()) == []
