import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from aether.artifacts import inspect_bundle, load_synthetic_bundle
from aether.config import load_config


def make_bundle(root: Path) -> Path:
    config = load_config("configs/model/tiny.json")
    files = {
        "config": ("config.json", config.model_dump_json().encode()),
        "tokenizer": ("tokenizer.json", b'{"kind":"synthetic"}'),
        "weights": (
            "weights.json",
            json.dumps(
                {
                    "parameters": {
                        "layer.weight": {
                            "shape": [2, 2],
                            "dtype": "float32",
                            "values": [1.0, 2.0, 3.0, 4.0],
                        }
                    }
                }
            ).encode(),
        ),
    }
    manifest = {
        "schema_version": 1,
        "layout_version": 1,
        "format": "synthetic_json_v1",
        "provenance": "local generated test fixture; no model weights",
        "dtype": "float32",
        "model": config.model.model_dump(mode="json"),
        "codec": config.codec.model_dump(mode="json"),
        "parameters": {"layer.weight": {"shape": [2, 2], "dtype": "float32"}},
    }
    for role, (name, data) in files.items():
        (root / name).write_bytes(data)
        manifest[role] = {
            "path": name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_exact_parameter_loading(tmp_path: Path) -> None:
    path = make_bundle(tmp_path)
    config = load_config("configs/model/tiny.json")
    manifest = inspect_bundle(path, config)
    target = Mock()
    target.parameter_specs.return_value = manifest.parameters
    load_synthetic_bundle(path, config, target)
    target.install_parameters.assert_called_once()
    values = target.install_parameters.call_args.args[0]["layer.weight"].values
    assert values == (1.0, 2.0, 3.0, 4.0)
    target.reset_mock()
    target.parameter_specs.return_value = {}
    with pytest.raises(ValueError, match="target parameter"):
        load_synthetic_bundle(path, config, target)
    target.install_parameters.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "corrupt",
        "truncated",
        "shape",
        "extra",
        "version",
        "codec",
        "escape",
        "symlink",
        "duplicate",
    ],
)
def test_invalid_bundle_rejected_before_target(tmp_path: Path, change: str) -> None:
    path = make_bundle(tmp_path)
    manifest = json.loads(path.read_text())
    weights = tmp_path / "weights.json"
    if change == "missing":
        weights.unlink()
    elif change == "corrupt":
        weights.write_bytes(weights.read_bytes().replace(b"1.0", b"9.0"))
    elif change == "truncated":
        weights.write_bytes(b"{}")
    elif change == "shape":
        manifest["parameters"]["layer.weight"]["shape"] = [4]
    elif change == "extra":
        manifest["parameters"]["unexpected"] = {"shape": [1], "dtype": "float32"}
    elif change == "version":
        manifest["layout_version"] = 2
    elif change == "codec":
        manifest["codec"]["vocabulary_size"] = 31
    elif change == "escape":
        manifest["weights"]["path"] = "../weights.json"
    elif change == "symlink":
        weights.rename(tmp_path / "real.json")
        weights.symlink_to(tmp_path / "real.json")
    path.write_text(json.dumps(manifest))
    if change == "duplicate":
        path.write_text(
            path.read_text().replace(
                '"layout_version": 1', '"layout_version": 1, "layout_version": 1'
            )
        )
    target = Mock()
    with pytest.raises((ValueError, OSError)):
        load_synthetic_bundle(path, load_config("configs/model/tiny.json"), target)
    assert target.mock_calls == []


@pytest.mark.parametrize("kind", ["nonfinite", "value_count", "dtype", "backend"])
def test_rehashed_invalid_content_is_not_trusted(tmp_path: Path, kind: str) -> None:
    path = make_bundle(tmp_path)
    manifest = json.loads(path.read_text())
    weights_path = tmp_path / "weights.json"
    weights = json.loads(weights_path.read_text())
    tensor = weights["parameters"]["layer.weight"]
    if kind == "nonfinite":
        tensor["values"][0] = float("nan")
    elif kind == "value_count":
        tensor["values"].pop()
    elif kind == "dtype":
        tensor["dtype"] = "bfloat16"
    else:
        manifest["format"] = "unimplemented_model"
    data = json.dumps(weights).encode()
    weights_path.write_bytes(data)
    manifest["weights"].update(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        inspect_bundle(path, load_config("configs/model/tiny.json"))


def test_inspect_cli_reports_scope_and_corruption(tmp_path: Path) -> None:
    import subprocess
    import sys

    path = make_bundle(tmp_path)
    command = [
        sys.executable,
        "-B",
        "-m",
        "aether",
        "inspect",
        "--config",
        "configs/model/tiny.json",
        "--manifest",
        str(path),
    ]
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "NOT_VERIFIED" in result.stdout
    assert "PASS" in result.stdout
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    (tmp_path / "weights.json").write_text("corrupt")
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    assert result.returncode == 2
    assert "INVALID_BUNDLE" in result.stderr
