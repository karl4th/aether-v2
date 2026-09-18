"""Backend-free trainer checks: no model imports, gradients, or optimizer steps."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from aether import trainer
from aether.config.adaptation import TrainingConfig


@pytest.mark.parametrize(
    "change",
    [
        {"steps": True},
        {"schema_version": True},
        {"schema_version": 1.0},
        {"steps": 0},
        {"frames": 129},
        {"rank": 0},
        {"learning_rate": float("nan")},
        {"alpha": float("inf")},
        {"seed": -1},
        {"unknown": 1},
        {"max_seconds": 7201},
    ],
)
def test_strict_bounded_configuration(change):
    with pytest.raises(ValidationError):
        TrainingConfig(**change)


def metadata():
    return {
        "schema_version": 1,
        "config": TrainingConfig().model_dump(),
        "dataset": "a" * 64,
        "source": "commit",
        "base": "weights",
        "backend": "runtime",
        "adapter": "adapter",
        "step": 5,
    }


@pytest.mark.parametrize(
    "field", ["schema_version", "config", "dataset", "source", "base", "backend", "adapter"]
)
def test_resume_rejects_every_identity_change(field):
    expected = metadata()
    changed = copy.deepcopy(expected)
    changed[field] = "different"
    with pytest.raises(ValueError, match=field):
        trainer.validate_resume(changed, expected)


@pytest.mark.parametrize("step", [True, -1, 21, 1.5, None])
def test_resume_rejects_invalid_progress(step):
    value = metadata()
    value["step"] = step
    with pytest.raises(ValueError, match="step"):
        trainer.validate_resume(value, metadata())


def test_resume_accepts_matching_identity():
    trainer.validate_resume(metadata(), metadata())


def test_checkpoint_integrity_checked_before_model_access(monkeypatch, tmp_path):
    verify = Mock(side_effect=ValueError("broken manifest"))
    monkeypatch.setattr(trainer, "verify_checkpoint", verify)
    with pytest.raises(ValueError, match="broken manifest"):
        trainer.apply_checkpoint(object(), tmp_path)


def _write_checkpoint_metadata(tmp_path, **overrides):
    payload = {
        "schema_version": 1,
        "config": TrainingConfig().model_dump(),
        "dataset": "a" * 64,
        "source": "0" * 40,
        "base": "weights",
        "backend": "runtime",
        "adapter": "adapter",
        "step": 3,
        **overrides,
    }
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "metadata.json").write_text(json.dumps(payload))
    return checkpoint_dir


def test_apply_checkpoint_for_inference_ignores_running_code_identity(monkeypatch, tmp_path):
    checkpoint_dir = _write_checkpoint_metadata(tmp_path)
    monkeypatch.setattr(trainer, "verify_checkpoint", Mock())
    # Stand in for "what the currently running code's identity would compute":
    # deliberately different from the checkpoint's own recorded source, to prove
    # apply_checkpoint does not require them to match for inference (dataset=None).
    monkeypatch.setattr(
        trainer,
        "_metadata",
        lambda dataset, config: {
            "schema_version": 1,
            "config": config.model_dump(),
            "dataset": None,
            "source": "f" * 40,
            "base": "weights",
            "backend": "runtime",
            "adapter": "adapter",
        },
    )
    monkeypatch.setattr(trainer, "install_adapter", Mock(return_value=[]))
    monkeypatch.setattr(trainer, "_load_adapter", Mock())
    fake_backend = SimpleNamespace(
        torch=SimpleNamespace(load=Mock(return_value={"adapter": {}})), lm=Mock()
    )
    metadata = trainer.apply_checkpoint(fake_backend, checkpoint_dir)
    assert metadata["source"] == "0" * 40


@pytest.mark.parametrize(
    "bad_source", ["not-hex-not-hex-not-hex-not-hex-not-hexx", "abc", 123, None]
)
def test_apply_checkpoint_rejects_malformed_source_identity(monkeypatch, tmp_path, bad_source):
    checkpoint_dir = _write_checkpoint_metadata(tmp_path, source=bad_source)
    monkeypatch.setattr(trainer, "verify_checkpoint", Mock())
    with pytest.raises(ValueError, match="invalid source revision identity"):
        trainer.apply_checkpoint(SimpleNamespace(), checkpoint_dir)


def test_local_training_denied_before_backend_access(monkeypatch, tmp_path):
    import aether.remote

    deny = Mock(side_effect=RuntimeError("remote only"))
    monkeypatch.setattr(aether.remote, "require_remote_runtime", deny)
    with pytest.raises(RuntimeError, match="remote only"):
        trainer.train(object(), tmp_path, permit_path=tmp_path / "permit", config=TrainingConfig())


def test_safe_loading_is_explicit_and_no_top_level_tensor_imports():
    tree = ast.parse(Path(trainer.__file__).read_text())
    loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
    ]
    assert loads
    assert all(
        any(
            k.arg == "weights_only" and isinstance(k.value, ast.Constant) and k.value.value is True
            for k in node.keywords
        )
        for node in loads
    )
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all("torch" not in ast.unparse(node) for node in imports)


def test_gated_projection_rejects_incompatible_architecture():
    lm = SimpleNamespace(
        requires_grad_=Mock(), transformer=SimpleNamespace(layers=[SimpleNamespace(gating=None)])
    )
    with pytest.raises(ValueError, match="gated feedforward"):
        trainer.install_adapter(lm, SimpleNamespace(), TrainingConfig())


def test_loss_selects_valid_logits_before_cross_entropy():
    from unittest.mock import MagicMock

    codes = MagicMock()
    targets = codes.__getitem__.return_value
    targets.__ge__.return_value = True
    targets.__lt__.return_value = True
    output = SimpleNamespace(logits=MagicMock(), mask=MagicMock())
    output.logits.shape = (1, 8, 16, 2048)
    mask = output.mask.__and__.return_value.__and__.return_value
    mask.sum.return_value.item.return_value = 7
    torch = MagicMock()
    torch.isfinite.return_value.item.return_value = True
    loss, count = trainer.masked_audio_loss(output, codes, torch)
    assert count == 7
    codes.__getitem__.assert_called_once_with((slice(None), slice(1, 9)))
    output.logits.__getitem__.assert_called_once_with(mask)
    targets.__getitem__.assert_called_once_with(mask)
    torch.nn.functional.cross_entropy.assert_called_once_with(
        output.logits.__getitem__.return_value.float.return_value,
        targets.__getitem__.return_value,
    )
    assert loss is torch.nn.functional.cross_entropy.return_value


def test_loss_rejects_empty_mask_without_cross_entropy():
    from unittest.mock import MagicMock

    output = SimpleNamespace(logits=MagicMock(), mask=MagicMock())
    output.logits.shape = (1, 8, 16, 2048)
    mask = output.mask.__and__.return_value.__and__.return_value
    mask.sum.return_value.item.return_value = 0
    torch = MagicMock()
    codes = MagicMock()
    codes.__getitem__.return_value.__ge__.return_value = True
    codes.__getitem__.return_value.__lt__.return_value = True
    with pytest.raises(ValueError, match="no valid"):
        trainer.masked_audio_loss(output, codes, torch)
    torch.nn.functional.cross_entropy.assert_not_called()


@pytest.mark.parametrize("value", [0, 9, True, 1.5, "2"])
def test_accumulation_strict_bounds(value):
    with pytest.raises(ValidationError):
        TrainingConfig(accumulation_steps=value)


def test_accumulation_defaults_and_upper_bound():
    assert TrainingConfig().accumulation_steps == 1
    assert TrainingConfig(accumulation_steps=8).accumulation_steps == 8


def fake_parameters(extra=False, missing=False, empty=False):
    prefix = "transformer.layers.1.gating.linear_out"
    params = {
        f"{prefix}.down.weight": SimpleNamespace(
            requires_grad=True, numel=lambda: 0 if empty else 8
        )
    }
    if not missing:
        params[f"{prefix}.up.weight"] = SimpleNamespace(requires_grad=True, numel=lambda: 8)
    params["base.weight"] = SimpleNamespace(requires_grad=extra, numel=lambda: 64)
    return SimpleNamespace(
        transformer=SimpleNamespace(layers=[None, None]), named_parameters=lambda: params.items()
    )


def test_optimizer_selection_contains_only_exact_adapter_matrices():
    selected = trainer.adapter_parameters(fake_parameters())
    assert len(selected) == 2
    assert all(name.endswith(("down.weight", "up.weight")) for name in selected)


@pytest.mark.parametrize("change", [{"extra": True}, {"missing": True}, {"empty": True}])
def test_optimizer_selection_rejects_unfrozen_base_missing_or_empty_adapter(change):
    with pytest.raises(ValueError):
        trainer.adapter_parameters(fake_parameters(**change))
