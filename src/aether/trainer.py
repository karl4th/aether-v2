"""Remote-only, bounded audio adaptation with a frozen base and initial codec.

The dataset supplies read speech without word timestamps. Text is therefore
masked, not artificially aligned. This is audio adaptation, not dialogue training.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from aether.config.adaptation import TrainingConfig as TrainingConfig
from aether.storage import publish_checkpoint, verify_checkpoint


def source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[2]
    ).strip()


def identity_digest(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        value = asdict(value)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def validate_resume(metadata: dict[str, Any], expected: dict[str, Any]) -> None:
    for name in ("schema_version", "config", "dataset", "source", "base", "backend", "adapter"):
        if metadata.get(name) != expected.get(name):
            raise ValueError(f"checkpoint {name} does not match this run")
    step = metadata.get("step")
    if type(step) is not int or not 0 <= step <= expected["config"]["steps"]:
        raise ValueError("invalid checkpoint step")


def install_adapter(lm: Any, torch: Any, config: TrainingConfig) -> list[Any]:
    """Adapt final temporal feedforward projection only; all original weights freeze."""
    lm.requires_grad_(False)
    layer = lm.transformer.layers[-1]
    gating = layer.gating
    if gating is None or not isinstance(gating.linear_in, torch.nn.Linear):
        raise ValueError("unsupported final gated feedforward input")
    base = gating.linear_out
    if not isinstance(base, torch.nn.Linear):
        raise ValueError("unsupported final feedforward projection")

    class Adapter(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.base = base
            self.down = torch.nn.Linear(
                base.in_features,
                config.rank,
                bias=False,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
            self.up = torch.nn.Linear(
                config.rank,
                base.out_features,
                bias=False,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
            torch.nn.init.normal_(self.down.weight, std=0.01)
            torch.nn.init.zeros_(self.up.weight)
            self.scale = config.alpha / config.rank

        def forward(self, value: Any) -> Any:
            return self.base(value) + self.up(self.down(value)) * self.scale

    class GatedAdapter(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.linear_in = gating.linear_in
            self.linear_out = Adapter()
            self.activation = gating.activation

        def forward(self, value: Any) -> Any:
            # The upstream fused gating reads .weight directly. Execute its generic
            # algebra explicitly so the low-rank projection participates in forward.
            value = self.linear_in(value)
            value = value.reshape(*value.shape[:-1], 2, -1)
            value = self.activation(value[..., 0, :]) * value[..., 1, :]
            return self.linear_out(value)

    layer.gating = GatedAdapter()
    return list(adapter_parameters(lm).values())


def adapter_parameters(lm: Any) -> dict[str, Any]:
    """Fail closed if any original parameter is trainable or an adapter is missing."""
    prefix = f"transformer.layers.{len(lm.transformer.layers) - 1}.gating.linear_out"
    expected = {f"{prefix}.down.weight", f"{prefix}.up.weight"}
    selected = {
        name: parameter for name, parameter in lm.named_parameters() if parameter.requires_grad
    }
    if selected.keys() != expected:
        raise ValueError("only the two final gated adapter matrices may be trainable")
    if any(parameter.numel() <= 0 for parameter in selected.values()):
        raise ValueError("adapter parameters must be nonempty")
    return selected


def _adapter_state(lm: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in lm.named_parameters()
        if parameter.requires_grad
    }


def _load_adapter(lm: Any, state: dict[str, Any]) -> None:
    parameters = {name: p for name, p in lm.named_parameters() if p.requires_grad}
    if parameters.keys() != state.keys():
        raise ValueError("adapter parameter names mismatch")
    for name, parameter in parameters.items():
        if not hasattr(state[name], "shape") or parameter.shape != state[name].shape:
            raise ValueError("adapter parameter shape mismatch")
        if parameter.dtype != state[name].dtype:
            raise ValueError("adapter parameter dtype mismatch")
        parameter.data.copy_(state[name].to(parameter))


def _codes(backend: Any, example: Any, frames: int) -> Any:
    import importlib

    sf = importlib.import_module("soundfile")
    audio, rate = sf.read(str(example.audio_path), dtype="float32", always_2d=True)
    torch = backend.torch
    waveform = torch.from_numpy(audio.mean(axis=1)).to(backend.device)[None, None]
    if rate != backend.codec.sample_rate:
        audio_utils = importlib.import_module("torchaudio.functional")
        waveform = audio_utils.resample(waveform, rate, backend.codec.sample_rate)
    frame_size = int(backend.codec.sample_rate / backend.codec.frame_rate)
    samples = frames * frame_size
    waveform = waveform[..., :samples]
    if waveform.shape[-1] < frame_size * 16:
        raise ValueError("utterance is too short for adaptation")
    # Whole codec frames only; no padded examples contribute loss.
    waveform = waveform[..., : waveform.shape[-1] // frame_size * frame_size]
    with torch.no_grad():
        speech = backend.codec.encode(waveform)
        silence = backend.codec.encode(torch.zeros_like(waveform))
    if speech.shape[1] != 8 or speech.shape != silence.shape:
        raise ValueError("codec does not satisfy the eight-book stream contract")
    text = torch.full_like(speech[:, :1], -1)
    return torch.cat([text, speech, silence], dim=1)


def masked_audio_loss(output: Any, codes: Any, torch: Any) -> tuple[Any, int]:
    """LMOutput already reverses delays and aligns targets; do not shift again."""
    targets = codes[:, 1:9]
    mask = output.mask & (targets >= 0) & (targets < output.logits.shape[-1])
    count = int(mask.sum().item())
    if count == 0:
        raise ValueError("no valid audio targets")
    # Select before CE: invalid delayed logits are NaN in the upstream contract.
    loss = torch.nn.functional.cross_entropy(output.logits[mask].float(), targets[mask])
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("non-finite audio loss")
    return loss, count


def _metadata(dataset: Any, config: TrainingConfig) -> dict[str, Any]:
    from aether.backend import BACKEND_REVISION, MODEL_REVISION

    return {
        "schema_version": 1,
        "config": config.model_dump(),
        "dataset": dataset.sha256 if dataset is not None else None,
        "source": source_revision(),
        "base": MODEL_REVISION,
        "backend": BACKEND_REVISION,
        "adapter": "last_temporal_gated_out_lora_v1",
    }


def apply_checkpoint(
    backend: Any, checkpoint: Path, config: TrainingConfig | None = None, dataset: Any = None
) -> dict[str, Any]:
    """Verify provenance before safe tensor loading, then attach the saved adapter."""
    checkpoint = Path(checkpoint)
    verify_checkpoint(checkpoint)
    metadata: dict[str, Any] = json.loads((checkpoint / "metadata.json").read_text())
    saved_config = TrainingConfig.model_validate(metadata.get("config"))
    config = config or saved_config
    expected = _metadata(dataset, config)
    if dataset is None:
        # Inference need not have the training corpus mounted.
        expected["dataset"] = metadata.get("dataset")
        if not isinstance(expected["dataset"], str) or len(expected["dataset"]) != 64:
            raise ValueError("invalid dataset identity")
    validate_resume(metadata, expected)
    state = backend.torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
    install_adapter(backend.lm, backend.torch, config)
    _load_adapter(backend.lm, state["adapter"])
    backend.lm.eval()
    return metadata


def _save(
    backend: Any,
    optimizer: Any,
    scheduler: Any,
    metadata: dict[str, Any],
    output_dir: Path,
    step: int,
    history: list[dict[str, Any]],
) -> Path:
    torch = backend.torch
    metadata = {**metadata, "step": step}
    with tempfile.TemporaryDirectory(prefix="aether-checkpoint-") as temporary:
        source = Path(temporary)
        torch.save(
            {
                "adapter": _adapter_state(backend.lm),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "python_rng": random.getstate(),
                "history": history,
            },
            source / "state.pt",
        )
        (source / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        from aether.dataset import ATTRIBUTION, LICENSE_URL, SOURCE_URL

        notices = Path(__file__).resolve().parents[2] / "THIRD_PARTY_NOTICES.md"
        (source / "THIRD_PARTY_NOTICES.md").write_bytes(notices.read_bytes())
        (source / "DATA_ATTRIBUTION.txt").write_text(
            f"{ATTRIBUTION}\n{SOURCE_URL}\n{LICENSE_URL}\n"
        )
        return publish_checkpoint(source, output_dir, f"step-{step:06d}")


def train(
    dataset: Any,
    output_dir: Path,
    *,
    permit_path: Path,
    config: TrainingConfig,
    resume: Path | None = None,
    stop_after_steps: int | None = None,
) -> dict[str, Any]:
    from aether.remote import require_remote_runtime

    require_remote_runtime(training=True, permit_path=permit_path)
    from aether.backend import load_backend

    if stop_after_steps is not None and (
        type(stop_after_steps) is not int or not 1 <= stop_after_steps <= config.steps
    ):
        raise ValueError("stop_after_steps must be within the planned step count")
    if not dataset.train:
        raise ValueError("training split is empty")
    metadata = _metadata(dataset, config)
    resume_metadata = None
    if resume is not None:
        verify_checkpoint(resume)
        resume_metadata = json.loads((resume / "metadata.json").read_text())
        validate_resume(resume_metadata, metadata)
    backend = load_backend(
        permit_path=permit_path, training=True, context=max(32, config.frames + 2)
    )
    torch = backend.torch
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    parameters = install_adapter(backend.lm, torch, config)
    selected = adapter_parameters(backend.lm)
    trainable = {
        "names": sorted(selected),
        "parameters": sum(parameter.numel() for parameter in selected.values()),
    }
    metadata = {**metadata, "trainable": trainable}
    backend.codec.eval().requires_grad_(False)
    backend.lm.train()
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, 1.0 - step / config.steps)
    )
    step = 0
    history: list[dict[str, Any]] = []
    if resume is not None:
        state = torch.load(resume / "state.pt", map_location="cpu", weights_only=True)
        _load_adapter(backend.lm, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        history = state["history"]
        assert resume_metadata is not None
        step = resume_metadata["step"]
    checkpoint = resume
    started = time.monotonic()
    stop_step = min(config.steps, step + (stop_after_steps or config.steps))
    while step < stop_step:
        require_remote_runtime(training=True, permit_path=permit_path)
        if time.monotonic() - started >= config.max_seconds:
            if checkpoint is None or not str(checkpoint).endswith(f"step-{step:06d}"):
                checkpoint = _save(
                    backend, optimizer, scheduler, metadata, output_dir, step, history
                )
            break
        # Each optimizer step averages utterance losses, with deterministic order.
        step_started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), random.getstate())
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        token_count = 0
        interrupted = False
        for microbatch in range(config.accumulation_steps):
            require_remote_runtime(training=True, permit_path=permit_path)
            if time.monotonic() - started >= config.max_seconds:
                interrupted = True
                break
            index = step * config.accumulation_steps + microbatch
            example = dataset.train[index % len(dataset.train)]
            codes = _codes(backend, example, config.frames)
            loss, count = masked_audio_loss(backend.lm(codes), codes, torch)
            (loss / config.accumulation_steps).backward()
            losses.append(float(loss.detach().item()))
            token_count += count
        if interrupted:
            # Partial gradients are discarded. Restore RNG to this step boundary
            # so a resumed invocation repeats the same complete microbatches.
            optimizer.zero_grad(set_to_none=True)
            torch.set_rng_state(rng[0])
            torch.cuda.set_rng_state_all(rng[1])
            random.setstate(rng[2])
            if checkpoint is None or not str(checkpoint).endswith(f"step-{step:06d}"):
                checkpoint = _save(
                    backend, optimizer, scheduler, metadata, output_dir, step, history
                )
            break
        norm = torch.nn.utils.clip_grad_norm_(parameters, config.clip_norm, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        step += 1
        history.append(
            {
                "step": step,
                "audio_ce": sum(losses) / len(losses),
                "tokens": token_count,
                "microbatches": config.accumulation_steps,
                "gradient_norm": float(norm.item()),
                "elapsed_seconds": time.monotonic() - step_started,
                "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            }
        )
        if step % config.save_every == 0 or step == stop_step:
            checkpoint = _save(backend, optimizer, scheduler, metadata, output_dir, step, history)
    return {
        "step": step,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "history": history,
        "trainable": trainable,
        "elapsed_seconds": time.monotonic() - started,
        "completed": step == config.steps,
        "objective": config.objective,
        "text_loss": "masked: no timestamp alignment",
    }


def evaluate(
    dataset: Any, *, permit_path: Path, config: TrainingConfig, checkpoint: Path | None = None
) -> dict[str, Any]:
    from aether.remote import require_remote_runtime

    require_remote_runtime(training=False, permit_path=permit_path)
    from aether.backend import load_backend

    if not dataset.evaluation:
        raise ValueError("evaluation split is empty")
    if checkpoint is not None:
        verify_checkpoint(checkpoint)
        metadata = json.loads((checkpoint / "metadata.json").read_text())
        validate_resume(metadata, _metadata(dataset, config))
    backend = load_backend(
        permit_path=permit_path, training=False, context=max(32, config.frames + 2)
    )
    torch = backend.torch
    if checkpoint is not None:
        apply_checkpoint(backend, checkpoint, config, dataset)
    backend.lm.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for example in dataset.evaluation:
            codes = _codes(backend, example, config.frames)
            loss, count = masked_audio_loss(backend.lm(codes), codes, torch)
            total += float(loss.item()) * count
            tokens += count
    ce = total / tokens
    return {
        "audio_ce": ce,
        "audio_perplexity": math.exp(min(ce, 80)),
        "tokens": tokens,
        "examples": len(dataset.evaluation),
        "checkpoint": str(checkpoint),
        "objective": config.objective,
        "limitation": "read-speech audio loss does not measure conversation quality",
    }
