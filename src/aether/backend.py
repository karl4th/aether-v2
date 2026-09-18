"""Lazy, pinned remote speech backend. Importing this module never loads weights."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODEL_REPOSITORY = "kyutai/moshiko-pytorch-bf16"
MODEL_REVISION = "2bfc9ae6e89079a5cc7ed2a68436010d91a3d289"
BACKEND_REVISION = "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362"
SAMPLE_RATE = 24000
FRAME_SIZE = 1920


def check_resources(torch: Any, *, training: bool) -> None:
    """Conservative admission limits, not a promise that every sequence fits."""
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; CPU execution is disabled")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 GPU required: select L4/A100; T4 is unsupported")
    free, total = torch.cuda.mem_get_info()
    minimum = (38 if training else 22) * 1024**3
    if free < minimum:
        raise RuntimeError(
            f"Insufficient free VRAM: {free / 1024**3:.1f} GiB free / "
            f"{total / 1024**3:.1f} GiB total; requires {minimum / 1024**3:.0f} GiB. "
            "Select A100 for adaptation or L4/A100 for inference and restart the runtime."
        )


@dataclass
class Backend:
    """Loaded components; tensors remain on the selected remote CUDA device."""

    lm: Any
    codec: Any
    tokenizer: Any
    torch: Any
    device: str = "cuda"
    artifacts: dict[str, Any] = field(default_factory=dict)

    def encode(self, pcm: Any) -> Any:
        """Encode [B,1,N] outside a streaming context, without gradients."""
        with self.torch.inference_mode():
            return self.codec.encode(pcm.to(self.device))


def load_backend(permit_path: str | Path, *, training: bool = False, context: int = 256) -> Backend:
    if type(context) is not int or not 32 <= context <= 512:
        raise ValueError("context must be an integer between 32 and 512 frames")
    remote = importlib.import_module("aether.remote")
    remote.require_remote_runtime(training=training, permit_path=permit_path)
    torch = importlib.import_module("torch")
    check_resources(torch, training=training)
    loaders = importlib.import_module("moshi.models.loaders")
    info = loaders.CheckpointInfo.from_hf_repo(MODEL_REPOSITORY, revision=MODEL_REVISION)
    artifacts: dict[str, Any] = {
        "model_revision": MODEL_REVISION,
        "backend_revision": BACKEND_REVISION,
        "context": context,
        "files": {},
    }
    for label, path in (
        ("model", info.moshi_weights),
        ("codec", info.mimi_weights),
        ("tokenizer", info.tokenizer),
    ):
        path = Path(path)
        if label != "tokenizer" and path.suffix != ".safetensors":
            raise ValueError("Only safetensors base weights are permitted")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        artifacts["files"][label] = {"sha256": digest, "bytes": path.stat().st_size}
    if info.lora_weights is not None:
        raise ValueError("Unexpected adapter in base checkpoint")
    codec = info.get_mimi(device="cuda")
    codec.requires_grad_(False)
    codec.eval()
    lm = info.get_moshi(
        device="cuda",
        dtype=torch.bfloat16,
        lm_kwargs_overrides={"context": context, "gradient_checkpointing": training},
    )
    lm.eval()
    return Backend(lm, codec, info.get_text_tokenizer(), torch, artifacts=artifacts)


def stream_generate(backend: Backend, pcm: Any, *, seed: int = 42) -> tuple[Any, str]:
    """Generate one session, isolating and releasing codec and LM streaming state."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    torch = backend.torch
    models = importlib.import_module("moshi.models")
    torch.manual_seed(seed)
    generator = models.LMGen(backend.lm, use_sampling=True, temp=0.8, temp_text=0.7)
    samples: list[Any] = []
    text_ids: list[int] = []
    if pcm.ndim != 3 or tuple(pcm.shape[:2]) != (1, 1):
        raise ValueError("PCM must have shape [1,1,samples]")
    if pcm.shape[-1] < FRAME_SIZE or pcm.shape[-1] % FRAME_SIZE:
        raise ValueError("PCM must contain complete frames")
    with torch.inference_mode(), backend.codec.streaming(1), generator.streaming(1):
        first = True
        for chunk in pcm.split(FRAME_SIZE, dim=-1):
            codes = backend.codec.encode(chunk.to(backend.device))
            if first:
                # The first step consumes BOS; repeat input to retain the first physical frame.
                generator.step(codes)
                first = False
            tokens = generator.step(codes)
            if tokens is None:
                continue
            if tokens.shape[1] != 9:
                raise RuntimeError("Incompatible output channel contract")
            samples.append(backend.codec.decode(tokens[:, 1:9]).cpu())
            token = int(tokens[0, 0, 0].item())
            if token not in (0, 3):
                text_ids.append(token)
    if not samples:
        raise RuntimeError("The model emitted no audio frames")
    return torch.cat(samples, dim=-1), str(backend.tokenizer.decode(text_ids))


def infer_file(
    input_path: str | Path,
    output_dir: str | Path,
    permit_path: str | Path,
    *,
    tail_seconds: float = 8.0,
    context: int = 256,
    backend: Backend | None = None,
    seed: int = 42,
    adapter_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Read mono/stereo WAV, resample, append listening silence, and save reply."""
    if not math.isfinite(tail_seconds) or not 0 <= tail_seconds <= 30:
        raise ValueError("tail_seconds must be between 0 and 30")
    remote = importlib.import_module("aether.remote")
    permit = remote.require_remote_runtime(training=False, permit_path=permit_path)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Inference output directory already exists; choose a new run path")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    np = importlib.import_module("numpy")
    sf = importlib.import_module("soundfile")
    sphn = importlib.import_module("sphn")
    pcm, rate = sf.read(str(input_path), dtype="float32", always_2d=True)
    if not 0 < len(pcm) / rate <= 120 or not np.isfinite(pcm).all():
        raise ValueError("Input must contain finite audio of 0–120 seconds")
    pcm = pcm.mean(axis=1)
    if rate != SAMPLE_RATE:
        pcm = sphn.resample(pcm, rate, SAMPLE_RATE)
    tail = int(tail_seconds * SAMPLE_RATE)
    output_samples = len(pcm) + tail
    padding = (-output_samples) % FRAME_SIZE
    pcm = np.pad(pcm, (0, tail + padding))
    backend = backend or load_backend(permit_path, context=context)
    adapter_metadata = None
    if adapter_checkpoint is not None:
        trainer = importlib.import_module("aether.trainer")
        adapter_metadata = trainer.apply_checkpoint(backend, Path(adapter_checkpoint))
    tensor = backend.torch.from_numpy(pcm).reshape(1, 1, -1)
    audio, text = stream_generate(backend, tensor, seed=seed)
    audio = audio[..., :output_samples]
    output.mkdir(parents=True, exist_ok=False)
    sf.write(str(output / "reply.wav"), audio[0, 0].numpy(), SAMPLE_RATE)
    (output / "reply.txt").write_text(text, encoding="utf-8")
    result = {
        "audio": str(output / "reply.wav"),
        "text": text,
        "sample_rate": SAMPLE_RATE,
        "seed": seed,
        "artifacts": backend.artifacts,
        "adapter": adapter_metadata,
        "context": context,
        "tail_seconds": tail_seconds,
        "source_revision": permit.source_revision,
    }
    (output / "inference.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
