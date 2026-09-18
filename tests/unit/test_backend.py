"""Backend admission tests never import torch or download model artifacts."""

import importlib
from unittest.mock import Mock

import pytest

from aether.backend import MODEL_REVISION, check_resources, load_backend


def test_gate_precedes_every_backend_import(monkeypatch):
    remote = Mock()
    remote.require_remote_runtime.side_effect = RuntimeError("denied")
    imports = []

    def fake_import(name):
        imports.append(name)
        assert name == "aether.remote"
        return remote

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(RuntimeError, match="denied"):
        load_backend("permit.json")
    assert imports == ["aether.remote"]


@pytest.mark.parametrize("context", [True, 31, 513, 128.5])
def test_bad_context_rejected_before_runtime(context):
    with pytest.raises(ValueError, match="context"):
        load_backend("absent", context=context)


@pytest.mark.parametrize("training,free", [(False, 21), (True, 37)])
def test_memory_checked_before_download(training, free):
    torch = Mock()
    torch.cuda.mem_get_info.return_value = (free * 1024**3, 80 * 1024**3)
    with pytest.raises(RuntimeError, match="VRAM"):
        check_resources(torch, training=training)


def test_t4_rejected():
    torch = Mock()
    torch.cuda.is_bf16_supported.return_value = False
    with pytest.raises(RuntimeError, match="T4"):
        check_resources(torch, training=False)
    torch.cuda.mem_get_info.assert_not_called()


def test_pinned_load_and_checkpointing(monkeypatch, tmp_path):
    remote, torch, loaders = Mock(), Mock(), Mock()
    torch.cuda.mem_get_info.return_value = (40 * 1024**3, 40 * 1024**3)
    modules = {"aether.remote": remote, "torch": torch, "moshi.models.loaders": loaders}
    monkeypatch.setattr(importlib, "import_module", modules.__getitem__)
    info = loaders.CheckpointInfo.from_hf_repo.return_value
    info.moshi_weights = tmp_path / "model.safetensors"
    info.mimi_weights = tmp_path / "codec.safetensors"
    info.tokenizer = tmp_path / "tokenizer.model"
    info.lora_weights = None
    for path in (info.moshi_weights, info.mimi_weights, info.tokenizer):
        path.write_bytes(b"fixture only")
    backend = load_backend("permit.json", training=True, context=128)
    assert len(backend.artifacts["files"]["model"]["sha256"]) == 64
    remote.require_remote_runtime.assert_called_once_with(training=True, permit_path="permit.json")
    assert loaders.CheckpointInfo.from_hf_repo.call_args.kwargs["revision"] == MODEL_REVISION
    info = loaders.CheckpointInfo.from_hf_repo.return_value
    info.get_moshi.assert_called_once_with(
        device="cuda",
        dtype=torch.bfloat16,
        lm_kwargs_overrides={"context": 128, "gradient_checkpointing": True},
    )
    assert backend.lm is info.get_moshi.return_value
    backend.codec.requires_grad_.assert_called_once_with(False)


def test_streaming_retains_first_frame_and_releases_session(monkeypatch):
    from contextlib import nullcontext
    from unittest.mock import MagicMock

    from aether.backend import Backend, stream_generate

    torch, codec, lm, tokenizer, models = (Mock() for _ in range(5))
    torch.inference_mode.side_effect = nullcontext
    codec.streaming.side_effect = nullcontext
    generator = models.LMGen.return_value
    generator.streaming.side_effect = nullcontext
    tokens = MagicMock()
    tokens.shape = (1, 9, 1)
    tokens.__getitem__.return_value.item.return_value = 23
    generator.step.side_effect = [None, tokens, tokens]
    pcm = Mock(ndim=3, shape=(1, 1, 3840))
    chunks = [Mock(), Mock()]
    pcm.split.return_value = chunks
    monkeypatch.setattr(importlib, "import_module", lambda name: models)
    backend = Backend(lm, codec, tokenizer, torch)
    stream_generate(backend, pcm)
    assert generator.step.call_count == 3
    assert codec.encode.call_count == 2
    assert codec.decode.call_count == 2
    tokenizer.decode.assert_called_once_with([23, 23])
    codec.streaming.assert_called_once_with(1)
    generator.streaming.assert_called_once_with(1)


def test_empty_generation_fails_without_output(monkeypatch):
    from contextlib import nullcontext

    from aether.backend import Backend, stream_generate

    torch, codec, models = Mock(), Mock(), Mock()
    torch.inference_mode.side_effect = nullcontext
    codec.streaming.side_effect = nullcontext
    models.LMGen.return_value.streaming.side_effect = nullcontext
    models.LMGen.return_value.step.return_value = None
    pcm = Mock(ndim=3, shape=(1, 1, 1920))
    pcm.split.return_value = [Mock()]
    monkeypatch.setattr(importlib, "import_module", lambda name: models)
    with pytest.raises(RuntimeError, match="no audio"):
        stream_generate(Backend(Mock(), codec, Mock(), torch), pcm)


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5])
def test_generation_seed_validated_before_import(seed):
    from aether.backend import Backend, stream_generate

    with pytest.raises(ValueError, match="seed"):
        stream_generate(Backend(None, None, None, None), None, seed=seed)


def test_memory_rejection_precedes_checkpoint_download(monkeypatch):
    remote, torch, loaders = Mock(), Mock(), Mock()
    torch.cuda.mem_get_info.return_value = (10 * 1024**3, 24 * 1024**3)
    modules = {"aether.remote": remote, "torch": torch, "moshi.models.loaders": loaders}
    monkeypatch.setattr(importlib, "import_module", modules.__getitem__)
    with pytest.raises(RuntimeError, match="VRAM"):
        load_backend("permit.json")
    loaders.CheckpointInfo.from_hf_repo.assert_not_called()


def test_pickle_base_weights_rejected_before_deserialization(monkeypatch, tmp_path):
    remote, torch, loaders = Mock(), Mock(), Mock()
    torch.cuda.mem_get_info.return_value = (40 * 1024**3, 40 * 1024**3)
    info = loaders.CheckpointInfo.from_hf_repo.return_value
    info.moshi_weights = tmp_path / "unsafe.pt"
    modules = {"aether.remote": remote, "torch": torch, "moshi.models.loaders": loaders}
    monkeypatch.setattr(importlib, "import_module", modules.__getitem__)
    with pytest.raises(ValueError, match="safetensors"):
        load_backend("permit.json")
    info.get_moshi.assert_not_called()
    info.get_mimi.assert_not_called()


def test_inference_refuses_existing_output_before_audio_import(monkeypatch, tmp_path):
    from aether.backend import infer_file

    remote = Mock()

    def guarded_import(name):
        assert name == "aether.remote"
        return remote

    monkeypatch.setattr(importlib, "import_module", guarded_import)
    with pytest.raises(FileExistsError, match="already exists"):
        infer_file("unused.wav", tmp_path, "permit.json")
