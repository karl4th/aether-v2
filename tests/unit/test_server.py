"""Real loopback transport with deterministic, CPU-only fake engines."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from aether import backend, server, trainer
from aether.protocol import AudioPacket
from aether.server import LiveBackend, StreamOutput, _load_engine_once, make_app

START = {
    "type": "session.start",
    "protocol": 1,
    "audio": {"encoding": "pcm_f32le", "sample_rate": 24000, "channels": 1},
    "seed": 42,
}


class FakeEngine:
    instances: list["FakeEngine"] = []

    def __init__(self) -> None:
        self.closed = False
        self.steps = 0
        FakeEngine.instances.append(self)

    def start(self, seed: int) -> None:
        self.seed = seed

    def step(self, pcm: bytes) -> StreamOutput | None:
        self.steps += 1
        if self.steps == 1:
            return None  # Delay fill-in, matching a real generator's warm-up.
        return StreamOutput(pcm, "" if self.steps % 2 else " hello")

    def close(self) -> None:
        self.closed = True


def audio_packet(index: int) -> bytes:
    # One packet == one 1920-sample model frame, so each call maps to one engine.step().
    return AudioPacket(1, index, index * 1920, bytes(1920 * 4)).encode()


async def connect(client: TestClient):
    ws = await client.ws_connect("/v1/session")
    await ws.send_json(START)
    ready = await ws.receive_json()
    assert ready["type"] == "session.ready"
    return ws


def test_round_trip_produces_audio_and_text() -> None:
    FakeEngine.instances.clear()

    async def run() -> None:
        async with TestClient(TestServer(await make_app(FakeEngine))) as client:
            ws = await connect(client)
            # First step fills the generator's delay warm-up and yields no output.
            for index in range(2):
                await ws.send_bytes(audio_packet(index))
            frame = await ws.receive()
            packet = AudioPacket.decode(frame.data)
            assert packet.kind == 2
            assert await ws.receive_json() == {
                "type": "text.output",
                "frame_index": 0,
                "text": " hello",
            }
            await ws.send_json({"type": "session.stop"})
            closed = await ws.receive_json()
            assert closed == {"type": "session.closed", "reason": "user_stop"}
            await ws.close()
        assert all(engine.closed for engine in FakeEngine.instances)

    asyncio.run(run())


@pytest.mark.parametrize(
    "mutation",
    [
        {"extra": True},
        {"protocol": True},
        {"seed": True},
        {"seed": -1},
        {"audio": {"encoding": "pcm_f32le", "sample_rate": 24000.0, "channels": 1}},
    ],
)
def test_strict_start(mutation: dict) -> None:
    async def run() -> None:
        async with TestClient(TestServer(await make_app(FakeEngine))) as client:
            ws = await client.ws_connect("/v1/session")
            await ws.send_json({**START, **mutation})
            event = await ws.receive_json()
            assert event["code"] == "PROTOCOL_ERROR"
            await ws.close()

    asyncio.run(run())


def test_second_session_is_busy() -> None:
    async def run() -> None:
        app = await make_app(FakeEngine)
        async with TestClient(TestServer(app)) as client:
            first = await connect(client)
            second = await client.ws_connect("/v1/session")
            event = await second.receive_json()
            assert event == {"type": "session.error", "code": "BUSY", "message": "Unavailable"}
            await second.close()
            await first.close()

    asyncio.run(run())


def test_oversized_control_is_protocol_error() -> None:
    async def run() -> None:
        async with TestClient(TestServer(await make_app(FakeEngine))) as client:
            ws = await connect(client)
            await ws.send_str(" " * 16385)
            assert (await ws.receive_json())["code"] == "PROTOCOL_ERROR"
            await ws.close()

    asyncio.run(run())


def test_model_error_marks_backend_unhealthy() -> None:
    class BrokenEngine(FakeEngine):
        def step(self, pcm: bytes) -> StreamOutput | None:
            raise RuntimeError("boom")

    async def run() -> None:
        async with TestClient(TestServer(await make_app(BrokenEngine))) as client:
            ws = await connect(client)
            await ws.send_bytes(audio_packet(0))
            event = await ws.receive_json()
            assert event == {
                "type": "session.error",
                "code": "MODEL_ERROR",
                "message": "MODEL_ERROR",
            }
            await ws.close()
            assert (await client.get("/health/ready")).status == 503

            # A crash must not be reported as BUSY: an operator needs to tell "someone
            # else is talking" apart from "this worker crashed and needs a restart".
            second = await client.ws_connect("/v1/session")
            event = await second.receive_json()
            assert event == {
                "type": "session.error",
                "code": "UNHEALTHY",
                "message": "Worker needs a restart",
            }
            await second.close()

    asyncio.run(run())


def test_bearer_token_required() -> None:
    async def run() -> None:
        app = await make_app(FakeEngine, token="secret")
        async with TestClient(TestServer(app)) as client:
            from aiohttp import WSServerHandshakeError

            with pytest.raises(WSServerHandshakeError):
                await client.ws_connect("/v1/session")
            ws = await client.ws_connect("/v1/session", headers={"Authorization": "Bearer secret"})
            await ws.close()

    asyncio.run(run())


def test_authorize_gate_runs_before_engine_construction() -> None:
    calls: list[int] = []

    def authorize() -> None:
        calls.append(1)

    async def run() -> None:
        app = await make_app(FakeEngine, authorize=authorize)
        async with TestClient(TestServer(app)) as client:
            ws = await connect(client)
            await ws.close()
        assert calls == [1]

    asyncio.run(run())


def test_context_limit_closes_session() -> None:
    async def run() -> None:
        async with TestClient(TestServer(await make_app(FakeEngine, max_duration_ms=10))) as client:
            ws = await connect(client)
            await ws.send_bytes(audio_packet(0))
            event = await ws.receive_json()
            assert event["code"] == "CONTEXT_LIMIT"
            await ws.close()

    asyncio.run(run())


def test_audio_timeout_closes_idle_session() -> None:
    async def run() -> None:
        async with TestClient(TestServer(await make_app(FakeEngine, audio_timeout=0.05))) as client:
            ws = await connect(client)
            event = await ws.receive_json()
            assert event["code"] == "AUDIO_TIMEOUT"
            await ws.close()

    asyncio.run(run())


def test_disconnect_releases_the_slot_for_a_new_session() -> None:
    async def run() -> None:
        app = await make_app(FakeEngine)
        async with TestClient(TestServer(app)) as client:
            first = await connect(client)
            await first.close()
            second = await connect(client)
            await second.close()
            assert (await client.get("/health/ready")).status == 200

    asyncio.run(run())


def test_load_engine_once_applies_checkpoint_when_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_backend = object()
    load_backend = Mock(return_value=fake_backend)
    apply_checkpoint = Mock()
    monkeypatch.setattr(backend, "load_backend", load_backend)
    monkeypatch.setattr(trainer, "apply_checkpoint", apply_checkpoint)

    factory = _load_engine_once("permit.json", checkpoint="checkpoints/step-000020")

    load_backend.assert_called_once_with("permit.json", context=256)
    apply_checkpoint.assert_called_once()
    args = apply_checkpoint.call_args.args
    assert args[0] is fake_backend
    assert str(args[1]) == "checkpoints/step-000020"
    # Weights load once: every call to the returned factory reuses the same engine.
    assert factory() is factory()


def test_load_engine_once_skips_checkpoint_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_backend = object()
    load_backend = Mock(return_value=fake_backend)
    apply_checkpoint = Mock()
    monkeypatch.setattr(backend, "load_backend", load_backend)
    monkeypatch.setattr(trainer, "apply_checkpoint", apply_checkpoint)

    _load_engine_once("permit.json")

    apply_checkpoint.assert_not_called()


def test_live_backend_rewarms_on_every_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: reusing one preloaded LiveBackend across sessions must not skip
    the per-session delay warm-up after the first session already flipped it off."""

    @contextmanager
    def noop():  # type: ignore[no-untyped-def]
        yield

    fake_backend = SimpleNamespace(
        torch=SimpleNamespace(manual_seed=Mock(), inference_mode=noop),
        codec=SimpleNamespace(streaming=lambda count: noop()),
        lm=Mock(),
    )
    fake_generator = SimpleNamespace(streaming=lambda count: noop())
    fake_models = SimpleNamespace(LMGen=Mock(return_value=fake_generator))
    real_import_module = server.importlib.import_module
    monkeypatch.setattr(
        server.importlib,
        "import_module",
        lambda name: fake_models if name == "moshi.models" else real_import_module(name),
    )

    live = LiveBackend(fake_backend)
    live.start(seed=1)
    assert live.first is True
    live.first = False  # Simulate having already stepped through the first frame.
    live.close()

    live.start(seed=2)  # A second, sequential session reusing the same instance.
    assert live.first is True
