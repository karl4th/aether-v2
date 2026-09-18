"""Mock transport and audio devices; no real network or microphone access."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from aether import client
from aether.protocol import AudioPacket

captured: dict[str, Any] = {}


class Message:
    def __init__(self, kind: Any, data: Any) -> None:
        self.type = kind
        self.data = data


class FakeWebSocket:
    def __init__(self, incoming: list[Any], ready: dict) -> None:
        self.incoming = incoming
        self.ready = ready
        self.sent_bytes: list[bytes] = []
        self.sent_json: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.sent_json.append(payload)

    async def send_bytes(self, payload: bytes) -> None:
        self.sent_bytes.append(payload)

    async def receive_json(self) -> dict:
        return self.ready

    def __aiter__(self) -> "FakeWebSocket":
        self._iterator = iter(self.incoming)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iterator)
        except StopIteration:
            raise StopAsyncIteration from None


class _WSContext:
    def __init__(self, ws: FakeWebSocket) -> None:
        self.ws = ws

    async def __aenter__(self) -> FakeWebSocket:
        return self.ws

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _ClientSession:
    def __init__(self, ws: FakeWebSocket) -> None:
        self.ws = ws

    async def __aenter__(self) -> "_ClientSession":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    def ws_connect(self, url: str, **kwargs: Any) -> _WSContext:
        captured["url"] = url
        captured["headers"] = kwargs.get("headers")
        return _WSContext(self.ws)


def fake_aiohttp(ws: FakeWebSocket) -> SimpleNamespace:
    class WSMsgType:
        BINARY = "binary"
        TEXT = "text"
        ERROR = "error"

    return SimpleNamespace(
        ClientSession=lambda: _ClientSession(ws),
        WSMsgType=WSMsgType,
    )


class Context:
    def __init__(self, name: str, closed: list[str]) -> None:
        self.name = name
        self.closed = closed

    def __enter__(self) -> "Context":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.closed.append(self.name)


def fake_sounddevice(closed: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        RawInputStream=lambda **kw: Context("input", closed),
        RawOutputStream=lambda **kw: Context("output", closed),
    )


def test_talk_streams_capture_to_socket_and_playback_from_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = FakeWebSocket(
        [
            Message("binary", AudioPacket(2, 0, 0, bytes(1920 * 4)).encode()),
            Message("text", '{"type": "text.output", "frame_index": 0, "text": "hi"}'),
            Message("text", '{"type": "session.closed", "reason": "user_stop"}'),
        ],
        ready={"type": "session.ready"},
    )
    closed: list[str] = []
    modules = {"aiohttp": fake_aiohttp(ws), "sounddevice": fake_sounddevice(closed)}
    monkeypatch.setattr(client.importlib, "import_module", modules.__getitem__)

    asyncio.run(client.talk("wss://example.invalid/live", seed=7))

    assert ws.sent_json[0]["seed"] == 7
    assert closed == ["output", "input"]


def test_talk_sends_auth_header(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeWebSocket(
        [Message("text", '{"type": "session.closed", "reason": "user_stop"}')],
        ready={"type": "session.ready"},
    )
    closed: list[str] = []
    modules = {"aiohttp": fake_aiohttp(ws), "sounddevice": fake_sounddevice(closed)}
    monkeypatch.setattr(client.importlib, "import_module", modules.__getitem__)

    asyncio.run(client.talk("wss://example.invalid/live", auth_token="secret"))

    assert captured["headers"] == {"Authorization": "Bearer secret"}


def test_talk_rejects_bad_ready_message(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeWebSocket([], ready={"type": "session.error", "code": "BUSY"})
    modules = {"aiohttp": fake_aiohttp(ws), "sounddevice": SimpleNamespace()}
    monkeypatch.setattr(client.importlib, "import_module", modules.__getitem__)

    with pytest.raises(RuntimeError, match="SESSION_START_FAILED"):
        asyncio.run(client.talk("wss://example.invalid/live"))


def test_connection_error_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeWebSocket([Message("error", None)], ready={"type": "session.ready"})
    closed: list[str] = []
    modules = {"aiohttp": fake_aiohttp(ws), "sounddevice": fake_sounddevice(closed)}
    monkeypatch.setattr(client.importlib, "import_module", modules.__getitem__)

    with pytest.raises(RuntimeError, match="CONNECTION_CLOSED"):
        asyncio.run(client.talk("wss://example.invalid/live"))
    assert closed == ["output", "input"]


def test_server_reported_error_is_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeWebSocket(
        [Message("text", '{"type": "session.error", "code": "OVERLOAD", "message": "full"}')],
        ready={"type": "session.ready"},
    )
    closed: list[str] = []
    modules = {"aiohttp": fake_aiohttp(ws), "sounddevice": fake_sounddevice(closed)}
    monkeypatch.setattr(client.importlib, "import_module", modules.__getitem__)

    with pytest.raises(RuntimeError, match="OVERLOAD"):
        asyncio.run(client.talk("wss://example.invalid/live"))


def test_audio_bridge_output_callback_pads_silence_and_reports_status() -> None:
    bridge = client.AudioBridge()
    outdata = bytearray(8)
    bridge.output_callback(outdata, 2, None, "underflow")
    assert bytes(outdata) == b"\x00" * 8
    assert not bridge.errors.empty()


def test_audio_bridge_output_callback_drains_playback_queue() -> None:
    bridge = client.AudioBridge()
    bridge.playback.put_nowait(b"\x01\x02\x03\x04")
    outdata = bytearray(4)
    bridge.output_callback(outdata, 1, None, None)
    assert bytes(outdata) == b"\x01\x02\x03\x04"


def test_audio_bridge_input_callback_reports_full_queue() -> None:
    bridge = client.AudioBridge(capacity=1)
    bridge.input_callback(b"\x00" * 4, 1, None, None)
    bridge.input_callback(b"\x00" * 4, 1, None, None)
    assert bridge.errors.get_nowait() == "CAPTURE_QUEUE_FULL"


def test_audio_bridge_input_callback_sanitizes_out_of_range_hardware_samples() -> None:
    import struct

    bridge = client.AudioBridge()
    bridge.input_callback(struct.pack("<f", 1.5), 1, None, None)
    assert bridge.capture.get_nowait() == struct.pack("<f", 1.0)
    assert bridge.errors.empty()
