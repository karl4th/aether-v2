"""Explicit opt-in microphone client; audio callbacks never perform network I/O."""

from __future__ import annotations

import asyncio
import importlib
import json
import queue
from typing import Any

from aether.protocol import AudioPacket, PacketSequence


class AudioBridge:
    """Bounded cross-thread buffers. Overflow terminates instead of losing speech."""

    def __init__(self, capacity: int = 32) -> None:
        self.capture: queue.Queue[bytes] = queue.Queue(capacity)
        self.playback: queue.Queue[bytes] = queue.Queue(capacity)
        self.errors: queue.Queue[str] = queue.Queue(1)
        self.pending = b""

    def fail(self, message: str) -> None:
        try:
            self.errors.put_nowait(message)
        except queue.Full:
            pass  # Preserve the first failure; audio data are never silently dropped.

    def input_callback(self, indata: Any, frames: int, time: Any, status: Any) -> None:
        if status:
            self.fail(f"CAPTURE_ERROR: {status}")
            return
        try:
            self.capture.put_nowait(bytes(indata))
        except queue.Full:
            self.fail("CAPTURE_QUEUE_FULL")

    def output_callback(self, outdata: Any, frames: int, time: Any, status: Any) -> None:
        if status:
            self.fail(f"PLAYBACK_ERROR: {status}")
        needed = frames * 4
        buffer = self.pending
        while len(buffer) < needed:
            try:
                buffer += self.playback.get_nowait()
            except queue.Empty:
                break
        chunk, self.pending = buffer[:needed], buffer[needed:]
        outdata[: len(chunk)] = chunk
        outdata[len(chunk) :] = b"\x00" * (needed - len(chunk))


async def _send_loop(
    ws: Any, bridge: AudioBridge, sequence: PacketSequence, deadline: float | None
) -> None:
    loop = asyncio.get_running_loop()
    while True:
        if not bridge.errors.empty():
            raise RuntimeError(bridge.errors.get_nowait())
        try:
            pcm = bridge.capture.get_nowait()
        except queue.Empty:
            if deadline is not None and loop.time() >= deadline:
                await ws.send_json({"type": "session.stop"})
                return
            await asyncio.sleep(0.01)
            continue
        packet = AudioPacket(1, sequence.next_sequence, sequence.next_offset, pcm)
        await ws.send_bytes(packet.encode())
        sequence.next_sequence += 1
        sequence.next_offset += len(pcm) // 4


async def _receive_loop(ws: Any, bridge: AudioBridge, aiohttp: Any) -> None:
    async for message in ws:
        if message.type == aiohttp.WSMsgType.BINARY:
            packet = AudioPacket.decode(message.data)
            try:
                bridge.playback.put_nowait(packet.pcm)
            except queue.Full:
                bridge.fail("PLAYBACK_QUEUE_FULL")
        elif message.type == aiohttp.WSMsgType.TEXT:
            event = json.loads(message.data)
            if event.get("type") == "text.output":
                print(event.get("text", ""), end="", flush=True)
            elif event.get("type") == "session.closed":
                return
            elif event.get("type") == "session.error":
                raise RuntimeError(f"{event.get('code', 'SESSION_ERROR')}: {event.get('message')}")
        elif message.type == aiohttp.WSMsgType.ERROR:
            raise RuntimeError("CONNECTION_CLOSED")
    raise RuntimeError("CONNECTION_CLOSED")


async def talk(
    url: str,
    *,
    duration: float | None = None,
    seed: int = 42,
    auth_token: str | None = None,
    ready_timeout: float = 300.0,
) -> None:
    """Open the microphone/speaker and a live protocol-v1 session; blocks until closed.

    ready_timeout is generous by default: a server that loads weights lazily on
    the first session (rather than once at startup) can easily exceed 30 seconds
    on a cold cache, and a short timeout here just races that instead of it.
    """
    aiohttp = importlib.import_module("aiohttp")
    sounddevice = importlib.import_module("sounddevice")
    bridge = AudioBridge()
    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
    loop = asyncio.get_running_loop()
    async with aiohttp.ClientSession() as http_session:
        async with http_session.ws_connect(
            url, heartbeat=20, max_msg_size=1_048_576, headers=headers
        ) as ws:
            await ws.send_json(
                {
                    "type": "session.start",
                    "protocol": 1,
                    "audio": {"encoding": "pcm_f32le", "sample_rate": 24000, "channels": 1},
                    "seed": seed,
                }
            )
            ready = await asyncio.wait_for(ws.receive_json(), timeout=ready_timeout)
            if not isinstance(ready, dict) or ready.get("type") != "session.ready":
                raise RuntimeError(f"SESSION_START_FAILED: {ready}")
            # Context managers stop and close both devices on error or cancellation.
            with (
                sounddevice.RawInputStream(
                    samplerate=24000,
                    channels=1,
                    dtype="float32",
                    blocksize=1920,
                    callback=bridge.input_callback,
                ),
                sounddevice.RawOutputStream(
                    samplerate=24000,
                    channels=1,
                    dtype="float32",
                    blocksize=1920,
                    callback=bridge.output_callback,
                ),
            ):
                sequence = PacketSequence(1)
                deadline = loop.time() + duration if duration is not None else None
                tasks = [
                    asyncio.create_task(_send_loop(ws, bridge, sequence, deadline)),
                    asyncio.create_task(_receive_loop(ws, bridge, aiohttp)),
                ]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    task.result()
