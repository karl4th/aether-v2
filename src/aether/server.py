"""Bounded protocol-v1 live service; model operations stay on one worker thread."""

from __future__ import annotations

import asyncio
import importlib
import json
import secrets
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from aether.audio import FRAME_SAMPLES, PCMBuffer, decode_pcm, encode_pcm
from aether.protocol import AudioPacket, PacketSequence

# Largest legitimate control (TEXT) message; enforced by the application so an
# oversized message yields a PROTOCOL_ERROR reply instead of a raw transport close.
MAX_CONTROL_BYTES = 16384
# Transport frame ceiling: control ceiling plus headroom, so an oversized text
# message reaches receive() instead of being closed by aiohttp before it is seen.
_MAX_FRAME_BYTES = MAX_CONTROL_BYTES * 2


@dataclass(frozen=True)
class StreamOutput:
    pcm: bytes
    text: str = ""


class Engine(Protocol):
    def start(self, seed: int) -> None: ...
    def step(self, pcm: bytes) -> StreamOutput | None: ...
    def close(self) -> None: ...


class LiveBackend:
    """Streaming contexts are entered, stepped, and exited on the same thread."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.stack = ExitStack()
        self.first = True

    def start(self, seed: int) -> None:
        b = self.backend
        b.torch.manual_seed(seed)
        models = importlib.import_module("moshi.models")
        self.generator = models.LMGen(b.lm, use_sampling=True, temp=0.8, temp_text=0.7)
        self.stack.enter_context(b.torch.inference_mode())
        self.stack.enter_context(b.codec.streaming(1))
        self.stack.enter_context(self.generator.streaming(1))
        # This instance is reused across sequential sessions (weights load once at
        # server startup); each new session needs its own delay warm-up regardless.
        self.first = True

    def step(self, pcm: bytes) -> StreamOutput | None:
        b = self.backend
        tensor = b.torch.tensor(decode_pcm(pcm), device=b.device).reshape(1, 1, -1)
        codes = b.codec.encode(tensor)
        if self.first:
            self.generator.step(codes)
            self.first = False
        tokens = self.generator.step(codes)
        if tokens is None:
            return None
        if tokens.shape[1] != 9:
            raise RuntimeError("Incompatible generated channels")
        audio = b.codec.decode(tokens[:, 1:9]).clamp(-1, 1).cpu().reshape(-1).tolist()
        token = int(tokens[0, 0, 0].item())
        # SentencePiece pieces preserve the leading word boundary across increments.
        text = "" if token in (0, 3) else str(b.tokenizer.id_to_piece(token)).replace("▁", " ")
        return StreamOutput(encode_pcm(audio), text)

    def close(self) -> None:
        self.stack.close()


def validate_start(value: Any) -> int:
    if not isinstance(value, dict) or set(value) != {"type", "protocol", "audio", "seed"}:
        raise ValueError("Invalid start fields")
    audio = value["audio"]
    if (
        value["type"] != "session.start"
        or type(value["protocol"]) is not int
        or value["protocol"] != 1
        or not isinstance(audio, dict)
        or set(audio) != {"encoding", "sample_rate", "channels"}
        or audio["encoding"] != "pcm_f32le"
        or type(audio["sample_rate"]) is not int
        or audio["sample_rate"] != 24000
        or type(audio["channels"]) is not int
        or audio["channels"] != 1
        or type(value["seed"]) is not int
        or not 0 <= value["seed"] < 2**32
    ):
        raise ValueError("Invalid start configuration")
    return int(value["seed"])


async def make_app(
    engine_factory: Callable[[], Engine],
    *,
    audio_timeout: float = 5.0,
    max_duration_ms: int = 230000,
    token: str | None = None,
    authorize: Callable[[], Any] | None = None,
    queue_frames: int = 6,
) -> Any:
    """Construct a service around an already loaded/warmed engine factory."""
    web = importlib.import_module("aiohttp.web")
    aiohttp = importlib.import_module("aiohttp")
    app = web.Application(client_max_size=_MAX_FRAME_BYTES)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aether-live")
    active = False
    healthy = True
    sockets: set[Any] = set()
    loop = asyncio.get_running_loop()

    async def worker(fn: Callable[..., Any], *args: Any) -> Any:
        return await loop.run_in_executor(executor, partial(fn, *args))

    async def live(request: Any) -> Any:
        return web.json_response({"live": True})

    async def ready(request: Any) -> Any:
        available = healthy and not active
        return web.json_response({"ready": available}, status=200 if available else 503)

    async def session(request: Any) -> Any:
        nonlocal active, healthy
        if token is not None and not secrets.compare_digest(
            request.headers.get("Authorization", ""), f"Bearer {token}"
        ):
            raise web.HTTPUnauthorized()
        ws = web.WebSocketResponse(max_msg_size=_MAX_FRAME_BYTES, compress=False, heartbeat=10)
        await ws.prepare(request)
        if not healthy:
            # Distinct from BUSY: no session is active, a previous one crashed and this
            # worker needs a restart. Reporting it as BUSY would hide that from the operator.
            await ws.send_json(
                {"type": "session.error", "code": "UNHEALTHY", "message": "Worker needs a restart"}
            )
            await ws.close()
            return ws
        if active:
            await ws.send_json({"type": "session.error", "code": "BUSY", "message": "Unavailable"})
            await ws.close()
            return ws
        active = True
        sockets.add(ws)
        engine: Engine | None = None
        tasks: list[asyncio.Task[Any]] = []
        reason = "disconnect"
        code: str | None = None
        incoming: asyncio.Queue[bytes] = asyncio.Queue(maxsize=queue_frames)
        outgoing: asyncio.Queue[tuple[int, StreamOutput]] = asyncio.Queue(maxsize=queue_frames)
        last_audio = loop.time()
        sequence = PacketSequence(1)
        buffer = PCMBuffer(max_queue_frames=queue_frames)

        async def receive() -> str:
            nonlocal last_audio
            async for message in ws:
                if message.type == aiohttp.WSMsgType.BINARY:
                    packet = sequence.accept(message.data)
                    last_audio = loop.time()
                    if sequence.next_offset * 1000 > max_duration_ms * 24000:
                        return "CONTEXT_LIMIT"
                    buffer.push(packet.pcm)
                    while (frame := buffer.pop()) is not None:
                        incoming.put_nowait(frame.pcm)
                elif message.type == aiohttp.WSMsgType.TEXT:
                    if len(message.data.encode()) > MAX_CONTROL_BYTES:
                        raise ValueError("Oversized control message")
                    if json.loads(message.data) != {"type": "session.stop"}:
                        raise ValueError("Invalid control message")
                    return "user_stop"
                elif message.type == aiohttp.WSMsgType.ERROR:
                    raise ValueError("Invalid WebSocket message")
            return "disconnect"

        async def generate() -> str:
            assert engine is not None
            index = 0
            while True:
                pcm = await incoming.get()
                try:
                    output = await worker(engine.step, pcm)
                    if output is not None:
                        AudioPacket(2, 0, 0, output.pcm).encode()
                        if not isinstance(output.text, str) or len(output.text.encode()) > 8192:
                            raise ValueError("Invalid model text output")
                except Exception as error:
                    raise RuntimeError("Model generation failed") from error
                if output is not None:
                    outgoing.put_nowait((index, output))
                    index += 1

        async def send() -> str:
            out_sequence = offset = 0
            while True:
                index, output = await outgoing.get()
                # Packet.encode validates model output before crossing the trust boundary.
                packet = AudioPacket(2, out_sequence, offset, output.pcm)
                await asyncio.wait_for(ws.send_bytes(packet.encode()), audio_timeout)
                out_sequence += 1
                offset += len(output.pcm) // 4
                if output.text:
                    await asyncio.wait_for(
                        ws.send_json(
                            {"type": "text.output", "frame_index": index, "text": output.text}
                        ),
                        audio_timeout,
                    )

        async def watchdog() -> str:
            while True:
                remaining = audio_timeout - (loop.time() - last_audio)
                if remaining <= 0:
                    return "AUDIO_TIMEOUT"
                await asyncio.sleep(remaining)

        try:
            message = await asyncio.wait_for(ws.receive(), audio_timeout)
            if message.type != aiohttp.WSMsgType.TEXT:
                raise ValueError("Expected start")
            if len(message.data.encode()) > MAX_CONTROL_BYTES:
                raise ValueError("Oversized control message")
            seed = validate_start(json.loads(message.data))
            if authorize is not None:
                await worker(authorize)
            try:
                engine = await worker(engine_factory)
                await worker(engine.start, seed)
            except Exception as error:
                raise RuntimeError("Model session initialization failed") from error
            await ws.send_json(
                {
                    "type": "session.ready",
                    "protocol": 1,
                    "session_id": str(uuid4()),
                    "frame_samples": FRAME_SAMPLES,
                    "max_input_packet_samples": 3840,
                    "max_duration_ms": max_duration_ms,
                }
            )
            last_audio = loop.time()
            tasks = [asyncio.create_task(fn()) for fn in (receive, generate, send, watchdog)]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                result = task.result()
                if result:
                    reason = result
            if reason in {"AUDIO_TIMEOUT", "CONTEXT_LIMIT"}:
                code = reason
        except (ValueError, TypeError) as error:
            code = "PROTOCOL_ERROR"
            print(f"PROTOCOL_ERROR: {error}", flush=True)
        except (asyncio.QueueFull, BufferError) as error:
            code = "OVERLOAD"
            print(f"OVERLOAD: {error!r} (a bounded queue could not drain in time)", flush=True)
        except TimeoutError as error:
            code = "AUDIO_TIMEOUT"
            print(f"AUDIO_TIMEOUT: {error}", flush=True)
        except Exception:
            code = "MODEL_ERROR"
            healthy = False
            # Never sent to the client, but this is the only record of what actually
            # broke; without it, MODEL_ERROR/UNHEALTHY is undiagnosable after the fact.
            traceback.print_exc()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if engine is not None:
                try:
                    # Queued after any unfinished step: never free an executing CUDA state.
                    await worker(engine.close)
                except Exception:
                    healthy = False
                    code = "MODEL_ERROR"
                    traceback.print_exc()
            buffer.reset()
            try:
                if not ws.closed:
                    event = (
                        {"type": "session.error", "code": code, "message": code}
                        if code
                        else {"type": "session.closed", "reason": reason}
                    )
                    await asyncio.wait_for(ws.send_json(event), audio_timeout)
                    await ws.close()
            except (ConnectionError, TimeoutError):
                pass
            sockets.discard(ws)
            active = False
        return ws

    async def shutdown(app: Any) -> None:
        for ws in tuple(sockets):
            await ws.close(code=1001, message=b"Server shutdown")

    async def cleanup(app: Any) -> None:
        await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)

    app.router.add_get("/health/live", live)
    app.router.add_get("/health/ready", ready)
    app.router.add_get("/v1/session", session)
    app.on_shutdown.append(shutdown)
    app.on_cleanup.append(cleanup)
    return app


def _warm_up(engine: Engine, *, steps: int = 40) -> None:
    """Run real forward passes on silent audio so CUDA kernels are compiled and
    the caching allocator is stable before the first real client connects.

    The first several real steps of a cold model are routinely much slower than
    steady state, and PyTorch's caching allocator can keep taking occasional
    slow paths (real cudaMalloc calls) for more steps than a short warm-up
    covers; either shows up as a same-session OVERLOAD once a real client
    starts streaming audio in real time (a bounded queue filling faster than
    it can drain). Every step's time is logged so a persistent slow steady
    state is distinguishable from a handful of one-off allocator spikes.
    """
    print(f"Warming up: running {steps} synthetic steps on silent audio...", flush=True)
    silence = encode_pcm([0.0] * FRAME_SAMPLES)
    engine.start(seed=0)
    try:
        timings = []
        for index in range(steps):
            started = time.monotonic()
            engine.step(silence)
            elapsed = time.monotonic() - started
            timings.append(elapsed)
            print(f"  warm-up step {index + 1}/{steps}: {elapsed * 1000:.0f}ms", flush=True)
    finally:
        engine.close()
    ordered = sorted(timings)
    low, median, high = ordered[0] * 1000, ordered[len(ordered) // 2] * 1000, ordered[-1] * 1000
    over_budget = sum(1 for t in timings if t > 0.080)
    print(
        f"Warm-up done: step time (ms) min={low:.0f} median={median:.0f} max={high:.0f}; "
        f"{over_budget}/{steps} steps exceeded the 80ms real-time frame budget",
        flush=True,
    )


def _load_engine_once(
    permit_path: str, *, context: int = 256, checkpoint: str | None = None
) -> Callable[[], Engine]:
    """Load weights exactly once, at startup; every session reuses this instance.

    Loading is slow (a cold cache downloads ~15 GB); doing it per-session made
    every new connection re-download and re-load the model, which routinely
    outran the client's own wait for `session.ready`.
    """
    from aether.backend import load_backend

    print("Loading model weights (a cold cache can take several minutes)...", flush=True)
    backend = load_backend(permit_path, context=context)
    if checkpoint is not None:
        from aether.trainer import apply_checkpoint

        print(f"Applying adapter checkpoint: {checkpoint}", flush=True)
        apply_checkpoint(backend, Path(checkpoint))
    engine = LiveBackend(backend)
    print("Model loaded.", flush=True)
    _warm_up(engine)
    print("Warmed up; starting the live service.", flush=True)
    return lambda: engine


def serve(
    permit_path: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    token: str | None = None,
    checkpoint: str | None = None,
    queue_frames: int = 25,
) -> None:
    """Run the live service in the foreground; blocks until interrupted.

    queue_frames default (25 frames, ~2s) is higher than the original 6-frame
    (~480ms) design target: measurement showed steady-state steps well inside
    the 80ms budget, but occasional one-off spikes (allocator/network) that a
    480ms buffer had essentially no slack to absorb.
    """
    web = importlib.import_module("aiohttp.web")

    def authorize() -> None:
        from aether.remote import require_remote_runtime

        require_remote_runtime(permit_path=permit_path)

    engine_factory = _load_engine_once(permit_path, checkpoint=checkpoint)

    async def build() -> Any:
        return await make_app(
            engine_factory, token=token, authorize=authorize, queue_frames=queue_frames
        )

    print(f"Listening on {host}:{port} (Ctrl-C to stop).", flush=True)
    web.run_app(build(), host=host, port=port, print=None)


__all__ = [
    "Engine",
    "LiveBackend",
    "StreamOutput",
    "make_app",
    "serve",
    "validate_start",
]
