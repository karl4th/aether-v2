"""Bounded PCM framing and explicit offline sample-rate conversion."""

import math
import struct
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

FRAME_SAMPLES = 1920
SAMPLE_RATE = 24000
MAX_PACKET_SAMPLES = 3840


def decode_pcm(payload: bytes) -> tuple[float, ...]:
    if len(payload) % 4:
        raise ValueError("PCM must contain complete float32 little-endian samples")
    values = tuple(value[0] for value in struct.iter_unpack("<f", payload))
    if any(not math.isfinite(x) or abs(x) > 1 for x in values):
        raise ValueError("PCM samples must be finite and within [-1, 1]")
    return values


def encode_pcm(values: Sequence[float]) -> bytes:
    if any(type(x) not in (float, int) or not math.isfinite(x) or abs(x) > 1 for x in values):
        raise ValueError("PCM samples must be finite and within [-1, 1]")
    return struct.pack(f"<{len(values)}f", *values)


@dataclass(frozen=True)
class AudioFrame:
    index: int
    sample_offset: int
    pcm: bytes


class PCMBuffer:
    """Atomic packet acceptance; overflow never removes accepted audio.

    Capacity is full frames plus at most one partial frame. Empty input is not a
    packet and never creates silence. Call pop until None after each push.
    """

    def __init__(self, max_queue_frames: int = 4) -> None:
        if type(max_queue_frames) is not int or max_queue_frames < 1:
            raise ValueError("max_queue_frames must be a positive integer")
        self.capacity = max_queue_frames * FRAME_SAMPLES + FRAME_SAMPLES - 1
        self.reset()

    def reset(self) -> None:
        self._samples = bytearray()
        self._frames: deque[AudioFrame] = deque()
        self.received_samples = 0
        self.emitted_samples = 0
        self.padded_samples = 0
        self.discarded_samples = 0
        self.accepted_packets = 0
        self.overflow_count = 0
        self._frame_index = 0
        self.closed = False

    @property
    def buffered_samples(self) -> int:
        return len(self._frames) * FRAME_SAMPLES + len(self._samples) // 4

    def push(self, pcm: bytes) -> None:
        if self.closed:
            raise ValueError("buffer is closed")
        count = len(pcm) // 4
        if not 1 <= count <= MAX_PACKET_SAMPLES:
            raise ValueError("packet must contain 1..3840 samples")
        decode_pcm(pcm)
        if self.buffered_samples + count > self.capacity:
            self.overflow_count += 1
            raise BufferError("OVERLOAD: audio queue capacity exceeded")
        self._samples.extend(pcm)
        self.received_samples += count
        self.accepted_packets += 1
        self._extract_frames()

    def _extract_frames(self) -> None:
        size = FRAME_SAMPLES * 4
        while len(self._samples) >= size:
            self._frames.append(
                AudioFrame(
                    self._frame_index,
                    self._frame_index * FRAME_SAMPLES,
                    bytes(self._samples[:size]),
                )
            )
            self._frame_index += 1
            del self._samples[:size]

    def pop(self) -> AudioFrame | None:
        if not self._frames:
            return None
        frame = self._frames.popleft()
        self.emitted_samples += FRAME_SAMPLES
        return frame

    def finish(self, *, pad: bool = False) -> None:
        """Offline pad is opt-in; live stop discards and reports only the tail."""
        if self.closed:
            raise ValueError("buffer already closed")
        tail = len(self._samples) // 4
        if tail and pad:
            added = FRAME_SAMPLES - tail
            if self.buffered_samples + added > self.capacity:
                self.overflow_count += 1
                raise BufferError("OVERLOAD: drain full frames before padding")
            self._samples.extend(bytes(added * 4))
            self.padded_samples += added
            self._extract_frames()
        else:
            self.discarded_samples += tail
            self._samples.clear()
        self.closed = True


def resample_pcm(pcm: bytes, source_rate: int, target_rate: int = SAMPLE_RATE) -> bytes:
    """Offline windowed-sinc conversion; never resample individual live packets.

    Output length is ceil(N * target/source). A normalized Hann-windowed sinc
    suppresses aliases on downsampling. Edge extension holds endpoint samples.
    This reference implementation prioritizes correctness over live performance.
    """
    if any(
        type(rate) is not int or not 8000 <= rate <= 192000 for rate in (source_rate, target_rate)
    ):
        raise ValueError("sample rates must be integers in 8000..192000")
    samples = decode_pcm(pcm)
    if source_rate == target_rate or not samples:
        return pcm
    cutoff = min(1.0, target_rate / source_rate)
    radius = math.ceil(24 / cutoff)
    result: list[float] = []
    count = (len(samples) * target_rate + source_rate - 1) // source_rate
    for index in range(count):
        position = index * source_rate / target_rate
        center = math.floor(position)
        total = weight_sum = 0.0
        for tap in range(center - radius + 1, center + radius + 1):
            distance = position - tap
            if abs(distance) >= radius:
                continue
            phase = cutoff * distance
            sinc = math.sin(math.pi * phase) / (math.pi * phase) if phase else 1.0
            weight = cutoff * sinc * (0.5 + 0.5 * math.cos(math.pi * distance / radius))
            total += samples[min(max(tap, 0), len(samples) - 1)] * weight
            weight_sum += weight
        result.append(max(-1.0, min(1.0, total / weight_sum)))
    return encode_pcm(result)
