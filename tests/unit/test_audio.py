import math
import random
import struct

import pytest

from aether.audio import (
    FRAME_SAMPLES,
    PCMBuffer,
    decode_pcm,
    encode_pcm,
    resample_pcm,
    sanitize_pcm,
)


def drain(buffer: PCMBuffer) -> bytes:
    frames = []
    while (frame := buffer.pop()) is not None:
        assert frame.sample_offset == frame.index * FRAME_SAMPLES
        frames.append(frame.pcm)
    return b"".join(frames)


@pytest.mark.parametrize("seed", range(5))
def test_packet_boundaries_and_conservation(seed: int) -> None:
    rng = random.Random(seed)
    original = encode_pcm([math.sin(i / 100) for i in range(1920 * 9 + 71)])
    buffer = PCMBuffer(2)
    output = b""
    offset = 0
    while offset < len(original):
        size = rng.randint(1, 3840) * 4
        packet = original[offset : offset + size]
        buffer.push(packet)
        output += drain(buffer)
        offset += len(packet)
    buffer.finish(pad=True)
    output += drain(buffer)
    assert output[: len(original)] == original
    assert output[len(original) :] == bytes((1920 - 71) * 4)
    assert buffer.received_samples + buffer.padded_samples == buffer.emitted_samples
    assert buffer.buffered_samples == 0


def test_overflow_is_atomic_and_tail_can_be_retried() -> None:
    buffer = PCMBuffer(1)
    buffer.push(encode_pcm([0.25] * (1920 + 10)))
    with pytest.raises(BufferError):
        buffer.push(encode_pcm([0.5] * 1920))
    assert buffer.received_samples == 1930
    assert buffer.accepted_packets == 1
    with pytest.raises(BufferError):
        buffer.finish(pad=True)
    assert not buffer.closed
    assert buffer.overflow_count == 2
    assert decode_pcm(drain(buffer)) == (0.25,) * 1920
    buffer.finish(pad=True)
    assert decode_pcm(drain(buffer)) == (0.25,) * 10 + (0.0,) * 1910


@pytest.mark.parametrize(
    "pcm",
    [
        b"",
        b"123",
        struct.pack("<f", float("nan")),
        struct.pack("<f", float("inf")),
        struct.pack("<f", 1.1),
        bytes(3841 * 4),
    ],
)
def test_bad_packets_do_not_change_state(pcm: bytes) -> None:
    buffer = PCMBuffer()
    with pytest.raises(ValueError):
        buffer.push(pcm)
    assert buffer.received_samples == buffer.buffered_samples == buffer.accepted_packets == 0


def test_sanitize_pcm_clamps_out_of_range_and_replaces_non_finite() -> None:
    raw = struct.pack("<5f", 1.5, -1.5, float("nan"), float("inf"), 0.25)
    assert decode_pcm(sanitize_pcm(raw)) == (1.0, -1.0, 0.0, 0.0, 0.25)


def test_sanitize_pcm_rejects_incomplete_samples() -> None:
    with pytest.raises(ValueError):
        sanitize_pcm(b"123")


def test_silence_absence_tail_reset_and_session_isolation() -> None:
    first, second = PCMBuffer(), PCMBuffer()
    assert first.pop() is None
    first.push(bytes(1920 * 4))
    assert first.pop() is not None
    assert second.pop() is None
    second.push(bytes(7 * 4))
    second.finish()
    assert second.discarded_samples == 7
    assert second.received_samples == second.emitted_samples + second.discarded_samples
    with pytest.raises(ValueError):
        second.push(bytes(4))
    second.reset()
    assert second.received_samples == second.discarded_samples == 0
    second.push(bytes(1920 * 4))
    frame = second.pop()
    assert frame is not None and frame.index == frame.sample_offset == 0


@pytest.mark.parametrize("source,target", [(48000, 24000), (44100, 24000), (16000, 24000)])
def test_resample_length_dc_and_identity(source: int, target: int) -> None:
    pcm = encode_pcm([0.25] * 997)
    converted = decode_pcm(resample_pcm(pcm, source, target))
    assert len(converted) == math.ceil(997 * target / source)
    assert converted == pytest.approx([0.25] * len(converted), abs=1e-6)
    assert resample_pcm(pcm, source, source) == pcm
    assert resample_pcm(b"", source, target) == b""


def test_resample_preserves_passband_and_suppresses_aliases() -> None:
    def convert(frequency: float) -> tuple[float, ...]:
        pcm = encode_pcm([0.5 * math.sin(2 * math.pi * frequency * i / 48000) for i in range(4800)])
        return decode_pcm(resample_pcm(pcm, 48000))[100:-100]

    low = convert(1000)
    high = convert(18000)
    rms_low = math.sqrt(sum(x * x for x in low) / len(low))
    rms_high = math.sqrt(sum(x * x for x in high) / len(high))
    assert 0.34 < rms_low < 0.36
    assert rms_high < 0.005
