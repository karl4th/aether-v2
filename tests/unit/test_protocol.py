import struct

import pytest

from aether.protocol import AudioPacket, PacketSequence


def test_exact_wire_layout_and_independent_sequences() -> None:
    packet = AudioPacket(1, 0, 0, struct.pack("<ff", 0.25, -0.25))
    raw = packet.encode()
    assert raw[:18] == struct.pack("<BBIQI", 1, 1, 0, 0, 2)
    assert AudioPacket.decode(raw) == packet
    first, second = PacketSequence(1), PacketSequence(1)
    first.accept(raw)
    second.accept(raw)
    with pytest.raises(ValueError, match="discontinuity"):
        first.accept(raw)
    with pytest.raises(ValueError, match="discontinuity"):
        first.accept(AudioPacket(1, 1, 1, bytes(4)).encode())
    first.accept(AudioPacket(1, 1, 2, bytes(4)).encode())
    assert first.next_offset == 3 and second.next_offset == 2
    first.reset()
    first.accept(raw)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        bytes(17),
        struct.pack("<BBIQI", 2, 1, 0, 0, 1) + bytes(4),
        struct.pack("<BBIQI", 1, 3, 0, 0, 1) + bytes(4),
        struct.pack("<BBIQI", 1, 1, 0, 0, 2) + bytes(4),
        struct.pack("<BBIQI", 1, 1, 0, 0, 0),
        struct.pack("<BBIQI", 1, 1, 0, 0, 1) + struct.pack("<f", float("nan")),
        bytes(15379),
    ],
)
def test_malformed_packets_fail_without_advancing(raw: bytes) -> None:
    sequence = PacketSequence(1)
    with pytest.raises(ValueError):
        sequence.accept(raw)
    assert sequence.next_sequence == sequence.next_offset == 0


def test_maximum_payload_and_counter_range() -> None:
    packet = AudioPacket(2, 2**32 - 1, 2**64 - 1, bytes(3840 * 4))
    assert len(packet.encode()) == 15378
    assert AudioPacket.decode(packet.encode()) == packet
    with pytest.raises(ValueError):
        AudioPacket(1, 2**32, 0, bytes(4)).encode()
