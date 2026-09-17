"""Version 1 binary audio transport, independent of server and model backend."""

import struct
from dataclasses import dataclass
from typing import Literal

from aether.audio import MAX_PACKET_SAMPLES, decode_pcm

_HEADER = struct.Struct("<BBIQI")


@dataclass(frozen=True)
class AudioPacket:
    kind: Literal[1, 2]
    sequence: int
    sample_offset: int
    pcm: bytes

    def encode(self) -> bytes:
        if type(self.kind) is not int or self.kind not in (1, 2):
            raise ValueError("PROTOCOL_ERROR: unknown audio direction")
        if type(self.sequence) is not int or not 0 <= self.sequence < 2**32:
            raise ValueError("PROTOCOL_ERROR: sequence out of uint32 range")
        if type(self.sample_offset) is not int or not 0 <= self.sample_offset < 2**64:
            raise ValueError("PROTOCOL_ERROR: offset out of uint64 range")
        count = len(self.pcm) // 4
        if not 1 <= count <= MAX_PACKET_SAMPLES:
            raise ValueError("PROTOCOL_ERROR: packet sample limit")
        decode_pcm(self.pcm)
        return _HEADER.pack(1, self.kind, self.sequence, self.sample_offset, count) + self.pcm

    @classmethod
    def decode(cls, raw: bytes) -> "AudioPacket":
        if not _HEADER.size <= len(raw) <= _HEADER.size + MAX_PACKET_SAMPLES * 4:
            raise ValueError("PROTOCOL_ERROR: invalid packet size")
        version, kind, sequence, offset, count = _HEADER.unpack_from(raw)
        if version != 1 or kind not in (1, 2):
            raise ValueError("PROTOCOL_ERROR: unknown version/direction")
        if not 1 <= count <= MAX_PACKET_SAMPLES or len(raw) != _HEADER.size + count * 4:
            raise ValueError("PROTOCOL_ERROR: count does not match payload")
        pcm = raw[_HEADER.size :]
        decode_pcm(pcm)
        return cls(1 if kind == 1 else 2, sequence, offset, pcm)


class PacketSequence:
    """Separate per direction/session; rejected packets do not advance counters."""

    def __init__(self, kind: Literal[1, 2]) -> None:
        if type(kind) is not int or kind not in (1, 2):
            raise ValueError("invalid direction")
        self.kind: Literal[1, 2] = kind
        self.reset()

    def reset(self) -> None:
        self.next_sequence = 0
        self.next_offset = 0

    def accept(self, raw: bytes) -> AudioPacket:
        packet = AudioPacket.decode(raw)
        if (packet.kind, packet.sequence, packet.sample_offset) != (
            self.kind,
            self.next_sequence,
            self.next_offset,
        ):
            raise ValueError("PROTOCOL_ERROR: sequence or sample offset discontinuity")
        self.next_sequence += 1
        self.next_offset += len(packet.pcm) // 4
        return packet
