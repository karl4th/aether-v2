"""Shared physical/delayed token schedule without a tensor backend.

Channels: text, eight output codebooks, eight input codebooks.
BOS is input-only; missing and pending positions never enter lookup or loss.
"""

from collections.abc import Sequence
from enum import IntEnum

from aether.config import ModelConfig


class Sentinel(IntEnum):
    MISSING = -1
    PENDING = -2


class TokenSchedule:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def vocabulary(self, channel: int) -> int:
        if not 0 <= channel < 17:
            raise ValueError("channel must be in 0..16")
        return self.config.text_vocab_size if channel == 0 else self.config.audio_vocab_size

    def lookup_mask(self, token: int, channel: int) -> bool:
        """BOS has an embedding; internal sentinels do not."""
        return type(token) is int and 0 <= token <= self.vocabulary(channel)

    def loss_mask(self, token: int, channel: int) -> bool:
        if type(token) is not int or not 0 <= token < self.vocabulary(channel):
            return False
        return channel < 9 and not (channel == 0 and token == 3)

    def validate_frame(self, frame: Sequence[int]) -> tuple[int, ...]:
        if len(frame) != 17:
            raise ValueError("expected 17 channels")
        for channel, token in enumerate(frame):
            if type(token) is not int or not 0 <= token < self.vocabulary(channel):
                raise ValueError(f"invalid physical token in channel {channel}")
        return tuple(frame)

    def shift(self, frames: Sequence[Sequence[int]]) -> list[tuple[int, ...]]:
        stream = DelayedStream(self)
        result = [stream.push(frame) for frame in frames]
        result.extend(stream.finish())
        return result

    def restore(self, steps: Sequence[Sequence[int]], count: int) -> list[tuple[int, ...]]:
        if count < 0 or len(steps) != (count + max(self.config.delays) if count else 0):
            raise ValueError("step count does not match physical length and flush")
        if any(len(step) != 17 for step in steps):
            raise ValueError("expected 17 channels")
        return [
            self.validate_frame(tuple(steps[t + d][k] for k, d in enumerate(self.config.delays)))
            for t in range(count)
        ]

    def decoder_codes(self, frame: Sequence[int]) -> tuple[int, ...]:
        """Only a validated full physical frame may reach the output decoder."""
        return self.validate_frame(frame)[1:9]


class DelayedStream:
    """One-frame bounded state for the v1 schedule, plus explicit reset/flush."""

    def __init__(self, schedule: TokenSchedule) -> None:
        self.schedule = schedule
        self.reset()

    def reset(self) -> None:
        self.previous: tuple[int, ...] | None = None
        self.closed = False

    def push(self, frame: Sequence[int]) -> tuple[int, ...]:
        if self.closed:
            raise ValueError("stream is closed; reset before reuse")
        current = self.schedule.validate_frame(frame)
        result = tuple(
            current[k]
            if d == 0
            else (self.previous[k] if self.previous is not None else self.schedule.vocabulary(k))
            for k, d in enumerate(self.schedule.config.delays)
        )
        self.previous = current
        return result

    def finish(self) -> list[tuple[int, ...]]:
        if self.closed:
            raise ValueError("stream already closed")
        self.closed = True
        if self.previous is None:
            return []
        return [
            tuple(
                self.previous[k] if d else int(Sentinel.MISSING)
                for k, d in enumerate(self.schedule.config.delays)
            )
        ]


class FrameAssembler:
    """Restore one physical frame after the delay, keeping one previous step."""

    def __init__(self, schedule: TokenSchedule) -> None:
        self.schedule = schedule
        self.reset()

    def reset(self) -> None:
        self.previous: tuple[int, ...] | None = None

    def push(self, step: Sequence[int]) -> tuple[int, ...] | None:
        if len(step) != 17:
            raise ValueError("expected 17 channels")
        if self.previous is None:
            for k, d in enumerate(self.schedule.config.delays):
                token = step[k]
                if type(token) is not int or (
                    token != self.schedule.vocabulary(k)
                    if d
                    else not 0 <= token < self.schedule.vocabulary(k)
                ):
                    raise ValueError("invalid initial delayed step")
            self.previous = tuple(step)
            return None
        frame = self.schedule.validate_frame(
            tuple(
                self.previous[k] if d == 0 else step[k]
                for k, d in enumerate(self.schedule.config.delays)
            )
        )
        self.previous = tuple(step)
        return frame
