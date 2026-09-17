from pathlib import Path

import pytest

from aether.config import load_config
from aether.tokens import DelayedStream, FrameAssembler, Sentinel, TokenSchedule


def test_unique_frames_batch_stream_and_inverse() -> None:
    schedule = TokenSchedule(load_config(Path("configs/model/tiny.json")).model)
    frames = [tuple(t for _ in range(17)) for t in range(12)]
    delayed = schedule.shift(frames)
    assert delayed[0] == (0, 0, *([32] * 7), 0, *([32] * 7))
    assert delayed[4] == (4, 4, *([3] * 7), 4, *([3] * 7))
    assert delayed[-1] == (-1, -1, *([11] * 7), -1, *([11] * 7))
    assert schedule.restore(delayed, len(frames)) == frames
    assembler = FrameAssembler(schedule)
    assert [frame for step in delayed if (frame := assembler.push(step)) is not None] == frames
    assembler.reset()
    assert assembler.push(delayed[0]) is None
    stream = DelayedStream(schedule)
    assert [stream.push(f) for f in frames] + stream.finish() == delayed
    stream.reset()
    assert stream.push(frames[0]) == delayed[0]
    assert schedule.shift([]) == []
    assert schedule.restore([], 0) == []


def test_masks_and_decoder_reject_service_tokens() -> None:
    schedule = TokenSchedule(load_config("configs/model/tiny.json").model)
    for channel in range(17):
        for token in (int(Sentinel.MISSING), int(Sentinel.PENDING), schedule.vocabulary(channel)):
            assert not schedule.loss_mask(token, channel)
            assert schedule.lookup_mask(token, channel) == (token >= 0)
            frame = [0] * 17
            frame[channel] = token
            with pytest.raises(ValueError):
                schedule.decoder_codes(frame)
    assert not schedule.loss_mask(3, 0)
    assert schedule.loss_mask(0, 0)
    assert schedule.loss_mask(3, 1)
    assert not schedule.loss_mask(3, 9)
    assert schedule.decoder_codes([1] * 17) == (1,) * 8
    with pytest.raises(ValueError):
        schedule.restore([[0] * 17], 1)


def test_independent_streams_and_invalid_input_atomicity() -> None:
    schedule = TokenSchedule(load_config("configs/model/tiny.json").model)
    first, second = DelayedStream(schedule), DelayedStream(schedule)
    first.push([1] * 17)
    second.push([2] * 17)
    with pytest.raises(ValueError):
        first.push([True] * 17)
    assert first.push([3] * 17)[2] == 1
    assert second.push([4] * 17)[2] == 2
    first.finish()
    with pytest.raises(ValueError):
        first.push([0] * 17)
