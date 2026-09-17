"""Replaceable streaming codec interface; no implementation or weights yet."""

from collections.abc import Sequence
from typing import Protocol, TypeVar

from aether.config import CodecConfig

StateT = TypeVar("StateT")


class AudioCodec(Protocol[StateT]):
    """Each session/direction owns its state; model weights may be shared.

    Encode consumes one mono frame; decode consumes eight codes of one physical
    frame. Implementations must reject BOS/sentinels and incompatible contracts.
    """

    @property
    def config(self) -> CodecConfig: ...

    def create_state(self) -> StateT: ...

    def encode(self, pcm: Sequence[float], state: StateT) -> tuple[int, ...]: ...

    def decode(self, codes: Sequence[int], state: StateT) -> tuple[float, ...]: ...

    def reset(self, state: StateT) -> None: ...
