"""Dialogue supervision contracts, distinct from unaligned read-speech adaptation."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from aether.config import StrictConfig

Name = Annotated[str, Field(min_length=1)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Index = Annotated[int, Field(ge=0)]


class AlignedToken(StrictConfig):
    token_id: Index
    frame_index: Index
    loss: bool


class DialogueTurn(StrictConfig):
    speaker_id: Name
    role: Literal["user", "assistant"]
    start_sample: Index
    end_sample: Annotated[int, Field(gt=0)]
    text: Name
    text_tokens: tuple[AlignedToken, ...]

    @model_validator(mode="after")
    def span(self) -> Self:
        if self.end_sample <= self.start_sample:
            raise ValueError("turn end must follow start")
        if any(token.loss for token in self.text_tokens) and self.role != "assistant":
            raise ValueError("user tokens cannot be supervision targets")
        frames = [token.frame_index for token in self.text_tokens]
        if frames != sorted(frames):
            raise ValueError("text alignment must be chronological")
        return self


class DialogueRecord(StrictConfig):
    id: Name
    conversation_id: Name
    split: Literal["train", "validation", "test"]
    audio_sha256: Digest
    source_uri: Name
    license: Name
    consent_reference: Name
    sample_rate: Annotated[int, Field(gt=0)]
    frame_samples: Annotated[int, Field(gt=0)]
    total_samples: Annotated[int, Field(gt=0)]
    codec_contract_sha256: Digest
    tokenizer_sha256: Digest
    text_vocab_size: Annotated[int, Field(gt=0)]
    # True = real codec frame; final padding is always false.
    valid_audio_frames: tuple[bool, ...]
    assistant_loss_frames: tuple[bool, ...]
    turns: tuple[DialogueTurn, ...]

    @model_validator(mode="after")
    def alignment(self) -> Self:
        count = (self.total_samples + self.frame_samples - 1) // self.frame_samples
        if len(self.valid_audio_frames) != count or len(self.assistant_loss_frames) != count:
            raise ValueError("audio mask length does not match timeline")
        if any(
            loss and not valid
            for loss, valid in zip(self.assistant_loss_frames, self.valid_audio_frames, strict=True)
        ):
            raise ValueError("padding cannot contribute to loss")
        if self.total_samples % self.frame_samples and self.valid_audio_frames[-1]:
            raise ValueError("partial final frame must be masked")
        speakers: dict[str, tuple[str, int]] = {}
        starts = [turn.start_sample for turn in self.turns]
        if not starts or starts != sorted(starts):
            raise ValueError("turns must be nonempty and ordered")
        for turn in self.turns:
            previous = speakers.get(turn.speaker_id)
            if previous and (previous[0] != turn.role or previous[1] > turn.start_sample):
                raise ValueError("speaker role mismatch or same-speaker overlap")
            speakers[turn.speaker_id] = (turn.role, turn.end_sample)
            if turn.end_sample > self.total_samples:
                raise ValueError("turn exceeds audio")
            for token in turn.text_tokens:
                if token.token_id >= self.text_vocab_size or not (
                    turn.start_sample // self.frame_samples
                    <= token.frame_index
                    < (turn.end_sample + self.frame_samples - 1) // self.frame_samples
                ):
                    raise ValueError("text token outside vocabulary or aligned turn")
                if token.loss and not self.assistant_loss_frames[token.frame_index]:
                    raise ValueError("text loss requires valid assistant audio frame")
        for frame, loss in enumerate(self.assistant_loss_frames):
            if loss and not any(
                turn.role == "assistant"
                and turn.start_sample <= frame * self.frame_samples
                and (frame + 1) * self.frame_samples <= turn.end_sample
                for turn in self.turns
            ):
                raise ValueError("audio loss outside complete assistant frame")
        return self


class DialogueManifest(StrictConfig):
    dataset_version: Name
    records: tuple[DialogueRecord, ...]

    @model_validator(mode="after")
    def leakage(self) -> Self:
        identities: set[str] = set()
        owners: dict[tuple[str, str], str] = {}
        contracts = {(r.codec_contract_sha256, r.tokenizer_sha256) for r in self.records}
        if not self.records or len(contracts) != 1:
            raise ValueError("nonempty data with a single codec/tokenizer contract required")
        for record in self.records:
            if record.id in identities:
                raise ValueError("duplicate record")
            identities.add(record.id)
            keys = [("conversation", record.conversation_id), ("audio", record.audio_sha256)]
            keys += [("speaker", turn.speaker_id) for turn in record.turns]
            for key in keys:
                if key in owners and owners[key] != record.split:
                    raise ValueError("conversation/audio/speaker leakage across splits")
                owners[key] = record.split
        return self
