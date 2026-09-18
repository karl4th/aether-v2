import copy
import json

import pytest
from pydantic import ValidationError

from aether.dialogue_data import DialogueManifest, DialogueRecord


def record():
    return dict(
        id="one",
        conversation_id="conversation-one",
        split="train",
        audio_sha256="a" * 64,
        source_uri="fixture.wav",
        license="synthetic-test",
        consent_reference="synthetic-test",
        sample_rate=24000,
        frame_samples=1920,
        total_samples=7680,
        codec_contract_sha256="b" * 64,
        tokenizer_sha256="c" * 64,
        text_vocab_size=100,
        valid_audio_frames=[True] * 4,
        assistant_loss_frames=[False, False, True, True],
        turns=[
            dict(
                speaker_id="u",
                role="user",
                start_sample=0,
                end_sample=3840,
                text="Hello",
                text_tokens=[dict(token_id=2, frame_index=0, loss=False)],
            ),
            dict(
                speaker_id="a",
                role="assistant",
                start_sample=3840,
                end_sample=7680,
                text="Hi",
                text_tokens=[dict(token_id=3, frame_index=2, loss=True)],
            ),
        ],
    )


def validate(payload):
    return DialogueRecord.model_validate_json(json.dumps(payload))


def test_valid_timeline_and_cross_speaker_overlap():
    payload = record()
    validate(payload)
    payload["turns"][0]["end_sample"] = 5000
    validate(payload)  # Duplex overlap is legal between different speakers.


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(schema_version=2),
        lambda p: p.update(total_samples=True),
        lambda p: p.update(codec_contract_sha256="unknown"),
        lambda p: p.update(valid_audio_frames=[True]),
        lambda p: p.update(assistant_loss_frames=[True] * 4),
        lambda p: p["turns"][0]["text_tokens"][0].update(loss=True),
        lambda p: p["turns"][1]["text_tokens"][0].update(token_id=100),
        lambda p: p["turns"][1]["text_tokens"][0].update(frame_index=1),
        lambda p: p["turns"][1].update(end_sample=8000),
        lambda p: p["turns"][1].update(speaker_id="u"),
    ],
)
def test_invalid_supervision(mutation):
    payload = record()
    mutation(payload)
    with pytest.raises(ValidationError):
        validate(payload)


def test_partial_final_frame_cannot_contribute_to_loss():
    payload = record()
    payload["total_samples"] = 7000
    payload["turns"][1]["end_sample"] = 7000
    with pytest.raises(ValidationError):
        validate(payload)
    payload["valid_audio_frames"][-1] = False
    payload["assistant_loss_frames"][-1] = False
    validate(payload)


@pytest.mark.parametrize("leak", ["conversation", "audio", "speaker"])
def test_cross_split_leakage(leak):
    first = record()
    second = copy.deepcopy(first)
    second.update(id="two", split="test", conversation_id="two", audio_sha256="d" * 64)
    for turn in second["turns"]:
        turn["speaker_id"] += "-other"
    if leak == "conversation":
        second["conversation_id"] = first["conversation_id"]
    elif leak == "audio":
        second["audio_sha256"] = first["audio_sha256"]
    else:
        second["turns"][0]["speaker_id"] = first["turns"][0]["speaker_id"]
    with pytest.raises(ValidationError, match="leakage"):
        DialogueManifest.model_validate_json(
            json.dumps(dict(dataset_version="fixture-v1", records=[first, second]))
        )


def test_same_speaker_overlap_rejected():
    payload = record()
    extra = copy.deepcopy(payload["turns"][1])
    extra["start_sample"] = 5000
    payload["turns"].append(extra)
    with pytest.raises(ValidationError, match="overlap"):
        validate(payload)
