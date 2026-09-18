"""Pinned, bounded English speech data for manually authorized remote adaptation.

SLR31 is read speech, not dialogue. Transcripts have no word timing annotations.
No download, decoding, or filesystem mutation happens on import.
"""

from __future__ import annotations

import hashlib
import json
import math
import ssl
import tarfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SOURCE_URL = "https://www.openslr.org/31/"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
ATTRIBUTION = (
    "Mini LibriSpeech, OpenSLR SLR31, subset of LibriSpeech by Vassil Panayotov, "
    "Guoguo Chen, Daniel Povey and Sanjeev Khudanpur (2015). CC BY 4.0. "
    "Selected short utterances; no alteration of source audio."
)
# Official archive checksums: https://openslr.trmal.net/resources/31/md5sum.txt
ARCHIVES = {
    "train-clean-5": "5df7d4e78065366204ca6845bb08f490",
    "dev-clean-2": "6d7ab67ac6a1d2c993d050e16d61080d",
}
MAX_ARCHIVE_BYTES = 500_000_000
MAX_EXTRACTED_BYTES = 1_500_000_000
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_RETRY_BASE_SECONDS = 5.0
# Transient transport failures (TLS handshake timeouts, resets, DNS blips) are
# retried; a size-limit ValueError from inside the loop is not.
_RETRYABLE_DOWNLOAD_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError, ssl.SSLError)


@dataclass(frozen=True)
class SpeechExample:
    utterance_id: str
    speaker_id: str
    text: str
    audio_path: str
    sample_rate: int
    frames: int
    audio_sha256: str = ""

    @property
    def seconds(self) -> float:
        return self.frames / self.sample_rate


@dataclass(frozen=True)
class DatasetManifest:
    train: tuple[SpeechExample, ...]
    evaluation: tuple[SpeechExample, ...]
    schema_version: int = 1
    source_url: str = SOURCE_URL
    license_url: str = LICENSE_URL
    attribution: str = ATTRIBUTION

    def identity_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        for split in ("train", "evaluation"):
            payload[split] = [
                {key: value for key, value in record.items() if key != "audio_path"}
                for record in payload[split]
            ]
        payload["archive_checksums"] = dict(ARCHIVES)
        return payload

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.identity_payload(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def write(self, path: Path) -> None:
        payload = asdict(self)
        base = path.parent.resolve()
        for split in ("train", "evaluation"):
            payload[split] = [
                dict(record, audio_path=str(Path(record["audio_path"]).resolve().relative_to(base)))
                for record in payload[split]
            ]
        payload["sha256"] = self.sha256
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path, *, audio_root: Path | None = None) -> DatasetManifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fields = {
            "train",
            "evaluation",
            "schema_version",
            "source_url",
            "license_url",
            "attribution",
            "sha256",
        }
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError("invalid dataset manifest fields")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError("unsupported dataset manifest schema")
        for key, expected in (
            ("source_url", SOURCE_URL),
            ("license_url", LICENSE_URL),
            ("attribution", ATTRIBUTION),
        ):
            if payload[key] != expected:
                raise ValueError("dataset provenance mismatch")
        base = (audio_root or path.parent).resolve()
        splits = []
        seen: set[str] = set()
        for split in ("train", "evaluation"):
            records = payload[split]
            if not isinstance(records, list) or not 1 <= len(records) <= 10000:
                raise ValueError("invalid dataset split")
            examples = []
            for record in records:
                if not isinstance(record, dict) or set(record) != {
                    "utterance_id",
                    "speaker_id",
                    "text",
                    "audio_path",
                    "sample_rate",
                    "frames",
                    "audio_sha256",
                }:
                    raise ValueError("invalid speech example fields")
                for key in ("utterance_id", "speaker_id", "text", "audio_path", "audio_sha256"):
                    if not isinstance(record[key], str) or not record[key].strip():
                        raise ValueError("invalid speech example string")
                identifier = record["utterance_id"]
                parts = identifier.split("-")
                if (
                    len(parts) != 3
                    or not all(part.isdecimal() for part in parts)
                    or parts[0] != record["speaker_id"]
                    or identifier in seen
                ):
                    raise ValueError("invalid or duplicate utterance identity")
                seen.add(identifier)
                if (
                    type(record["sample_rate"]) is not int
                    or record["sample_rate"] != 16000
                    or type(record["frames"]) is not int
                    or not 0 < record["frames"] <= 30 * 16000
                ):
                    raise ValueError("invalid speech dimensions")
                relative = Path(record["audio_path"])
                audio = (base / relative).resolve()
                if relative.is_absolute() or not audio.is_relative_to(base):
                    raise ValueError("unsafe manifest audio path")
                digest = record["audio_sha256"]
                if (
                    len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                    or not audio.is_file()
                    or _sha256(audio) != digest
                ):
                    raise ValueError("audio checksum mismatch")
                examples.append(SpeechExample(**dict(record, audio_path=str(audio))))
            splits.append(tuple(examples))
        if {e.speaker_id for e in splits[0]} & {e.speaker_id for e in splits[1]}:
            raise ValueError("speaker leakage between training and evaluation")
        manifest = cls(splits[0], splits[1])
        if manifest.sha256 != payload["sha256"]:
            raise ValueError("dataset manifest checksum mismatch")
        return manifest


def load_manifest(path: Path, *, audio_root: Path | None = None) -> DatasetManifest:
    return DatasetManifest.read(path, audio_root=audio_root)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(path: Path) -> str:
    # MD5 is the publisher's integrity identifier, not a security signature.
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    try:
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                with (
                    urllib.request.urlopen(url, timeout=60) as response,
                    partial.open("wb") as output,
                ):
                    total = 0
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_ARCHIVE_BYTES:
                            raise ValueError("dataset archive exceeds download limit")
                        output.write(chunk)
                partial.replace(path)
                return
            except _RETRYABLE_DOWNLOAD_ERRORS:
                partial.unlink(missing_ok=True)
                if attempt == DOWNLOAD_ATTEMPTS:
                    raise
                time.sleep(DOWNLOAD_RETRY_BASE_SECONDS * attempt)
    finally:
        partial.unlink(missing_ok=True)


def _extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as source:
        members = []
        for member in source:
            members.append(member)
            if len(members) > 100_000:
                raise ValueError("dataset archive exceeds extraction limit")
        total = 0
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("unsafe dataset archive path")
            if not member.isfile() and not member.isdir():
                raise ValueError("dataset archive links and special files are forbidden")
            total += member.size
            if total > MAX_EXTRACTED_BYTES or len(members) > 100_000:
                raise ValueError("dataset archive exceeds extraction limit")
        source.extractall(destination, members=members, filter="data")


def _audio_info(path: Path) -> tuple[int, int]:
    import importlib

    sf: Any = importlib.import_module("soundfile")
    info = sf.info(str(path))
    if info.channels != 1 or info.samplerate != 16000:
        raise ValueError(f"unexpected source audio format: {path}")
    return int(info.samplerate), int(info.frames)


def _scan(
    root: Path, limit: int, max_seconds: float, min_seconds: float = 2.0
) -> tuple[tuple[SpeechExample, ...], set[str]]:
    examples: list[SpeechExample] = []
    speakers: set[str] = set()
    seen: set[str] = set()
    for transcript in sorted(root.rglob("*.trans.txt")):
        for line in transcript.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            identifier, separator, text = line.partition(" ")
            parts = identifier.split("-")
            if (
                not separator
                or not text.strip()
                or len(parts) != 3
                or not all(part.isdecimal() for part in parts)
            ):
                raise ValueError(f"invalid transcript entry: {transcript}")
            if identifier in seen:
                raise ValueError("duplicate utterance identifier")
            seen.add(identifier)
            speaker, chapter, _ = parts
            if transcript.parent.name != chapter or transcript.parent.parent.name != speaker:
                raise ValueError("speaker/chapter path disagrees with transcript")
            speakers.add(speaker)
            path = transcript.parent / f"{identifier}.flac"
            if not path.is_file():
                raise ValueError(f"missing source audio: {path}")
            if len(examples) >= limit:
                continue
            rate, frames = _audio_info(path)
            if frames <= 0:
                raise ValueError("empty source audio")
            if min_seconds <= frames / rate <= max_seconds:
                examples.append(
                    SpeechExample(
                        identifier, speaker, text.strip(), str(path), rate, frames, _sha256(path)
                    )
                )
    if not examples:
        raise ValueError(f"no suitable speech examples in {root}")
    return tuple(examples), speakers


def prepare_dataset(
    root: Path,
    *,
    authorize_download: Callable[[], None],
    max_train: int = 64,
    max_eval: int = 16,
    max_seconds: float = 5.0,
    min_seconds: float = 2.0,
) -> DatasetManifest:
    """Download verified source archives and select complete short utterances.

    Caller supplies its remote-runtime authorization gate, called before any I/O,
    even when cached. This function never infers remote authorization itself.
    Source 16 kHz audio is preserved; backend must resample to its codec rate.
    Official split speaker sets are checked in full, before returning a subset.
    """
    if type(max_train) is not int or type(max_eval) is not int:
        raise ValueError("example limits must be integers")
    if not 1 <= max_train <= 10000 or not 1 <= max_eval <= 10000:
        raise ValueError("example limits must be between 1 and 10000")
    if (
        type(max_seconds) not in (int, float)
        or not math.isfinite(max_seconds)
        or not 0 < max_seconds <= 30
    ):
        raise ValueError("max_seconds must be finite and in (0, 30]")
    if (
        type(min_seconds) not in (int, float)
        or not math.isfinite(min_seconds)
        or not 0 < min_seconds <= max_seconds
    ):
        raise ValueError("min_seconds must be positive and no greater than max_seconds")
    authorize_download()
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected: list[tuple[SpeechExample, ...]] = []
    speaker_sets: list[set[str]] = []
    for (split, checksum), limit in zip(ARCHIVES.items(), (max_train, max_eval), strict=True):
        archive = root / f"{split}.tar.gz"
        if not archive.exists():
            _download(f"https://openslr.trmal.net/resources/31/{archive.name}", archive)
        if archive.stat().st_size > MAX_ARCHIVE_BYTES or _digest(archive) != checksum:
            raise ValueError(f"dataset archive checksum mismatch: {archive}")
        # Re-extract verified bytes; never trust an unverified completion marker.
        _extract(archive, root)
        examples, speakers = _scan(root / "LibriSpeech" / split, limit, max_seconds, min_seconds)
        if len(examples) != limit:
            raise ValueError(
                f"{split}: requested {limit} examples in "
                f"[{min_seconds}, {max_seconds}] seconds, found {len(examples)}"
            )
        selected.append(examples)
        speaker_sets.append(speakers)
    if speaker_sets[0] & speaker_sets[1]:
        raise ValueError("speaker leakage between training and evaluation")
    manifest = DatasetManifest(selected[0], selected[1])
    manifest.write(root / "dataset.json")
    (root / "ATTRIBUTION.txt").write_text(
        f"{ATTRIBUTION}\n{SOURCE_URL}\n{LICENSE_URL}\n"
        + json.dumps(ARCHIVES, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest
