import io
import json
import tarfile
from pathlib import Path

import pytest

from aether import dataset


def archive(path: Path, entries: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as output:
        for name, data in entries.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            output.addfile(member, io.BytesIO(data))


def fixture_archives(root: Path, monkeypatch: pytest.MonkeyPatch, *, leak: bool = False) -> None:
    root.mkdir()
    checksums = {}
    for split, speaker in (("train-clean-5", "1"), ("dev-clean-2", "1" if leak else "2")):
        prefix = f"LibriSpeech/{split}/{speaker}/10"
        entries = {
            f"{prefix}/{speaker}-10.trans.txt": (
                f"{speaker}-10-0001 SHORT TEXT\n{speaker}-10-0002 LONG TEXT\n"
            ).encode(),
            f"{prefix}/{speaker}-10-0001.flac": b"short",
            f"{prefix}/{speaker}-10-0002.flac": b"long",
        }
        path = root / f"{split}.tar.gz"
        archive(path, entries)
        checksums[split] = dataset._digest(path)
    monkeypatch.setattr(dataset, "ARCHIVES", checksums)
    monkeypatch.setattr(
        dataset, "_audio_info", lambda p: (16000, 32000 if "0001" in p.name else 320000)
    )


def test_remote_gate_before_io(tmp_path: Path) -> None:
    root = tmp_path / "untouched"

    def deny() -> None:
        raise RuntimeError("remote only")

    with pytest.raises(RuntimeError, match="remote only"):
        dataset.prepare_dataset(root, authorize_download=deny)
    assert not root.exists()


def test_verified_selection_and_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "data"
    fixture_archives(root, monkeypatch)
    gates = []
    result = dataset.prepare_dataset(
        root,
        authorize_download=lambda: gates.append(True),
        max_train=1,
        max_eval=1,
        max_seconds=4.0,
    )
    assert gates == [True]
    assert len(result.train) == len(result.evaluation) == 1
    assert result.train[0].seconds == 2
    assert result.train[0].speaker_id != result.evaluation[0].speaker_id
    saved = json.loads((root / "dataset.json").read_text())
    assert saved["train"][0]["text"] == "SHORT TEXT"
    assert saved["license_url"] == dataset.LICENSE_URL
    assert "CC BY 4.0" in (root / "ATTRIBUTION.txt").read_text()


def test_speaker_leakage_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "data"
    fixture_archives(root, monkeypatch, leak=True)
    with pytest.raises(ValueError, match="speaker leakage"):
        dataset.prepare_dataset(root, authorize_download=lambda: None, max_train=1, max_eval=1)
    assert not (root / "dataset.json").exists()


def test_corrupt_archive_rejected_before_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "data"
    fixture_archives(root, monkeypatch)
    (root / "train-clean-5.tar.gz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        dataset.prepare_dataset(root, authorize_download=lambda: None, max_train=1, max_eval=1)
    assert not (root / "LibriSpeech").exists()


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape"])
def test_archive_path_escape_rejected(tmp_path: Path, name: str) -> None:
    source = tmp_path / "bad.tar.gz"
    archive(source, {name: b"x"})
    with pytest.raises(ValueError, match="unsafe"):
        dataset._extract(source, tmp_path / "output")


def test_archive_link_rejected(tmp_path: Path) -> None:
    source = tmp_path / "bad.tar.gz"
    with tarfile.open(source, "w:gz") as output:
        member = tarfile.TarInfo("link")
        member.type = tarfile.SYMTYPE
        member.linkname = "target"
        output.addfile(member)
    with pytest.raises(ValueError, match="links"):
        dataset._extract(source, tmp_path / "output")


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), 31])
def test_duration_bounds_before_gate(tmp_path: Path, seconds: float) -> None:
    with pytest.raises(ValueError, match="max_seconds"):
        dataset.prepare_dataset(
            tmp_path, authorize_download=lambda: pytest.fail("gate called"), max_seconds=seconds
        )


def test_missing_audio_rejected(tmp_path: Path) -> None:
    folder = tmp_path / "1" / "10"
    folder.mkdir(parents=True)
    (folder / "1-10.trans.txt").write_text("1-10-0001 HELLO\n")
    with pytest.raises(ValueError, match="missing source audio"):
        dataset._scan(tmp_path, 1, 4)


def test_download_bound_cleans_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dataset, "MAX_ARCHIVE_BYTES", 3)
    monkeypatch.setattr(dataset.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"large"))
    destination = tmp_path / "archive.tar.gz"
    with pytest.raises(ValueError, match="download limit"):
        dataset._download("https://example.invalid/archive", destination)
    assert not destination.exists()
    assert not destination.with_suffix(".gz.partial").exists()


def test_download_retries_transient_transport_errors_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dataset, "DOWNLOAD_RETRY_BASE_SECONDS", 0.0)
    attempts = []

    def flaky_urlopen(*_args: object, **_kwargs: object) -> io.BytesIO:
        attempts.append(1)
        if len(attempts) < 3:
            raise dataset.urllib.error.URLError("<urlopen error _ssl.c:993: handshake timed out>")
        return io.BytesIO(b"payload")

    monkeypatch.setattr(dataset.urllib.request, "urlopen", flaky_urlopen)
    destination = tmp_path / "archive.tar.gz"
    dataset._download("https://example.invalid/archive", destination)
    assert len(attempts) == 3
    assert destination.read_bytes() == b"payload"
    assert not destination.with_suffix(".gz.partial").exists()


def test_download_gives_up_after_max_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dataset, "DOWNLOAD_RETRY_BASE_SECONDS", 0.0)
    attempts = []

    def always_fails(*_args: object, **_kwargs: object) -> io.BytesIO:
        attempts.append(1)
        raise dataset.urllib.error.URLError("<urlopen error _ssl.c:993: handshake timed out>")

    monkeypatch.setattr(dataset.urllib.request, "urlopen", always_fails)
    destination = tmp_path / "archive.tar.gz"
    with pytest.raises(dataset.urllib.error.URLError):
        dataset._download("https://example.invalid/archive", destination)
    assert len(attempts) == dataset.DOWNLOAD_ATTEMPTS
    assert not destination.exists()
    assert not destination.with_suffix(".gz.partial").exists()


def test_missing_archive_downloads_only_after_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "data"
    fixture_archives(root, monkeypatch)
    source = root / "train-clean-5.tar.gz"
    contents = source.read_bytes()
    source.unlink()
    events = []

    def download(url: str, path: Path) -> None:
        events.append("download")
        assert url.startswith("https://openslr.elda.org/resources/31/")
        path.write_bytes(contents)

    monkeypatch.setattr(dataset, "_download", download)
    dataset.prepare_dataset(
        root, authorize_download=lambda: events.append("gate"), max_train=1, max_eval=1
    )
    assert events == ["gate", "download"]


def test_manifest_roundtrip_relocation_and_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shutil

    root = tmp_path / "data"
    fixture_archives(root, monkeypatch)
    original = dataset.prepare_dataset(
        root, authorize_download=lambda: None, max_train=1, max_eval=1
    )
    assert dataset.load_manifest(root / "dataset.json") == original
    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    loaded = dataset.load_manifest(relocated / "dataset.json")
    assert loaded.sha256 == original.sha256
    assert loaded.train[0].audio_path != original.train[0].audio_path
    Path(loaded.train[0].audio_path).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="audio checksum"):
        dataset.load_manifest(relocated / "dataset.json")


@pytest.mark.parametrize("change", ["unknown", "escape", "bool", "text", "schema"])
def test_manifest_malformed_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    root = tmp_path / "data"
    fixture_archives(root, monkeypatch)
    dataset.prepare_dataset(root, authorize_download=lambda: None, max_train=1, max_eval=1)
    path = root / "dataset.json"
    payload = json.loads(path.read_text())
    if change == "unknown":
        payload["unknown"] = True
    elif change == "escape":
        payload["train"][0]["audio_path"] = "../escape.flac"
    elif change == "bool":
        payload["train"][0]["frames"] = True
    elif change == "text":
        payload["train"][0]["text"] = "changed transcript"
    else:
        payload["schema_version"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        dataset.load_manifest(path)


def test_insufficient_short_examples_fails_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "data"
    fixture_archives(root, monkeypatch)
    with pytest.raises(ValueError, match="requested 2 examples.*found 1"):
        dataset.prepare_dataset(root, authorize_download=lambda: None, max_train=2, max_eval=1)


def test_extraction_size_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "large.tar.gz"
    archive(source, {"large": b"1234"})
    monkeypatch.setattr(dataset, "MAX_EXTRACTED_BYTES", 3)
    with pytest.raises(ValueError, match="extraction limit"):
        dataset._extract(source, tmp_path / "output")
    assert not (tmp_path / "output" / "large").exists()
