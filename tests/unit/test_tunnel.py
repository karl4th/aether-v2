"""cloudflared is mocked as a subprocess; no real network egress or binary run."""

import sys
from pathlib import Path

import pytest

from aether import tunnel


def fake_binary(tmp_path: Path, script: str) -> Path:
    # A short /bin/sh shebang avoids the historical shebang-length limit that
    # a long interpreter path (e.g. a deep uv-managed venv path) could hit.
    impl = tmp_path / "cloudflared_impl.py"
    impl.write_text(script)
    wrapper = tmp_path / "cloudflared"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{impl}" "$@"\n')
    wrapper.chmod(0o755)
    return wrapper


def test_rejects_invalid_port() -> None:
    with pytest.raises(ValueError, match="port"):
        with tunnel.cloudflare_tunnel(0):
            pass
    with pytest.raises(ValueError, match="port"):
        with tunnel.cloudflare_tunnel(70000):
            pass


def test_missing_binary_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tunnel.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="cloudflared is not installed"):
        with tunnel.cloudflare_tunnel(8080):
            pass


def test_yields_parsed_url_and_terminates_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = fake_binary(
        tmp_path,
        "import sys, time\n"
        "print('some log line')\n"
        "print('https://random-words.trycloudflare.com')\n"
        "sys.stdout.flush()\n"
        "time.sleep(60)\n",
    )
    monkeypatch.setattr(tunnel.shutil, "which", lambda name: str(binary))
    with tunnel.cloudflare_tunnel(8080, timeout=10) as url:
        assert url == "https://random-words.trycloudflare.com"


def test_process_exit_before_url_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = fake_binary(tmp_path, "print('boot failed')")
    monkeypatch.setattr(tunnel.shutil, "which", lambda name: str(binary))
    with pytest.raises(RuntimeError, match="exited before printing"):
        with tunnel.cloudflare_tunnel(8080, timeout=10):
            pass


def test_timeout_without_url_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = fake_binary(tmp_path, "import time\ntime.sleep(60)\n")
    monkeypatch.setattr(tunnel.shutil, "which", lambda name: str(binary))
    with pytest.raises(TimeoutError, match="did not print a tunnel URL"):
        with tunnel.cloudflare_tunnel(8080, timeout=0.3):
            pass


def test_terminates_stubborn_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = fake_binary(
        tmp_path,
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('https://ignore-term.trycloudflare.com', flush=True)\n"
        "time.sleep(60)\n",
    )
    monkeypatch.setattr(tunnel.shutil, "which", lambda name: str(binary))
    with tunnel.cloudflare_tunnel(8080, timeout=10) as url:
        assert url == "https://ignore-term.trycloudflare.com"
    # Exiting the context must not leave the process running.


def test_run_tunnel_prints_endpoints_and_blocks_until_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    binary = fake_binary(
        tmp_path,
        "import time\n"
        "print('https://random-words.trycloudflare.com', flush=True)\n"
        "time.sleep(60)\n",
    )
    monkeypatch.setattr(tunnel.shutil, "which", lambda name: str(binary))
    real_sleep = tunnel.time.sleep

    def raising_sleep(seconds: float) -> None:
        # Only intercept run_tunnel's own long wait; leave short internal
        # sleeps (e.g. inside subprocess.Popen.wait's poll loop) unaffected,
        # since patching time.sleep globally affects the whole process.
        if seconds >= 3600:
            raise KeyboardInterrupt
        real_sleep(seconds)

    monkeypatch.setattr(tunnel.time, "sleep", raising_sleep)
    tunnel.run_tunnel(8080)
    out = capsys.readouterr().out
    assert "TUNNEL_READY: https://random-words.trycloudflare.com" in out
    assert "wss://random-words.trycloudflare.com/v1/session" in out
