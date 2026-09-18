"""Cloudflare quick tunnel: exposes a local port with no account or DNS setup.

Used only to reach a live session server running inside a remote Colab VM,
which has no public IP of its own. The tunnel forwards an ephemeral
`https://<random>.trycloudflare.com` hostname to `127.0.0.1:<port>`; nothing
else on the machine is exposed. Quick tunnels are unauthenticated and
disposable: closing the process ends the tunnel and the hostname stops
resolving. This is a development convenience, not a production ingress.
"""

from __future__ import annotations

import queue
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

_URL_PATTERN = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
_BINARY_CANDIDATES = ("cloudflared",)


def _binary() -> str:
    for name in _BINARY_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError(
        "cloudflared is not installed; install it before requesting a tunnel "
        "(e.g. `apt-get install cloudflared` or download the official binary)"
    )


@contextmanager
def cloudflare_tunnel(port: int, *, timeout: float = 30.0) -> Iterator[str]:
    """Start a quick tunnel to localhost:port; yields the public https URL.

    The subprocess is terminated when the context exits, including on error.
    """
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be an integer between 1 and 65535")
    process = subprocess.Popen(
        [_binary(), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        url = _read_url(process, timeout)
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def _read_url(process: subprocess.Popen[str], timeout: float) -> str:
    """Read lines on a daemon thread so a silent process cannot block past `timeout`.

    Mixing `select()` with a buffered text stream is unsafe here: a single
    `readline()` call can pull more than one line into the stream's internal
    buffer, after which the OS-level file descriptor looks empty to `select`
    even though a line is already waiting to be read. A background reader
    plus a queue sidesteps that mismatch entirely.
    """
    assert process.stdout is not None
    lines: queue.Queue[str | None] = queue.Queue()

    def pump() -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                lines.put(line)
        finally:
            lines.put(None)  # EOF sentinel.

    threading.Thread(target=pump, daemon=True).start()
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.terminate()
            raise TimeoutError(f"cloudflared did not print a tunnel URL within {timeout:.0f}s")
        try:
            line = lines.get(timeout=min(remaining, 0.5))
        except queue.Empty:
            if process.poll() is not None:
                raise RuntimeError("cloudflared exited before printing a tunnel URL") from None
            continue
        if line is None:
            raise RuntimeError("cloudflared exited before printing a tunnel URL")
        match = _URL_PATTERN.search(line)
        if match:
            return match.group(0)


def run_tunnel(port: int) -> None:
    """Foreground helper for the CLI: print the URL, then block until interrupted."""
    with cloudflare_tunnel(port) as url:
        print(f"TUNNEL_READY: {url}")
        print(f"Live session endpoint: {url.replace('https://', 'wss://')}/v1/session")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
