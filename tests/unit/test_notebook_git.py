"""Exercise only the notebook Git helper against a local repository, never Colab."""

import ast
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest


def test_git_checkout_is_pinned_and_repeatable(tmp_path: Path) -> None:
    notebook = json.loads(Path("notebooks/aether_colab.ipynb").read_text())
    functions = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            functions.extend(
                node
                for node in ast.parse("".join(cell["source"])).body
                if isinstance(node, ast.FunctionDef) and node.name == "checkout_source"
            )
    assert len(functions) == 1
    namespace = {"Path": Path, "os": os, "subprocess": subprocess, "tempfile": tempfile}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "git-helper", "exec"), namespace)
    checkout = namespace["checkout_source"]
    repo = tmp_path / "origin"
    repo.mkdir()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-b", "main")
    for name in ("pyproject.toml", "uv.lock", ".python-version"):
        (repo / name).write_text("fixture-v1\n")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "v1")
    first_sha = git("rev-parse", "HEAD")
    first, actual = checkout(str(repo), "main", tmp_path)
    assert actual == first_sha
    (repo / "uv.lock").write_text("fixture-v2\n")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "v2")
    pinned, actual = checkout(str(repo), first_sha, tmp_path)
    assert actual == first_sha
    assert pinned != first
    assert (pinned / "uv.lock").read_text() == "fixture-v1\n"
    latest, actual = checkout(str(repo), "main", tmp_path)
    assert actual == git("rev-parse", "HEAD")
    assert (latest / "uv.lock").read_text() == "fixture-v2\n"
    with pytest.raises(ValueError, match="Git branch"):
        checkout(str(repo), "--upload-pack=bad", tmp_path)
    with pytest.raises(subprocess.CalledProcessError):
        checkout(str(repo), "missing-branch", tmp_path)
