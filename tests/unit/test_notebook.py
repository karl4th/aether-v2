import ast
import json
from pathlib import Path

import pytest

NOTEBOOKS = ["notebooks/aether_colab.ipynb", "notebooks/aether_live_v1.ipynb"]


def _load_code(path: str) -> tuple[dict, list[str]]:
    notebook = json.loads(Path(path).read_text())
    assert notebook["nbformat"] == 4
    ids = set()
    code = []
    for cell in notebook["cells"]:
        assert cell["id"] not in ids
        ids.add(cell["id"])
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
            source = "".join(cell["source"])
            compile(source, "notebook", "exec")
            code.append(source)
    return notebook, code


@pytest.mark.parametrize("path", NOTEBOOKS)
def test_notebook_compiles_without_execution_or_saved_outputs(path: str) -> None:
    _, code = _load_code(path)
    assert "--locked" in "\n".join(code)


def test_training_notebook_disables_training_by_default() -> None:
    _, code = _load_code("notebooks/aether_colab.ipynb")
    tree = ast.parse("\n".join(code))
    training = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "RUN_TRAINING" for target in node.targets
        )
    ]
    assert len(training) == 1
    assert isinstance(training[0].value, ast.Constant) and training[0].value.value is False
    assert "--validate-only" in "\n".join(code)


def test_live_notebook_requires_confirmation_and_never_trains() -> None:
    _, code = _load_code("notebooks/aether_live_v1.ipynb")
    joined = "\n".join(code)
    tree = ast.parse(joined)
    confirmed = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "CONFIRM_REMOTE_PAID_COLAB"
            for target in node.targets
        )
    ]
    assert len(confirmed) == 1
    assert isinstance(confirmed[0].value, ast.Constant) and confirmed[0].value.value is False
    # The permit is built with a hard-coded literal, not a variable that could be flipped.
    assert '"allow_training": False' in joined
    assert '"serve"' in joined
    assert '"tunnel"' in joined
