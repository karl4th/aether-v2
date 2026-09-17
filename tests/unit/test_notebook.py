import ast
import json
from pathlib import Path


def test_notebook_compiles_without_execution_or_saved_outputs() -> None:
    notebook = json.loads(Path("notebooks/aether_colab.ipynb").read_text())
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
    assert "--locked" in "\n".join(code)
    assert "--validate-only" in "\n".join(code)
