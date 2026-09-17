"""Build a source-only archive including uncommitted work; exclude private data."""

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

root = Path(__file__).resolve().parents[1]
files = [root / name for name in ("pyproject.toml", "uv.lock", ".python-version")]
for folder in ("src", "configs", "tests", "docs", "notebooks", "scripts"):
    files.extend(
        path
        for path in (root / folder).rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".json", ".md", ".ipynb"}
        and "__pycache__" not in path.parts
    )
files.sort()
if any(path.is_symlink() for path in files):
    raise ValueError("source archive may not contain symlinks")
output = root / "dist" / "aether-source.zip"
output.parent.mkdir(exist_ok=True)
entries = {path.relative_to(root).as_posix(): path.read_bytes() for path in files}
manifest = {
    "schema_version": 1,
    "revision": "uncommitted-source-snapshot",
    "files": {name: hashlib.sha256(data).hexdigest() for name, data in entries.items()},
}
with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
    for name, data in entries.items():
        info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = ZIP_DEFLATED
        archive.writestr(info, data)
    info = ZipInfo("source-manifest.json", date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    archive.writestr(info, json.dumps(manifest, sort_keys=True))
digest = hashlib.sha256(output.read_bytes()).hexdigest()
(root / "dist/aether-source.sha256").write_text(digest + "\n")
print(f"{output}\nSHA256={digest}")
