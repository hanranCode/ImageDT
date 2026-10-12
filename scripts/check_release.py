"""Check release metadata and source distribution completeness without uploading."""

import hashlib
import json
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


def check(directory):
    wheels = list(Path(directory).glob("imagedt-*.whl"))
    sources = list(Path(directory).glob("imagedt-*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("Expected exactly one wheel and one source distribution")
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
        metadata_path = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(wheel.read(metadata_path))
        assert metadata["Name"] == "imagedt"
        assert metadata["Requires-Python"] == ">=3.10"
        assert (
            "Tag: py3-none-any"
            in wheel.read(
                next(name for name in names if name.endswith(".dist-info/WHEEL"))
            ).decode()
        )
        assert not any(name.endswith(".pyc") for name in names)
        for name in names:
            if name.endswith(".py"):
                compile(wheel.read(name), name, "exec")
    with tarfile.open(sources[0]) as source:
        names = source.getnames()
        for required in [
            "tests/fixtures/legacy_api.json",
            "pyproject.toml",
            "docs/migration.md",
            "README.md",
        ]:
            assert any(name.endswith("/" + required) for name in names), (
                f"Missing {required}"
            )
    return {
        "version": metadata["Version"],
        "artifacts": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for path in [wheels[0], sources[0]]
        ],
        "metadata_and_source_checks": "passed",
    }


if __name__ == "__main__":
    import sys

    print(json.dumps(check(sys.argv[1] if len(sys.argv) > 1 else "dist"), indent=2))
