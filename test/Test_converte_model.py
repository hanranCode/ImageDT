"""Legacy test entry point; runs the maintained Python 3 regression suite."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "pytest",
                str(root / "tests" / "test_caffe_backends.py"),
            ],
            cwd=root,
        )
    )
