"""Check that unused workflows do not delay a fresh app process."""

from pathlib import Path
import subprocess
import sys


def test_workflow_registry_does_not_import_scientific_dependencies() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import app.pages; "
            "loaded = set(sys.modules); "
            "assert not loaded.intersection({'numpy', 'pandas', 'scipy', 'matplotlib'}); "
            "assert not any(name.startswith('app.pages.') for name in loaded)",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
