"""Package workflows stay installable, headless, and shared with the app."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

APP_ALIASES = (
    ("batch_workflows", "batch"),
    ("bigwig_service", "bigwig"),
    ("data_conversion_service", "conversion"),
    ("export_settings", "export_settings"),
    ("mapping_catalog", "mapping_catalog"),
    ("mapping_defaults", "mapping_defaults"),
    ("mapping_exports", "mapping_exports"),
    ("mapping_service", "mapping"),
    ("mapping_uploads", "uploads"),
    ("result_naming", "result_naming"),
    ("simulation_exports", "exports"),
    ("simulation_service", "simulation"),
)


def _run_python(source: str, *, cwd: Path, import_path: Path | None = None) -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MPLCONFIGDIR"] = str(cwd / "matplotlib")
    bootstrap = ""
    if import_path is not None:
        bootstrap = f"import sys; sys.path.insert(0, {str(import_path)!r})\n"
    result = subprocess.run(
        [sys.executable, "-I", "-c", bootstrap + textwrap.dedent(source)],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_workflow_namespace_keeps_scientific_dependencies_lazy(tmp_path: Path) -> None:
    _run_python(
        """
        import sys
        import dnascape
        workflows = dnascape.workflows
        assert workflows.__name__ == 'dnascape.workflows'
        assert 'run_mapping' in dir(workflows)
        assert 'run_mapping_recipe' in dir(workflows)
        assert not {'numpy', 'pandas', 'scipy', 'matplotlib', 'nicegui', 'app'} & sys.modules.keys()
        """,
        cwd=tmp_path,
        import_path=ROOT,
    )


def test_every_public_workflow_imports_without_gui_or_app(tmp_path: Path) -> None:
    _run_python(
        """
        import importlib.abc
        import sys
        class RejectUI(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'app', 'nicegui'}:
                    raise AssertionError('Headless workflow tried to import ' + fullname)
        sys.meta_path.insert(0, RejectUI())
        import dnascape.workflows as workflows
        for name in workflows.__all__:
            assert callable(getattr(workflows, name)), name
        assert not {'app', 'nicegui'} & sys.modules.keys()
        """,
        cwd=tmp_path,
        import_path=ROOT,
    )


@pytest.mark.parametrize(("app_module", "package_module"), APP_ALIASES)
def test_app_aliases_share_the_package_module(app_module: str, package_module: str) -> None:
    app = importlib.import_module(f"app.{app_module}")
    package = importlib.import_module(f"dnascape.workflows.{package_module}")
    assert app is package


def test_app_overrides_reach_the_shared_mapping_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.mapping_service as app_mapping
    from dnascape.workflows import MappingRequest, mapping_route, validate_mapping_request

    monkeypatch.setattr(app_mapping, "MAX_INPUT_VALUES", 2)
    request = MappingRequest(
        route=mapping_route("firing_and_speed_to_timing_deterministic"),
        inputs={"firing_rate": [0.001, 0.002, 0.003], "fork_speed": 1.4},
        input_forms={"firing_rate": "x", "fork_speed": "scalar"},
        settings={"resolution": 1.0, "neighbourhood_range": 10.0},
    )
    with pytest.raises(ValueError, match="interactive limits are 2 values"):
        validate_mapping_request(request)


def test_installed_wheel_runs_mapping_and_raw_exports_without_app(tmp_path: Path) -> None:
    # Build outside the checkout so an editable install or an untracked source
    # directory cannot conceal omissions in setuptools package discovery.
    source = tmp_path / "source"
    source.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(ROOT / name, source / name)
    shutil.copytree(
        ROOT / "dnascape",
        source / "dnascape",
        ignore=shutil.ignore_patterns("tests", "__pycache__", "*.pyc"),
    )
    dist = tmp_path / "dist"
    dist.mkdir()
    _run_python(
        f"from setuptools.build_meta import build_wheel; build_wheel({str(dist)!r})",
        cwd=source,
    )
    wheels = tuple(dist.glob("*.whl"))
    assert len(wheels) == 1
    installed = tmp_path / "installed"
    installation = subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile",
            "--disable-pip-version-check", "--target", str(installed), str(wheels[0]),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert installation.returncode == 0, installation.stdout + installation.stderr
    assert not (installed / "app").exists()
    assert not (installed / "dnascape" / "tests").exists()
    _run_python(
        """
        import importlib.abc
        import sys
        from pathlib import Path
        class RejectUI(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'app', 'nicegui'}:
                    raise AssertionError('Installed workflow tried to import ' + fullname)
        sys.meta_path.insert(0, RejectUI())
        import dnascape.workflows as workflows
        assert Path(workflows.__file__).parent.parent.parent == Path.cwd() / 'installed'
        for name in workflows.__all__:
            assert callable(getattr(workflows, name)), name
        request = workflows.MappingRequest(
            route=workflows.mapping_route('firing_and_speed_to_timing_deterministic'),
            inputs={'firing_rate': 0.001, 'fork_speed': 1.4},
            input_forms={'firing_rate': 'scalar', 'fork_speed': 'scalar'},
            settings={'resolution': 1.0, 'neighbourhood_range': 10.0},
            region_start_kb=0, region_end_kb=4,
        )
        result = workflows.run_mapping(request)
        assert result.outputs[result.primary_output].size == 4
        artifact = workflows.build_mapping_raw_export(result, result.primary_output, 'npy')
        assert artifact.data.startswith(b'\\x93NUMPY')
        assert not {'app', 'nicegui'} & sys.modules.keys()
        """,
        cwd=tmp_path,
        import_path=installed,
    )
