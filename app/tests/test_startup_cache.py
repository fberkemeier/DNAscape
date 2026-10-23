"""Local bytecode retains Python's source validation and import behavior."""

import _io
import os
import py_compile
import sys
from pathlib import Path

from app import _startup_cache as startup


def value(loader):
    namespace = {}
    exec(loader.get_code("app.cache_probe"), namespace)
    return namespace["value"]


def test_existing_bytecode_is_seeded_once_then_uses_only_the_local_cache(tmp_path, monkeypatch):
    source = tmp_path / "cache_probe.py"
    source.write_text("value = 7\n")
    original_cache = py_compile.compile(str(source), doraise=True)
    local_cache = tmp_path / "local" / "cache_probe.pyc"
    loader = startup._CachedSourceLoader("app.cache_probe", str(source), local_cache)
    assert value(loader) == 7
    assert local_cache.read_bytes() == Path(original_cache).read_bytes()
    open_code = _io.open_code
    def local_only(path):
        assert str(path) not in {str(source), original_cache}, "Warm imports must avoid synced code reads"
        return open_code(path)
    monkeypatch.setattr(_io, "open_code", local_only)
    assert value(loader) == 7


def test_changed_source_and_corrupt_bytecode_are_revalidated(tmp_path):
    source = tmp_path / "cache_probe.py"
    source.write_text("value = 1\n")
    cache = tmp_path / "local" / "cache_probe.pyc"
    loader = startup._CachedSourceLoader("app.cache_probe", str(source), cache)
    assert value(loader) == 1
    previous = source.stat().st_mtime
    source.write_text("value = 2\n")
    os.utime(source, (previous + 5, previous + 5))
    assert value(loader) == 2
    cache.write_bytes(b"invalid bytecode")
    assert value(loader) == 2


def test_no_bytecode_flag_prevents_seeding_or_writing_a_cache(tmp_path, monkeypatch):
    source = tmp_path / "cache_probe.py"
    source.write_text("value = 7\n")
    py_compile.compile(str(source), doraise=True)
    cache = tmp_path / "local" / "cache_probe.pyc"
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    loader = startup._CachedSourceLoader("app.cache_probe", str(source), cache)
    assert value(loader) == 7
    assert not cache.exists()


def test_finder_changes_only_project_sources_and_preserves_origin(tmp_path):
    project = tmp_path / "project"
    package = project / "app"
    package.mkdir(parents=True)
    source = package / "cache_probe.py"
    source.write_text("value = 7\n")
    finder = startup._LocalCacheFinder(project, tmp_path / "local")
    spec = finder.find_spec("app.cache_probe", [str(package)])
    assert isinstance(spec.loader, startup._CachedSourceLoader)
    assert spec.origin == str(source)
    assert Path(spec.cached).is_relative_to(tmp_path / "local")
    assert finder.find_spec("numpy") is None
    assert finder.find_spec("app.main", [str(Path(startup.__file__).parent)]) is None


def test_installation_is_idempotent_optional_and_tolerates_read_only_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "meta_path", [item for item in sys.meta_path if not isinstance(item, startup._LocalCacheFinder)])
    monkeypatch.setattr(sys, "pycache_prefix", None)
    monkeypatch.delenv("DNASCAPE_BYTECODE_CACHE", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    cache = startup.install_local_bytecode_cache()
    assert cache.is_relative_to(tmp_path)
    assert startup.install_local_bytecode_cache() == cache
    assert sum(isinstance(item, startup._LocalCacheFinder) for item in sys.meta_path) == 1
    monkeypatch.setattr(sys, "pycache_prefix", str(tmp_path / "explicit"))
    assert startup.install_local_bytecode_cache() is None
    monkeypatch.setattr(sys, "pycache_prefix", None)
    monkeypatch.setenv("DNASCAPE_BYTECODE_CACHE", "off")
    assert startup.install_local_bytecode_cache() is None
    monkeypatch.delenv("DNASCAPE_BYTECODE_CACHE")
    monkeypatch.setattr(sys, "meta_path", [item for item in sys.meta_path if not isinstance(item, startup._LocalCacheFinder)])
    def read_only(*args, **kwargs):
        raise PermissionError("read-only cache")
    monkeypatch.setattr(Path, "mkdir", read_only)
    assert startup.install_local_bytecode_cache() is None


def test_asset_cache_reuses_text_and_refreshes_after_edits_or_corruption(tmp_path, monkeypatch):
    source = tmp_path / "styles.css"
    source.write_text("body {color: red}", encoding="utf-8")
    finder = startup._LocalCacheFinder(tmp_path, tmp_path / "local")
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    assert startup.read_asset_text(source) == "body {color: red}"
    original = Path.read_text
    def local_only(path, **kwargs):
        assert path != source, "Warm asset reads must avoid the synced source"
        return original(path, **kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "read_text", local_only)
        assert startup.read_asset_text(source) == "body {color: red}"
    source.write_text("body {color: blue}", encoding="utf-8")
    assert startup.read_asset_text(source) == "body {color: blue}"
    next((finder.cache / "assets").iterdir()).write_text("broken cache")
    assert startup.read_asset_text(source) == "body {color: blue}"


def test_assets_fall_back_when_cache_cannot_be_written(tmp_path, monkeypatch):
    source = tmp_path / "styles.css"
    source.write_text("readable asset", encoding="utf-8")
    finder = startup._LocalCacheFinder(tmp_path, tmp_path / "local")
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    def read_only(*args, **kwargs):
        raise PermissionError("read-only cache")
    monkeypatch.setattr(Path, "mkdir", read_only)
    assert startup.read_asset_text(source) == "readable asset"
