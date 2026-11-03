"""Keep application bytecode out of slow synced or network source folders.

DNAscape sources and dependencies in a project-local virtual environment use
this cache. Python still checks source timestamps, sizes and bytecode versions.
Source paths, resource lookup and tracebacks stay unchanged. On Windows,
slow networking extensions also use validated, disposable local copies.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from importlib.machinery import ExtensionFileLoader, PathFinder, SourceFileLoader
from importlib.util import cache_from_source
from pathlib import Path


class _CachedSourceLoader(SourceFileLoader):
    def __init__(self, fullname: str, source: str, cache: Path) -> None:
        super().__init__(fullname, source)
        self.cache = cache
        self._original_cache = cache_from_source(source)

    def get_data(self, path: str) -> bytes:
        if path != self._original_cache:
            return super().get_data(path)
        try:
            return super().get_data(str(self.cache))
        except OSError:
            # Seed from existing bytecode once, avoiding needless recompilation.
            # SourceFileLoader validates these bytes before executing any code.
            data = super().get_data(path)
            if not sys.dont_write_bytecode:
                self.set_data(path, data)
            return data

    def set_data(self, path: str, data: bytes, *, _mode: int = 0o666) -> None:
        destination = str(self.cache) if path == self._original_cache else path
        super().set_data(destination, data, _mode=_mode)


class _CachedExtensionLoader(ExtensionFileLoader):
    """Avoid repeated DLL reads from synced storage, preserving module paths.

    Windows opens an extension binary several times during loading. Keep a
    versioned local copy; its name changes whenever source size or timestamp
    changes. Extensions needing DLLs relative to their original location fall
    back to the ordinary loader. No installed package files are modified.
    """

    def __init__(self, fullname: str, source: str, cache: Path) -> None:
        super().__init__(fullname, source)
        self.cache = cache

    def _local_binary(self) -> Path:
        source = Path(self.path)
        stat = source.stat()
        identity = hashlib.sha256(
            f"{stat.st_mtime_ns}:{stat.st_size}:{sys.implementation.cache_tag}".encode()
        ).hexdigest()[:20]
        destination = self.cache / f"{source.stem}-{identity}{source.suffix}"
        if destination.is_file() and destination.stat().st_size == stat.st_size:
            return destination
        self.cache.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.cache, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(source.read_bytes())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return destination

    def create_module(self, spec):
        original = spec.origin
        try:
            spec.origin = str(self._local_binary())
            return super().create_module(spec)
        except (OSError, ImportError):
            # Dependent DLL lookup may require the original package directory.
            spec.origin = original
            return super().create_module(spec)
        finally:
            spec.origin = original

    def exec_module(self, module):
        # Keep resource lookup and package diagnostics tied to the installation.
        module.__file__ = self.path
        super().exec_module(module)


class _LocalCacheFinder:
    def __init__(self, project: Path, cache: Path, *, cache_dependencies: bool = False) -> None:
        self.project, self.cache = project, cache
        self.cache_dependencies = cache_dependencies

    def find_spec(self, fullname, path=None, target=None):
        if not self.cache_dependencies and fullname.split(".", 1)[0] not in {"app", "dnascape"}:
            return None
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is None or not spec.origin:
            return None
        # Limit binary relocation to the networking accelerators measured at
        # startup. Scientific/other extensions keep their ordinary loaders
        # and package-relative native-library/resource lookup.
        network_extension = (fullname == "_brotli" or fullname.startswith("aiohttp.")
                             or fullname.startswith("backports.zstd."))
        if (self.cache_dependencies and network_extension and os.name == "nt"
                and isinstance(spec.loader, ExtensionFileLoader)):
            try:
                relative = Path(spec.origin).relative_to(self.project)
            except ValueError:
                return None
            spec.loader = _CachedExtensionLoader(fullname, spec.origin, self.cache / "native" / relative.parent)
            return spec
        if not isinstance(spec.loader, SourceFileLoader):
            return None
        try:
            relative_cache = Path(cache_from_source(spec.origin)).relative_to(self.project)
        except ValueError:
            return None
        cache = self.cache / relative_cache
        spec.loader = _CachedSourceLoader(fullname, spec.origin, cache)
        spec.cached = str(cache)
        return spec


def install_local_bytecode_cache() -> Path | None:
    """Install one scoped cache, respecting explicit Python cache preferences."""
    if sys.pycache_prefix is not None or os.getenv("DNASCAPE_BYTECODE_CACHE", "").casefold() in {"0", "off", "false"}:
        return None
    for finder in sys.meta_path:
        if isinstance(finder, _LocalCacheFinder):
            return finder.cache
    project = Path(__file__).resolve().parent.parent
    if os.name == "nt":
        home = Path(os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        home = Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache")
    identity = hashlib.sha256(os.fsencode(str(project))).hexdigest()[:16]
    cache = home / "DNAscape" / "bytecode" / identity
    try:
        cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError:
        return None  # A read-only cache must never prevent the app from starting.
    # Preserve import hooks such as pytest's assertion rewriter, replacing only
    # the normal filesystem lookup which would otherwise handle these modules.
    position = next((index for index, finder in enumerate(sys.meta_path) if finder is PathFinder), len(sys.meta_path))
    # A .venv inside Dropbox/network storage suffers the same bytecode I/O
    # delays as the application. Include only sources under this project;
    # external environments and the standard library retain their loaders.
    dependencies = Path(sys.prefix).is_relative_to(project)
    sys.meta_path.insert(position, _LocalCacheFinder(project, cache, cache_dependencies=dependencies))
    return cache


def read_asset_text(source: Path) -> str:
    """Read a shipped UTF-8 asset, reusing a validated local copy across launches."""
    finder = next((item for item in sys.meta_path if isinstance(item, _LocalCacheFinder)), None)
    if finder is None or not source.is_relative_to(finder.project):
        return source.read_text(encoding="utf-8")
    stat = source.stat()
    fingerprint = [stat.st_mtime_ns, stat.st_size]
    identity = hashlib.sha256(os.fsencode(str(source))).hexdigest()
    cache = finder.cache / "assets" / f"{identity}.json"
    try:
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved["source"] == fingerprint and isinstance(saved["text"], str):
            return saved["text"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    text = source.read_text(encoding="utf-8")
    temporary = None
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump({"source": fingerprint, "text": text}, stream)
        os.replace(temporary, cache)
    except OSError:
        pass  # Optional caches must not make a readable asset unavailable.
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return text
