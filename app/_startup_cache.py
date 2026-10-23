"""Keep application bytecode out of slow synced or network source folders.

Only DNAscape's own source loaders change cache location. Python still checks
source timestamps, sizes and bytecode versions, and dependencies retain their
normal caches. Source paths, resource lookup and tracebacks stay unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from importlib.machinery import PathFinder, SourceFileLoader
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


class _LocalCacheFinder:
    def __init__(self, project: Path, cache: Path) -> None:
        self.project, self.cache = project, cache

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] not in {"app", "dnascape"}:
            return None
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is None or not isinstance(spec.loader, SourceFileLoader) or not spec.origin:
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
    sys.meta_path.insert(position, _LocalCacheFinder(project, cache))
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
