"""DNAscape NiceGUI interface with a local application bytecode cache."""

from app._startup_cache import install_local_bytecode_cache as _install_cache

_install_cache()

__all__ = ["__version__"]

__version__ = "1.0.0-prototype"
