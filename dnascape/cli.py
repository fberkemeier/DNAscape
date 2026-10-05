"""Command-line interface for DNAscape."""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PACKAGE_NAME = "dnascape"
MODULE_SKIP = {"__init__", "cli", "core", "constants"}


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    kind: str  # "posonly" | "arg" | "kwonly" | "vararg" | "varkw"
    has_default: bool
    default_repr: str | None = None


@dataclass(frozen=True)
class FunctionSpec:
    module: str
    name: str
    command: str
    summary: str
    parameters: tuple[ParameterSpec, ...]


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def _project_root() -> Path:
    return _package_dir().parent


def _import_numpy():
    try:
        import numpy as np

        return np
    except Exception as exc:
        raise RuntimeError(
            "NumPy is required for array file parsing (.npy/.npz/.txt/.csv). "
            "Install project requirements first."
        ) from exc


def _package_version() -> str:
    try:
        return importlib.metadata.version(PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        pyproject = _project_root() / "pyproject.toml"
        if pyproject.exists():
            match = re.search(
                r'^\s*version\s*=\s*["\']([^"\']+)["\']',
                pyproject.read_text(encoding="utf-8"),
                flags=re.MULTILINE,
            )
            if match:
                return match.group(1)
    return "0+unknown"


def _default_repr(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _summary_from_doc(node: ast.FunctionDef) -> str:
    doc = ast.get_docstring(node)
    if not doc:
        return ""
    return doc.strip().splitlines()[0].strip()


def _parameter_specs(node: ast.FunctionDef) -> list[ParameterSpec]:
    specs: list[ParameterSpec] = []
    posonly = list(node.args.posonlyargs)
    pos_or_kw = list(node.args.args)
    all_pos = posonly + pos_or_kw
    all_defaults = [None] * (len(all_pos) - len(node.args.defaults)) + list(node.args.defaults)

    for idx, (arg_node, default_node) in enumerate(zip(all_pos, all_defaults)):
        kind = "posonly" if idx < len(posonly) else "arg"
        specs.append(
            ParameterSpec(
                name=arg_node.arg,
                kind=kind,
                has_default=default_node is not None,
                default_repr=_default_repr(default_node),
            )
        )

    if node.args.vararg is not None:
        specs.append(ParameterSpec(name=node.args.vararg.arg, kind="vararg", has_default=False))

    for kw_arg, kw_default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        specs.append(
            ParameterSpec(
                name=kw_arg.arg,
                kind="kwonly",
                has_default=kw_default is not None,
                default_repr=_default_repr(kw_default),
            )
        )

    if node.args.kwarg is not None:
        specs.append(ParameterSpec(name=node.args.kwarg.arg, kind="varkw", has_default=False))

    return specs


def discover_functions() -> dict[str, FunctionSpec]:
    discovered: list[FunctionSpec] = []
    pkg_dir = _package_dir()

    for path in sorted(pkg_dir.glob("*.py")):
        module = path.stem
        if module in MODULE_SKIP:
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name.startswith("_"):
                continue

            discovered.append(
                FunctionSpec(
                    module=module,
                    name=node.name,
                    command=node.name,
                    summary=_summary_from_doc(node),
                    parameters=tuple(_parameter_specs(node)),
                )
            )

    by_name: dict[str, list[FunctionSpec]] = {}
    for spec in discovered:
        by_name.setdefault(spec.name, []).append(spec)

    result: dict[str, FunctionSpec] = {}
    for same_name_specs in by_name.values():
        if len(same_name_specs) == 1:
            spec = same_name_specs[0]
            result[spec.command] = spec
            continue

        for spec in same_name_specs:
            command = f"{spec.module}-{spec.name}"
            result[command] = FunctionSpec(
                module=spec.module,
                name=spec.name,
                command=command,
                summary=spec.summary,
                parameters=spec.parameters,
            )

    return dict(sorted(result.items(), key=lambda item: item[0]))


def _is_bool_default(default_repr: str | None) -> bool:
    return default_repr in {"True", "False"}


def _param_flag(name: str) -> str:
    return "--" + name.replace("_", "-").lower()


def _add_parameter_argument(parser: argparse.ArgumentParser, param: ParameterSpec) -> None:
    help_default = ""
    if param.has_default and param.default_repr is not None:
        help_default = f" (default: {param.default_repr})"

    if param.kind == "varkw":
        parser.add_argument(
            _param_flag(param.name),
            dest=param.name,
            default=argparse.SUPPRESS,
            help=f"Extra keyword arguments as JSON/Python dict{help_default}",
        )
        return

    if param.kind == "vararg":
        parser.add_argument(
            _param_flag(param.name),
            dest=param.name,
            nargs="+",
            default=argparse.SUPPRESS,
            help=f"One or more values for *{param.name}{help_default}",
        )
        return

    required = not param.has_default
    if _is_bool_default(param.default_repr):
        parser.add_argument(
            _param_flag(param.name),
            dest=param.name,
            action="store_true",
            default=argparse.SUPPRESS,
            help=f"Set {param.name}=True{help_default}",
        )
        parser.add_argument(
            "--no-" + param.name.replace("_", "-").lower(),
            dest=param.name,
            action="store_false",
            default=argparse.SUPPRESS,
            help=f"Set {param.name}=False{help_default}",
        )
        return

    parser.add_argument(
        _param_flag(param.name),
        dest=param.name,
        required=required,
        default=argparse.SUPPRESS,
        help=f"Value for {param.name}{help_default}",
    )


def _parse_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_parse_value(v) for v in value]
    if not isinstance(value, str):
        return value

    text = value.strip()
    lowered = text.lower()
    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    p = Path(text)
    if p.exists():
        suffix = p.suffix.lower()
        if suffix in {".npy", ".npz", ".txt", ".csv"}:
            np = _import_numpy()
            if suffix == ".npy":
                return np.load(p, allow_pickle=True)
            if suffix == ".npz":
                with np.load(p, allow_pickle=True) as payload:
                    return {k: payload[k] for k in payload.files}
            delimiter = "," if suffix == ".csv" else None
            return np.loadtxt(p, delimiter=delimiter)
        if suffix == ".json":
            return json.loads(p.read_text(encoding="utf-8"))
        return str(p)

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except Exception:
        return value


def _prepare_call_arguments(func, parsed: argparse.Namespace) -> tuple[list[Any], dict[str, Any]]:
    sig = inspect.signature(func)
    provided = vars(parsed)
    call_args: list[Any] = []
    call_kwargs: dict[str, Any] = {}

    for param in sig.parameters.values():
        if param.name not in provided:
            continue

        raw_value = provided[param.name]
        value = _parse_value(raw_value)

        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            call_args.append(value)
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            values = value if isinstance(value, list) else [value]
            call_args.extend(values)
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            if not isinstance(value, dict):
                flag = _param_flag(param.name)
                raise TypeError(f"{flag} must parse to a dict, got {type(value).__name__}.")
            call_kwargs.update(value)
        else:
            call_kwargs[param.name] = value

    return call_args, call_kwargs


def _resolve_function(spec: FunctionSpec):
    if spec.module == "plotting":
        import matplotlib

        matplotlib.use("Agg", force=True)

    module = importlib.import_module(f".{spec.module}", PACKAGE_NAME)
    func = getattr(module, spec.name, None)
    if func is None or not callable(func):
        raise AttributeError(f"Could not load callable {spec.module}.{spec.name}.")
    return func


def _print_result(result: Any) -> None:
    if result is None:
        return

    try:
        np = _import_numpy()
        if isinstance(result, np.ndarray):
            print(f"ndarray(shape={result.shape}, dtype={result.dtype})")
            return
    except Exception:
        pass

    if isinstance(result, dict):
        keys = ", ".join(map(str, result.keys()))
        print(f"dict(keys=[{keys}])")
        return

    print(result)


def build_parser(functions: dict[str, FunctionSpec] | None = None) -> argparse.ArgumentParser:
    functions = discover_functions() if functions is None else functions

    parser = argparse.ArgumentParser(
        prog="dnascape",
        description="DNAscape command-line interface",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {_package_version()}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, spec in functions.items():
        desc = f"{spec.module}.{spec.name}"
        if spec.summary:
            desc = f"{desc}: {spec.summary}"

        sub = subparsers.add_parser(
            command,
            help=desc,
            description=desc,
        )
        sub.set_defaults(_spec=spec)

        for param in spec.parameters:
            _add_parameter_argument(sub, param)

    return parser


def main(argv=None) -> int:
    functions = discover_functions()
    parser = build_parser(functions=functions)
    parsed = parser.parse_args(argv)

    spec: FunctionSpec = parsed._spec
    func = _resolve_function(spec)
    call_args, call_kwargs = _prepare_call_arguments(func, parsed)
    result = func(*call_args, **call_kwargs)
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
