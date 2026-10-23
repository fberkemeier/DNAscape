"""Array expressions shared by data transforms and selected-region modulation."""

from __future__ import annotations

import ast
import operator
from functools import lru_cache
from collections.abc import Mapping

import numpy as np


_FUNCTION_NAMES = """
abs absolute sqrt cbrt square exp exp2 expm1 log log2 log10 log1p
sin cos tan arcsin arccos arctan arctan2 sinh cosh tanh
minimum maximum fmin fmax clip where power sign floor ceil round rint trunc
mean median std var sum prod min max amin amax nanmean nanmedian nanstd nanvar
nansum nanprod nanmin nanmax percentile nanpercentile quantile nanquantile
isnan isfinite isinf nan_to_num gradient diff cumsum cumprod nancumsum nancumprod
roll flip fliplr flipud sort argsort interp concatenate stack column_stack
logical_and logical_or logical_not logical_xor mod remainder hypot sinc
""".split()
_FUNCTIONS = {name: getattr(np, name) for name in _FUNCTION_NAMES}
_CONSTANTS = {"pi": np.pi, "e": np.e, "nan": np.nan, "inf": np.inf}
_METHODS = {name for name in _FUNCTION_NAMES if hasattr(np.ndarray, name)}
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
           ast.Pow: np.float_power, ast.BitAnd: np.logical_and, ast.BitOr: np.logical_or,
           ast.BitXor: np.logical_xor, ast.MatMult: operator.matmul}
_COMPARE = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
            ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne}


def _convolve(source, profile, mode="same"):
    from scipy.signal import convolve
    return convolve(source, profile, mode=mode, method="auto")


def _smooth(source, sigma=1, axis=0, mode="reflect"):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(source, sigma, axis=axis, mode=mode)


_FUNCTIONS.update(convolve=_convolve, smooth=_smooth)


@lru_cache(maxsize=128)
def _expression(text: str) -> ast.Expression:
    if not text.strip():
        raise ValueError("Enter a formula.")
    if len(text) > 4096:
        raise ValueError("The formula is too long.")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except (SyntaxError, RecursionError):
        raise ValueError("Check the formula's Python syntax.") from None
    if sum(1 for _ in ast.walk(tree)) > 512:
        raise ValueError("Simplify the formula into fewer operations.")
    return tree


def evaluate_formula(text: str, variables: Mapping[str, np.ndarray], *, shape=None, check_mask=None) -> np.ndarray:
    """Evaluate numeric Python/NumPy syntax without executing Python statements.

    Only the supplied arrays and numeric functions are available. Inputs stay
    read-only, and a scalar result broadcasts across the existing data grid.
    """
    arrays = {}
    for name, values in variables.items():
        value = np.asarray(values, dtype=float).view()
        value.flags.writeable = False
        arrays[name] = value
    expected = tuple(shape if shape is not None else next(iter(arrays.values())).shape)

    def function(node):
        if isinstance(node, ast.Name) and node.id in _FUNCTIONS:
            return _FUNCTIONS[node.id], ()
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "np" and node.attr in _FUNCTIONS:
                return _FUNCTIONS[node.attr], ()
            if node.value.id in arrays and node.attr in _METHODS:
                return _FUNCTIONS[node.attr], (arrays[node.value.id],)
        raise ValueError("Use numeric NumPy functions in the formula, such as np.log or np.where.")

    def visit(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool, str, type(None))):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in arrays:
                return arrays[node.id]
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            raise ValueError(f"Unknown formula name '{node.id}'. Use {', '.join(arrays)} and NumPy functions.")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "np" and node.attr in _CONSTANTS:
            return _CONSTANTS[node.attr]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp):
            operations = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: np.logical_not, ast.Invert: np.logical_not}
            if type(node.op) in operations:
                return operations[type(node.op)](visit(node.operand))
        if isinstance(node, ast.Compare):
            left, result = visit(node.left), True
            for op, right_node in zip(node.ops, node.comparators):
                if type(op) not in _COMPARE:
                    break
                right = visit(right_node)
                result = np.logical_and(result, _COMPARE[type(op)](left, right))
                left = right
            else:
                return result
        if isinstance(node, ast.BoolOp):
            values = [visit(value) for value in node.values]
            operation = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
            result = values[0]
            for value in values[1:]:
                result = operation(result, value)
            return result
        if isinstance(node, ast.IfExp):
            return np.where(visit(node.test), visit(node.body), visit(node.orelse))
        if isinstance(node, (ast.List, ast.Tuple)):
            return [visit(value) for value in node.elts] if isinstance(node, ast.List) else tuple(visit(value) for value in node.elts)
        if isinstance(node, ast.Slice):
            return slice(*(visit(value) if value is not None else None for value in (node.lower, node.upper, node.step)))
        if isinstance(node, ast.Subscript):
            return visit(node.value)[visit(node.slice)]
        if isinstance(node, ast.Call):
            callback, prefix = function(node.func)
            if any(key.arg is None or key.arg == "out" or key.arg.startswith("_") for key in node.keywords):
                raise ValueError("Use direct numeric arguments in the formula.")
            return callback(*prefix, *(visit(value) for value in node.args), **{key.arg: visit(key.value) for key in node.keywords})
        raise ValueError("Enter a numeric Python expression using the source arrays and NumPy functions.")

    try:
        with np.errstate(all="ignore"):
            result = np.asarray(visit(_expression(text).body))
        if result.dtype.kind not in "biuf":
            raise ValueError("The formula must produce real numeric values.")
        if result.ndim == 0:
            result = np.full(expected, result.item(), dtype=float)
        if result.shape != expected:
            raise ValueError(f"The formula must keep the source shape {expected}; its result has shape {result.shape}.")
        checked = result if check_mask is None else result[check_mask]
        if np.isinf(checked).any():
            raise ValueError("The formula produced infinite values. Check divisions, logarithms and powers.")
        return np.array(result, dtype=float, copy=True)
    except (TypeError, IndexError, ZeroDivisionError, OverflowError, FloatingPointError) as exc:
        raise ValueError("The formula could not be applied. Check its arguments and array dimensions.") from exc
