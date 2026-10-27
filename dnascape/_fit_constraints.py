"""Worker-local projection of fitted fields; supplied fields are never edited."""

from contextlib import contextmanager
from threading import local

_STATE = local()


@contextmanager
def fit_constraint_context(constraint=None):
    previous = getattr(_STATE, 'constraint', None)
    _STATE.constraint = constraint
    try:
        yield
    finally:
        _STATE.constraint = previous


def project_fit(values, target, form, *, transposed=False):
    constraint = getattr(_STATE, 'constraint', None)
    if constraint is None or constraint.target != target or constraint.form != form:
        return values
    return constraint.project(values, transposed=transposed)


def fit_free_mask(values, target, form):
    constraint = getattr(_STATE, 'constraint', None)
    if constraint is None or constraint.target != target or constraint.form != form:
        return None
    constraint.project(values)
    return ~constraint.mask.reshape(values.shape)


def has_fit_constraint(target=None):
    constraint = getattr(_STATE, 'constraint', None)
    return constraint is not None and (target is None or constraint.target == target)
