"""Validated reductions and local inverses of supplied replication fields.

Space-time arrays use positions by time. Undefined inverse values are masked
with NaN by default; ``return_info=True`` in rmap exposes the validity mask.
"""

from collections.abc import Mapping

import numpy as np
from scipy.ndimage import uniform_filter1d

from .mapping import map_coalescence_events_xt


class _ConversionArray(np.ndarray):
    def __new__(cls, values, **info):
        result = np.asarray(values, dtype=float).view(cls)
        result._rmap_info = info
        return result

    def __array_finalize__(self, obj):
        self._rmap_info = getattr(obj, "_rmap_info", {})


def _positive(value, name, *, zero=False):
    value = float(value)
    if not np.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}.")
    return value


def _field(value, name, *, fraction=False):
    values = np.asarray(value, dtype=float)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError(f"{name} must be a non-empty position-by-time matrix.")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{name} must contain finite nonnegative values.")
    if fraction and np.any(values > 1):
        raise ValueError(f"{name} must lie between zero and one.")
    return values


def _axis(count, values, resolution, name, *, minimum=1):
    if count < minimum:
        raise ValueError(f"{name} requires at least {minimum} samples.")
    resolution = _positive(resolution, f"{name} resolution")
    axis = np.arange(count, dtype=float) * resolution if values is None else np.asarray(values, dtype=float)
    if axis.ndim != 1 or axis.size != count or np.any(~np.isfinite(axis)):
        raise ValueError(f"{name} must be finite, one-dimensional, and match its field axis.")
    if count > 1 and np.any(np.diff(axis) <= 0):
        raise ValueError(f"{name} must be strictly increasing.")
    return axis


def _spatial_mean(values, resolution=1.0, x_array=None, spatial_average="bins"):
    axis = _axis(values.shape[0], x_array, resolution, "x_array")
    if spatial_average == "bins":
        return np.mean(values, axis=0)
    if spatial_average != "trapezoid":
        raise ValueError("spatial_average must be 'bins' or 'trapezoid'.")
    if axis.size == 1:
        return values[0].copy()
    return np.einsum("ij,i->j", values, _trapezoid_weights(axis)) / (axis[-1] - axis[0])


def _trapezoid_weights(axis):
    """Quadrature weights avoid full-matrix temporaries in field reductions."""
    weights = np.empty(axis.size, dtype=float)
    delta = np.diff(axis)
    weights[0], weights[-1] = delta[0] / 2., delta[-1] / 2.
    weights[1:-1] = (delta[:-1] + delta[1:]) / 2.
    return weights


def _integrate_rows(field, time):
    return np.einsum("ij,j->i", field, _trapezoid_weights(time))


def map_fraction_to_mean_fraction(
    replicated_fraction_xt, resolution=1.0, x_array=None, spatial_average="bins",
):
    """Reduce an existing f(x,t) to its spatial mean f(t)."""
    field = _field(replicated_fraction_xt, "replicated_fraction_xt", fraction=True)
    return _spatial_mean(field, resolution, x_array, spatial_average)


def _monotone_fraction(field, monotonic_policy="raise", temporal_smoothw=1):
    window = int(temporal_smoothw)
    if window != temporal_smoothw or window < 1:
        raise ValueError("temporal_smoothw must be a positive integer number of time samples.")
    if window > field.shape[1]:
        raise ValueError("temporal_smoothw cannot exceed the number of time samples.")
    if window > 1:
        field = uniform_filter1d(field, size=window, axis=1, mode="nearest")
    if monotonic_policy == "project":
        # The cumulative upper envelope is a transparent, inexpensive monotone
        # correction. It is not an isotonic least-squares estimator.
        return np.maximum.accumulate(field, axis=1)
    if monotonic_policy != "raise":
        raise ValueError("monotonic_policy must be 'raise' or 'project'.")
    if np.any(np.diff(field, axis=1) < -1e-12):
        raise ValueError("Replicated fraction decreases in time; use monotonic_policy='project' for noisy data.")
    return field


def map_fraction_to_timing(
    replicated_fraction_xt,
    resolution_time=5.0,
    time_grid=None,
    tail_policy="require_complete",
    tail_tolerance=1e-6,
    tail_fit_points=3,
    initial_policy="require_unreplicated",
    monotonic_policy="raise",
):
    """Integrate supplied survival, with explicit left/right censor handling.

    ``truncate`` returns E[min(T, t_end)], while ``exponential`` extends the
    final survival using a log-linear fit to the last ``tail_fit_points``.
    ``conditional`` reports timing conditional on remaining unreplicated at
    the first sample; otherwise the first sample must be unreplicated.
    """
    field = _monotone_fraction(
        _field(replicated_fraction_xt, "replicated_fraction_xt", fraction=True), monotonic_policy,
    )
    time = _axis(field.shape[1], time_grid, resolution_time, "time_grid", minimum=2)
    tolerance = _positive(tail_tolerance, "tail_tolerance", zero=True)
    if tolerance >= 1:
        raise ValueError("tail_tolerance must be less than one.")
    survival = 1.0 - field
    if initial_policy == "require_unreplicated":
        if np.any(field[:, 0] > tolerance):
            raise ValueError("The first fraction sample is already replicated; provide earlier data or initial_policy='conditional'.")
    elif initial_policy == "conditional":
        if np.any(survival[:, 0] <= tolerance):
            raise ValueError("Conditional timing is undefined for rows already completely replicated at the first sample.")
        survival = survival / survival[:, :1]
    else:
        raise ValueError("initial_policy must be 'require_unreplicated' or 'conditional'.")
    tail = np.zeros(field.shape[0], dtype=float)
    if tail_policy == "require_complete":
        if np.any(survival[:, -1] > tolerance):
            raise ValueError("The observation window does not reach replication completion; choose an explicit tail_policy.")
    elif tail_policy == "exponential":
        points = int(tail_fit_points)
        if points != tail_fit_points or points < 2 or points > time.size:
            raise ValueError("tail_fit_points must be an integer between two and the time sample count.")
        active = survival[:, -1] > tolerance
        if np.any(active):
            centered = time[-points:] - np.mean(time[-points:])
            log_survival = np.log(survival[active, -points:])
            rate = -(log_survival @ centered) / np.dot(centered, centered)
            if np.any(~np.isfinite(rate)) or np.any(rate <= 0):
                raise ValueError("An exponential tail requires decreasing positive terminal survival.")
            tail[active] = survival[active, -1] / rate
    elif tail_policy != "truncate":
        raise ValueError("tail_policy must be 'require_complete', 'truncate', or 'exponential'.")
    timing = time[0] + _integrate_rows(survival, time) + tail
    return _ConversionArray(
        timing, tail_policy=tail_policy, initial_policy=initial_policy,
        terminal_survival=survival[:, -1].copy(), extrapolated_tail=tail,
        timing_is_restricted_mean=tail_policy == "truncate",
    )


def map_density_to_temporal(
    density_xt, resolution=1.0, x_array=None, spatial_average="bins", target_form="time",
):
    """Spatial mean active-fork density, returned as a time-only row array."""
    if target_form not in {"time", "t"}:
        raise ValueError("Temporal density reduction requires target_form='time'.")
    field = _field(density_xt, "density_xt")
    return _spatial_mean(field, resolution, x_array, spatial_average).reshape(1, -1)


def map_density_to_spatial(
    density_xt, resolution_time=5.0, time_grid=None, target_form="space",
):
    """Time-integrated density profile; its units are fork-minutes per kb."""
    if target_form not in {"space", "x"}:
        raise ValueError("Spatial density reduction requires target_form='space'.")
    field = _field(density_xt, "density_xt")
    time = _axis(field.shape[1], time_grid, resolution_time, "time_grid", minimum=2)
    return _integrate_rows(field, time)


def map_spatial_events_to_pdf(event_profile, resolution=1.0, x_array=None):
    """Normalize a supplied spatial event profile without inventing a time axis."""
    profile = np.asarray(event_profile, dtype=float)
    if profile.ndim == 2 and profile.shape[1] == 1:
        profile = profile[:, 0]
    if profile.ndim != 1 or not profile.size or np.any(~np.isfinite(profile)) or np.any(profile < 0):
        raise ValueError("event_profile must be a finite, nonnegative spatial vector.")
    axis = _axis(profile.size, x_array, resolution, "x_array")
    integral = profile[0] * float(resolution) if profile.size == 1 else np.trapezoid(profile, axis)
    if integral <= 0 or not np.isfinite(integral):
        raise ValueError("event_profile must have a positive finite spatial integral.")
    return profile / integral


def _bundle_field(bundle, member):
    if not isinstance(bundle, Mapping):
        raise ValueError("fork_densities_xt must be a mapping of named fields.")
    if member == "fork_density_xt" and member not in bundle:
        right = _bundle_field(bundle, "right_fork_density_xt")
        left = _bundle_field(bundle, "left_fork_density_xt")
        if right.shape != left.shape:
            raise ValueError("Directional fork-density bundle fields must share the same shape.")
        return right + left
    if member not in bundle:
        raise ValueError(f"The fork-density bundle is missing {member}.")
    return _field(bundle[member], member, fraction=member == "replicated_fraction_xt")


def map_bundle_to_fraction(bundle):
    return _bundle_field(bundle, "replicated_fraction_xt").copy()


def map_bundle_to_right(bundle):
    return _bundle_field(bundle, "right_fork_density_xt").copy()


def map_bundle_to_left(bundle):
    return _bundle_field(bundle, "left_fork_density_xt").copy()


def _coalescence_pdf(source, *, resolution, x_array, resolution_time, time_grid, unreplicated_floor):
    events = map_coalescence_events_xt(
        source["fork_speed"], source["replicated_fraction_xt"],
        source["right_fork_density_xt"], source["left_fork_density_xt"],
        resolution_time=resolution_time, time_grid=time_grid, unreplicated_floor=unreplicated_floor,
    )
    profile = map_density_to_spatial(events, resolution_time=resolution_time, time_grid=time_grid)
    return map_spatial_events_to_pdf(profile, resolution=resolution, x_array=x_array)


def map_density_fields_to_coalescence_pdf(
    source, resolution=1.0, x_array=None, resolution_time=5.0, time_grid=None,
    unreplicated_floor=1e-12,
):
    fields = {key: _field(source[key], key, fraction=key == "replicated_fraction_xt") for key in (
        "replicated_fraction_xt", "right_fork_density_xt", "left_fork_density_xt",
    )}
    return _coalescence_pdf(
        {**fields, "fork_speed": source["fork_speed"]}, resolution=resolution, x_array=x_array,
        resolution_time=resolution_time, time_grid=time_grid,
        unreplicated_floor=_positive(unreplicated_floor, "unreplicated_floor", zero=True),
    )


def map_density_bundle_to_coalescence_pdf(
    source, resolution=1.0, x_array=None, resolution_time=5.0, time_grid=None,
    unreplicated_floor=1e-12,
):
    bundle = source["fork_densities_xt"]
    fields = {key: _bundle_field(bundle, key) for key in (
        "replicated_fraction_xt", "right_fork_density_xt", "left_fork_density_xt",
    )}
    return map_density_fields_to_coalescence_pdf(
        {**fields, "fork_speed": source["fork_speed"]}, resolution=resolution,
        x_array=bundle.get("x_array") if x_array is None else x_array,
        resolution_time=resolution_time,
        time_grid=bundle.get("time_grid") if time_grid is None else time_grid,
        unreplicated_floor=unreplicated_floor,
    )


def _masked_ratio(numerator, denominator, valid, invalid_policy, **info):
    if invalid_policy not in {"mask", "zero", "raise"}:
        raise ValueError("invalid_policy must be 'mask', 'zero', or 'raise'.")
    if invalid_policy == "raise" and not np.all(valid):
        raise ValueError("The requested inverse is unidentifiable where its denominator is below the numerical floor.")
    fill = 0.0 if invalid_policy == "zero" else np.nan
    values = np.full(numerator.shape, fill, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        np.divide(numerator, denominator, out=values, where=valid)
    overflow = valid & ~np.isfinite(values)
    if np.any(overflow):
        if invalid_policy == "raise":
            raise ValueError("The requested inverse exceeds the finite numeric range.")
        valid = valid & ~overflow
        values[overflow] = fill
    return _ConversionArray(
        values, valid_mask=valid, unidentifiable_count=int(valid.size - np.count_nonzero(valid)),
        invalid_policy=invalid_policy, **info,
    )


def map_events_and_fraction_to_firing(
    source, unreplicated_floor=1e-8, invalid_policy="mask", target_form="spacetime",
):
    """Infer local firing hazard E_i/(1-f), leaving completed DNA unidentified."""
    if target_form not in {"spacetime", "xt"}:
        raise ValueError("Event inversion produces target_form='spacetime'.")
    events = _field(source["initiation_events_xt"], "initiation_events_xt")
    fraction = _field(source["replicated_fraction_xt"], "replicated_fraction_xt", fraction=True)
    if events.shape != fraction.shape:
        raise ValueError("Initiation events and replicated fraction must have the same shape.")
    floor = _positive(unreplicated_floor, "unreplicated_floor", zero=True)
    if floor >= 1:
        raise ValueError("unreplicated_floor must be less than one.")
    survival = 1.0 - fraction
    valid = survival > floor
    return _masked_ratio(events, survival, valid, invalid_policy, unreplicated_floor=floor)


def map_fraction_and_density_to_speed(
    source, resolution_time=5.0, time_grid=None, density_floor=1e-10,
    temporal_smoothw=1, monotonic_policy="raise", derivative_method="gradient",
    invalid_policy="mask", target_form="spacetime", target_shape="best",
    unreplicated_floor=0.001,
):
    """Infer symmetric fork speed from df/dt divided by total fork density.

    Gradient uses local polynomial differences on the supplied nonuniform time
    axis (second order with at least three samples). Forward differences use
    the following interval, repeating its slope at the last sample. Optional
    smoothing is a centered nearest-padded mean over sample indices.
    Cells with too little unreplicated DNA are excluded because dividing a
    finite-difference tail by a vanishing fork density is poorly conditioned.
    Set unreplicated_floor=0 to retain all incomplete cells above density_floor.
    """
    if target_form not in {"spacetime", "xt"}:
        raise ValueError("Flux inversion produces target_form='spacetime'.")
    if target_shape != "best":
        raise ValueError("Flux inversion requires target_shape='best'.")
    fraction = _field(source["replicated_fraction_xt"], "replicated_fraction_xt", fraction=True)
    density = _field(source["fork_density_xt"], "fork_density_xt")
    if fraction.shape != density.shape:
        raise ValueError("Replicated fraction and total fork density must have the same shape.")
    time = _axis(fraction.shape[1], time_grid, resolution_time, "time_grid", minimum=2)
    raw_survival = 1.0 - fraction
    fraction = _monotone_fraction(fraction, monotonic_policy, temporal_smoothw)
    if derivative_method == "gradient":
        derivative = np.gradient(fraction, time, axis=1, edge_order=2 if time.size > 2 else 1)
    elif derivative_method == "forward":
        slopes = np.diff(fraction, axis=1) / np.diff(time)
        derivative = np.concatenate((slopes, slopes[:, -1:]), axis=1)
    else:
        raise ValueError("derivative_method must be 'gradient' or 'forward'.")
    # Nonuniform polynomial endpoint differences can overshoot even monotone
    # data. Keep physical nonnegative replication flux and report this clipping.
    negative = derivative < 0
    derivative = np.maximum(derivative, 0.0)
    floor = _positive(density_floor, "density_floor", zero=True)
    survival_floor = _positive(unreplicated_floor, "unreplicated_floor", zero=True)
    if survival_floor >= 1:
        raise ValueError("unreplicated_floor must be less than one.")
    valid = (
        (density > floor) & (raw_survival > survival_floor)
        & ((1.0 - fraction) > survival_floor)
    )
    return _masked_ratio(
        derivative, density, valid, invalid_policy, density_floor=floor,
        unreplicated_floor=survival_floor,
        derivative_method=derivative_method, clipped_derivative_count=int(np.count_nonzero(negative)),
        monotonic_policy=monotonic_policy, temporal_smoothw=int(temporal_smoothw),
    )
