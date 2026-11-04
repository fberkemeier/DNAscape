"""Native-grid forward fractions with bounded working arrays and mapped output."""

import shutil
import tempfile

import numpy as np

from .utils import check_cancelled, measured_work_phase, report_measured_work

DENSE_FRACTION_VALUES = 4_000_000
WORK_VALUES = 1_000_000
STATIC_CORE_POINTS = 16384


def disk_backed(values):
    """Recognize mapped storage through the public ndarray result wrappers."""
    while values is not None:
        if isinstance(values, np.memmap):
            return True
        values = getattr(values, "base", None)
    return False


def mapped_fraction(firing, speed, n, clock, dx, dt, periodic, radius, backend):
    from ._kinetic_tiled import disk_array
    from .mapping import _expand_space_time_field, _field_time_length, _spatial_length

    required = n * len(clock) * np.dtype(float).itemsize
    free = shutil.disk_usage(tempfile.gettempdir()).free
    if required > free:
        raise ValueError(
            f"The fraction result needs {required / 1024**3:.2f} GiB of temporary disk space; "
            f"{free / 1024**3:.2f} GiB is available."
        )
    result = disk_array((n, len(clock)), order="F")
    static = (backend == "auto" and _field_time_length(firing, n) is None
              and _field_time_length(speed, n) is None)
    if static:
        from ._hazard_static import StationaryHazard

        rate = np.broadcast_to(np.asarray(firing, dtype=float).reshape(-1), (n,))
        velocity = np.broadcast_to(np.asarray(speed, dtype=float).reshape(-1), (n,))
        cap = min(radius, n * dx / 2 if periodic else n * dx)
        halo = min(n, int(np.ceil(cap / dx)) + 1)
        core = max(STATIC_CORE_POINTS, halo)
        if core + 2 * halo >= n:
            core = n
        for first in range(0, n, core):
            check_cancelled()
            stop = min(n, first + core)
            if core == n:
                left, right, local_periodic = 0, n, periodic
            else:
                left, right, local_periodic = first - halo, stop + halo, False
                if not periodic:
                    left, right = max(0, left), min(n, right)
            indices = np.arange(left, right) % n
            block = max(1, WORK_VALUES // len(indices))
            for start in range(0, len(clock), block):
                check_cancelled()
                selected = clock[start:start + block]
                leading = selected[0] > 0
                times = np.r_[0., selected] if leading else selected
                low = (first + (stop - first) * start / len(clock)) / n
                high = (first + (stop - first) * min(1., (start + block) / len(clock))) / n
                with measured_work_phase(low, high):
                    op = StationaryHazard(velocity[indices], times, dx, local_periodic, cap, cache_bytes=0)
                    hazard = op.forward(rate[indices])[int(leading):, first - left:stop - left].copy()
                np.negative(hazard, out=hazard)
                np.expm1(hazard, out=hazard)
                np.negative(hazard, out=hazard)
                result[first:stop, start:start + block] = hazard.T
                del hazard, op
                report_measured_work(high)
        return result

    # Every tile includes the full physical source halo. Retarded potentials
    # additionally need their numerical history before radius subtraction.
    leading = clock[0] > 0
    times = np.r_[0., clock] if leading else clock

    def fields(values, indices, name, positive=False):
        raw = np.asarray(values)
        if _spatial_length(raw) is not None:
            raw = raw[indices]
        expanded = _expand_space_time_field(raw, len(clock), len(indices), clock, dt, name, positive)
        return np.vstack((expanded[:1], expanded)) if leading else expanded

    uniform_speed = uniform_rate = True
    uniform_clock = len(times) > 1 and np.allclose(np.diff(times), times[1], rtol=1e-11, atol=1e-12)
    maximum = np.zeros(len(times))
    change = np.zeros(max(0, len(times) - 1))
    reference = rate_reference = None
    scan = max(1, min(16384, WORK_VALUES // len(times)))
    for first in range(0, n, scan):
        check_cancelled()
        indices = np.arange(first, min(n, first + scan))
        velocity = fields(speed, indices, "fork_speed", True)
        rate = fields(firing, indices, "initiation_rate")
        if reference is None:
            reference = velocity[:, :1].copy()
        uniform_speed &= bool(np.all(velocity == reference))
        if rate_reference is None:
            rate_reference = rate[:, :1].copy()
        uniform_rate &= bool(np.all(rate == rate_reference))
        maximum = np.maximum(maximum, velocity.max(axis=1))
        if len(times) > 1:
            change = np.maximum(change, np.max(
                abs(velocity[1:] - velocity[:-1]) / (velocity[1:] + velocity[:-1]), axis=1))
    constant_speed = uniform_speed and bool(np.all(reference == reference[0]))
    stationary_source = uniform_speed and _field_time_length(firing, n) is None
    convolution = backend == "auto" and constant_speed and uniform_clock
    stationary_source = backend == "auto" and stationary_source and len(times) > 1
    uniform_kinetics = backend == "auto" and uniform_rate and uniform_speed and len(times) > 1
    steps = np.maximum(1, np.ceil(2. * change).astype(int))
    cap = min(radius, n * dx / 2 if periodic else n * dx)
    halo = int(np.ceil(cap / dx)) + 8
    if not convolution and not stationary_source and not uniform_kinetics:
        reach = float(np.maximum(maximum[:-1], maximum[1:]) @ np.diff(times))
        halo = int(np.ceil(max(cap, reach) / dx)) + 2 * int(steps.sum()) + 8
    halo = min(n, halo)
    core = min(16384, max(1024, halo, WORK_VALUES // len(times) - 2 * halo))
    if core + 2 * halo >= n:
        core = n
    for first in range(0, n, core):
        check_cancelled()
        stop = min(n, first + core)
        if core == n:
            left, right, local_periodic = 0, n, periodic
        else:
            left, right, local_periodic = first - halo, stop + halo, False
            if not periodic:
                left, right = max(0, left), min(n, right)
        indices = np.arange(left, right) % n
        rate = fields(firing, indices, "initiation_rate")
        velocity = fields(speed, indices, "fork_speed", True)
        args = (velocity, times, dx, local_periodic, cap)
        if uniform_kinetics:
            from ._hazard_uniform import UniformKineticsHazard
            op = UniformKineticsHazard(*args)
        elif convolution:
            from ._hazard_fft import ConstantSpeedConvolution
            op = ConstantSpeedConvolution(*args)
        elif stationary_source:
            from ._hazard_stationary_source import StationarySourceHazard
            op = StationarySourceHazard(*args, differentiate_speed=False)
        elif backend == "auto":
            from ._hazard_transport import FastTransportHazard
            op = FastTransportHazard(*args, uniform_speed=True, differentiate_speed=False)
            op._steps_override = steps
        else:
            from ._hazard import HazardOperator
            op = HazardOperator(velocity, times, resolution=dx, periodic=local_periodic,
                                radius=cap, backend=backend)
        with measured_work_phase(first / n, stop / n):
            hazard = op.forward(rate)[int(leading):, first - left:stop - left].copy()
        np.negative(hazard, out=hazard)
        np.expm1(hazard, out=hazard)
        np.negative(hazard, out=hazard)
        result[first:stop] = hazard.T
        del rate, velocity, hazard, op
        report_measured_work(stop / n)
    return result
