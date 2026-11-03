"""One iteration budget and measured cost estimate across refinement grids."""

import math
import time
from collections import Counter
from contextlib import contextmanager


class FitProgress:
    def __init__(self, total):
        self.total = max(1, total)
        self.pending = []
        self.costs = {}

    def reserve(self, points, iterations, *, phases=None):
        counts = Counter(phases) if phases is not None else Counter(iteration=iterations)
        self.pending.append((points, counts))

    def release(self):
        self.pending.pop()

    def estimate(self, seconds, points, remaining):
        # A future grid also needs one forward/adjoint initialization. Include
        # it now, rather than discovering that work at the end of the run.
        units = remaining + sum((sum(counts.values()) + 1) * n / points for n, counts in self.pending)
        return max(0.0, seconds * units)

    def transfer_unused(self, phases):
        # Adaptive fitting spends unused coarse iterations on the next grid.
        # The estimate must move that work too, at the finer grid's cost.
        if self.pending:
            self.pending[-1][1].update(phases)

    @staticmethod
    def work(points, time_points, backend):
        spatial = max(1, points)
        if "convolution" in backend or "fft" in backend:
            spatial *= math.log2(max(2, points))
        return spatial * max(1, time_points)

    def observe(self, points, time_points, backend, phase, seconds):
        key = (points, time_points, backend, phase)
        previous = self.costs.get(key)
        # The first real iteration replaces the prediction-based bootstrap.
        # Thereafter smooth line-search variation within each phase/grid.
        self.costs[key] = seconds if previous is None else .65 * previous + .35 * seconds

    def step_cost(self, points, time_points, backend, phase, fallback):
        work = self.work(points, time_points, backend)
        samples = [(self.work(n, nt, kind), seconds)
                   for (n, nt, kind, name), seconds in self.costs.items()
                   if name == phase and kind == backend]
        if not samples:
            return fallback
        measured_work, seconds = min(samples, key=lambda item: abs(math.log(work / item[0])))
        return seconds * work / measured_work


class FitStage:
    def __init__(self, tracker, points, *, phases=None):
        self.tracker = tracker
        self.points = points
        self.started = time.perf_counter()
        self.step_seconds = None
        self.last = None
        self.phases = None if phases is None else Counter(phases)
        self.completed = Counter()
        self.time_points = 1
        self.backend = ""
        self.prediction_seconds = None

    @contextmanager
    def prediction(self):
        started = time.perf_counter()
        try:
            yield
        finally:
            # Clock extensions can repeat initialization. Their cumulative
            # cost is not the cost of every subsequent optimizer step.
            self.prediction_seconds = max(1e-6, time.perf_counter() - started)

    def ready(self, *, time_points=1, backend=""):
        now = time.perf_counter()
        # Start with two measured forwards' worth of work for a proposal and
        # its reverse pass. Later updates measure actual line searches too.
        self.prediction_seconds = self.prediction_seconds or max(1e-6, now - self.started)
        self.step_seconds = 2 * self.prediction_seconds
        self.time_points, self.backend = time_points, str(backend)
        self.last = now

    def update(self, phase="iteration"):
        now = time.perf_counter()
        duration = max(1e-6, now - self.last)
        self.tracker.observe(self.points, self.time_points, self.backend, phase, duration)
        self.completed[phase] += 1
        self.last = now

    def eta(self, remaining):
        phases = ((self.phases - self.completed) if remaining and self.phases is not None
                  else Counter(iteration=remaining))
        work = self.tracker.work(self.points, self.time_points, self.backend)

        def cost(points, counts, initialize=False):
            ratio = self.tracker.work(points, self.time_points, self.backend) / work
            seconds = self.prediction_seconds * ratio if initialize else 0.
            return seconds + sum(count * self.tracker.step_cost(
                points, self.time_points, self.backend, phase, self.step_seconds * ratio,
            ) for phase, count in counts.items())

        return max(0., cost(self.points, phases) + sum(
            cost(points, counts, initialize=True) for points, counts in self.tracker.pending
        ))

    def finish(self, *, transfer=False):
        if transfer and self.phases is not None:
            self.tracker.transfer_unused(self.phases - self.completed)
