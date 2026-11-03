"""One iteration budget and measured cost estimate across refinement grids."""

import time


class FitProgress:
    def __init__(self, total):
        self.total = max(1, total)
        self.pending = []

    def reserve(self, points, iterations):
        self.pending.append((points, iterations))

    def release(self):
        self.pending.pop()

    def estimate(self, seconds, points, remaining):
        # A future grid also needs one forward/adjoint initialization. Include
        # it now, rather than discovering that work at the end of the run.
        units = remaining + sum((count + 1) * n / points for n, count in self.pending)
        return max(0.0, seconds * units)


class FitStage:
    def __init__(self, tracker, points):
        self.tracker = tracker
        self.points = points
        self.started = time.perf_counter()
        self.step_seconds = None
        self.last = None

    def ready(self):
        now = time.perf_counter()
        # Start with two measured forwards' worth of work for a proposal and
        # its reverse pass. Later updates measure actual line searches too.
        self.step_seconds = max(1e-6, 2 * (now - self.started))
        self.last = now

    def update(self):
        now = time.perf_counter()
        duration = max(1e-6, now - self.last)
        self.step_seconds = 0.5 * self.step_seconds + 0.5 * duration
        self.last = now

    def eta(self, remaining):
        return self.tracker.estimate(self.step_seconds, self.points, remaining)
