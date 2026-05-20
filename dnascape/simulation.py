"""Stochastic replication simulation engines."""

import heapq
import os
import tempfile
import time

import numpy as np

from .utils import etaf, fast_shift_add


def _resolve_n_bins(firing_rate, fork_speed, n_bins=None):
    """Infer the spatial genome length from spatial or space-time inputs."""
    candidates = []

    def add_candidate(field, name):
        if field is None:
            return
        arr = np.asarray(field)
        if arr.size == 1:
            return
        if arr.ndim == 1:
            candidates.append((name, int(arr.size)))
            return
        if arr.ndim == 2:
            if arr.shape[0] == 1 and arr.shape[1] > 1:
                return
            if arr.shape[1] == 1 and arr.shape[0] > 1:
                candidates.append((name, int(arr.shape[0])))
                return
            candidates.append((name, int(arr.shape[1])))
            return
        raise ValueError(f"{name} must be scalar, (n,), (1,T), (n,1), or (T,n).")

    add_candidate(firing_rate, "firing_rate")
    add_candidate(fork_speed, "fork_speed")

    if n_bins is not None:
        n = int(n_bins)
        if n <= 0:
            raise ValueError("n_bins must be a positive integer.")
        mismatches = [f"{name} has n={candidate}" for name, candidate in candidates if candidate != n]
        if mismatches:
            raise ValueError("n_bins is inconsistent with supplied fields: " + "; ".join(mismatches))
        return n

    unique = {candidate for _, candidate in candidates}
    if len(unique) == 1:
        return unique.pop()
    if len(unique) > 1:
        detail = ", ".join(f"{name}: n={candidate}" for name, candidate in candidates)
        raise ValueError(f"Spatial field lengths disagree ({detail}).")
    raise ValueError(
        "Cannot infer genome length from scalar or time-only inputs. "
        "Pass n_bins, or provide at least one spatial profile or space-time field."
    )


def _as_rate_field(field, name, n_bins):
    arr = np.asarray(field, dtype=float)
    if arr.size == 1:
        value = float(arr.reshape(-1)[0])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} scalar must be finite and >= 0.")
        return "scalar", value, None
    if arr.ndim == 1:
        if arr.size != n_bins:
            raise ValueError(f"{name} (n,) must have length {n_bins}, got {arr.size}.")
        if np.any(~np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError(f"{name} (n,) must be finite and >= 0.")
        return "x", arr.astype(float, copy=False), None
    if arr.ndim == 2:
        if arr.shape[0] == 1 and arr.shape[1] > 1:
            values_t = arr.reshape(-1)
            if np.any(~np.isfinite(values_t)) or np.any(values_t < 0):
                raise ValueError(f"{name} (1,T) must be finite and >= 0.")
            return "t", values_t.astype(float, copy=False), int(values_t.size)
        if arr.shape[1] == 1 and arr.shape[0] > 1:
            values_x = arr.reshape(-1)
            if values_x.size != n_bins:
                raise ValueError(f"{name} (n,1) must have length {n_bins}, got {values_x.size}.")
            if np.any(~np.isfinite(values_x)) or np.any(values_x < 0):
                raise ValueError(f"{name} (n,1) must be finite and >= 0.")
            return "x", values_x.astype(float, copy=False), None
        if arr.shape[1] != n_bins:
            raise ValueError(f"{name} (T,n) must have n={n_bins} columns, got {arr.shape[1]}.")
        if np.any(~np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError(f"{name} (T,n) must be finite and >= 0.")
        return "xt", arr.astype(float, copy=False), int(arr.shape[0])
    raise ValueError(f"{name} must be scalar, (n,), (1,T), (n,1), or (T,n).")


def _as_speed_field(field, name, n_bins):
    arr = np.asarray(field, dtype=float)
    if arr.size == 1:
        value = float(arr.reshape(-1)[0])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} scalar must be finite and > 0.")
        return "scalar", value, None
    if arr.ndim == 1:
        if arr.size != n_bins:
            raise ValueError(f"{name} (n,) must have length {n_bins}, got {arr.size}.")
        if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
            raise ValueError(f"{name} (n,) must be finite and > 0.")
        return "x", arr.astype(float, copy=False), None
    if arr.ndim == 2:
        if arr.shape[0] == 1 and arr.shape[1] > 1:
            values_t = arr.reshape(-1)
            if np.any(~np.isfinite(values_t)) or np.any(values_t <= 0):
                raise ValueError(f"{name} (1,T) must be finite and > 0.")
            return "t", values_t.astype(float, copy=False), int(values_t.size)
        if arr.shape[1] == 1 and arr.shape[0] > 1:
            values_x = arr.reshape(-1)
            if values_x.size != n_bins:
                raise ValueError(f"{name} (n,1) must have length {n_bins}, got {values_x.size}.")
            if np.any(~np.isfinite(values_x)) or np.any(values_x <= 0):
                raise ValueError(f"{name} (n,1) must be finite and > 0.")
            return "x", values_x.astype(float, copy=False), None
        if arr.shape[1] != n_bins:
            raise ValueError(f"{name} (T,n) must have n={n_bins} columns, got {arr.shape[1]}.")
        if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
            raise ValueError(f"{name} (T,n) must be finite and > 0.")
        return "xt", arr.astype(float, copy=False), int(arr.shape[0])
    raise ValueError(f"{name} must be scalar, (n,), (1,T), (n,1), or (T,n).")


def _normalise_firing_rate(firing_rate=None, ori_rate=None, initiation_rate=None):
    supplied = [
        name
        for name, value in (
            ("firing_rate", firing_rate),
            ("ori_rate", ori_rate),
            ("initiation_rate", initiation_rate),
        )
        if value is not None
    ]
    if len(supplied) > 1:
        raise ValueError(
            "Pass only one firing-rate argument. Use firing_rate as the preferred name; "
            "ori_rate and initiation_rate are compatibility aliases."
        )
    if firing_rate is not None:
        return firing_rate
    if ori_rate is not None:
        return ori_rate
    if initiation_rate is not None:
        return initiation_rate
    raise ValueError("A firing_rate input is required.")


def simulate_replication(
    firing_rate=None,
    fork_speed=1.4,
    resolution_space=1.0,
    resolution_time=1.0,
    rng=None,
    perQ=False,
    stall_rate=0.0,
    tau=np.inf,
    ffiring_nr=None,
    ffiring_recycle=0.0,
    ffiring_forkQ=False,
    time_statsQ=False,
    time_stats_xtQ=False,  # space-time stats are handled in rsim; kept for API symmetry
    time_stats_densQ=None,  # deprecated alias; kept for backwards compatibility
    time_grid=None,
    max_rep_time=1200.0,
    n_bins=None,
    ori_rate=None,
    initiation_rate=None,
):
    """Run one stochastic replication simulation.

    ``firing_rate`` and ``fork_speed`` may be scalar, spatial ``(n,)``,
    time-only ``(1, T)``, or space-time ``(T, n)`` fields. If neither input
    contains a spatial axis, pass ``n_bins`` explicitly.
    """
    rng = np.random.default_rng() if rng is None else rng
    if time_stats_densQ is not None:
        time_stats_xtQ = bool(time_stats_densQ)

    resolution_space = float(resolution_space)
    if not np.isfinite(resolution_space) or resolution_space <= 0:
        raise ValueError("resolution_space must be a finite positive number.")

    resolution_time = float(resolution_time)
    if not np.isfinite(resolution_time) or resolution_time <= 0:
        raise ValueError("resolution_time must be a finite positive number.")

    firing_rate = _normalise_firing_rate(
        firing_rate=firing_rate,
        ori_rate=ori_rate,
        initiation_rate=initiation_rate,
    )
    n = _resolve_n_bins(firing_rate, fork_speed, n_bins=n_bins)

    I_kind, I_val, TI = _as_rate_field(firing_rate, "firing_rate", n_bins=n)
    v_kind, v_val, Tv = _as_speed_field(fork_speed, "fork_speed", n_bins=n)

    inv_v_x = None
    inv_v_t = None
    inv_v_xt = None
    if v_kind == "scalar":
        inv_v_scalar = 1.0 / v_val
    elif v_kind == "x":
        inv_v_x = 1.0 / v_val
    elif v_kind == "t":
        inv_v_t = 1.0 / v_val
    else:
        inv_v_xt = 1.0 / v_val
        Tv = inv_v_xt.shape[0]

    p = float(stall_rate)
    stall_left_mask = rng.random(n) < p
    stall_right_mask = rng.random(n) < p
    stall_delay_left = np.zeros(n, float)
    stall_delay_right = np.zeros(n, float)
    if p > 0.0:
        if np.isfinite(tau):
            if stall_left_mask.any():
                stall_delay_left[stall_left_mask] = rng.exponential(scale=tau, size=int(stall_left_mask.sum()))
            if stall_right_mask.any():
                stall_delay_right[stall_right_mask] = rng.exponential(scale=tau, size=int(stall_right_mask.sum()))
        else:
            stall_delay_left[stall_left_mask] = np.inf
            stall_delay_right[stall_right_mask] = np.inf

    def _row_for_time(t):
        if not np.isfinite(t) or t < 0.0:
            return 0
        r = int(t // resolution_time)
        if (inv_v_t is not None or inv_v_xt is not None) and r >= Tv:
            r = Tv - 1
        return r

    def _edge_forward(t_here, k_to):
        if inv_v_xt is not None:
            r = _row_for_time(t_here)
            return resolution_space * inv_v_xt[r, k_to] + stall_delay_left[k_to]
        if inv_v_t is not None:
            r = _row_for_time(t_here)
            return resolution_space * inv_v_t[r] + stall_delay_left[k_to]
        if inv_v_x is not None:
            return resolution_space * inv_v_x[k_to] + stall_delay_left[k_to]
        return resolution_space * inv_v_scalar + stall_delay_left[k_to]

    def _edge_backward(t_here, k_to):
        if inv_v_xt is not None:
            r = _row_for_time(t_here)
            return resolution_space * inv_v_xt[r, k_to] + stall_delay_right[k_to]
        if inv_v_t is not None:
            r = _row_for_time(t_here)
            return resolution_space * inv_v_t[r] + stall_delay_right[k_to]
        if inv_v_x is not None:
            return resolution_space * inv_v_x[k_to] + stall_delay_right[k_to]
        return resolution_space * inv_v_scalar + stall_delay_right[k_to]

    # firing proposals t0
    t0 = np.full(n, np.inf, float)
    if I_kind == "x":
        rate = I_val
        pos = rate > 0
        if np.any(pos):
            t0[pos] = rng.exponential(scale=1.0 / rate[pos])
    elif I_kind == "scalar":
        if I_val > 0:
            t0[:] = rng.exponential(scale=1.0 / I_val, size=n)
    elif I_kind == "t":
        u = rng.random(n)
        target = -np.log(u)
        H = np.cumsum(I_val * resolution_time)
        hit = target <= H[-1] if H.size else np.zeros(n, dtype=bool)
        if np.any(hit):
            kh = np.flatnonzero(hit).astype(np.int64, copy=False)
            jh = np.searchsorted(H, target[hit], side="left").astype(np.int64, copy=False)
            prevH = np.zeros(kh.size, float)
            m0 = jh > 0
            if np.any(m0):
                prevH[m0] = H[jh[m0] - 1]
            lam = I_val[jh]
            frac = np.ones_like(lam)
            poslam = lam > 0
            frac[poslam] = (target[kh[poslam]] - prevH[poslam]) / (lam[poslam] * resolution_time)
            frac = np.clip(frac, 0.0, 1.0)
            t0[kh] = (jh.astype(float) + frac) * resolution_time
    else:
        u = rng.random(n)
        target = -np.log(u)
        H = np.cumsum(I_val * resolution_time, axis=0)
        ge = H >= target[None, :]
        hit = ge.any(axis=0)
        j = np.argmax(ge, axis=0).astype(np.int64, copy=False)
        if np.any(hit):
            jh = j[hit]
            kh = np.flatnonzero(hit).astype(np.int64, copy=False)
            prevH = np.zeros(kh.size, float)
            m0 = jh > 0
            if np.any(m0):
                prevH[m0] = H[jh[m0] - 1, kh[m0]]
            lam = I_val[jh, kh]
            frac = np.ones_like(lam)
            poslam = lam > 0
            frac[poslam] = (target[kh[poslam]] - prevH[poslam]) / (lam[poslam] * resolution_time)
            frac = np.clip(frac, 0.0, 1.0)
            t0[kh] = (jh.astype(float) + frac) * resolution_time

    # firing-factor gating (unchanged)
    use_factors = ffiring_nr is not None
    if use_factors:
        Amax = float(ffiring_nr)
        if not np.isfinite(Amax) or Amax <= 0:
            raise ValueError("ffiring_nr must be a finite positive number.")
        cost = 2.0
        if Amax < cost:
            raise ValueError("ffiring_nr must be >= 2 if modelling two forks per origin firing.")

        r = 0.0 if ffiring_forkQ else float(ffiring_recycle)
        if not ffiring_forkQ and (not np.isfinite(r) or r < 0):
            raise ValueError("ffiring_recycle must be a finite non-negative number.")

        def recover(A_now, dt):
            if r == 0.0 or dt <= 0.0:
                return A_now
            return Amax - (Amax - A_now) * np.exp(-r * dt)

        def wait_time_to_cost(A_now):
            if A_now >= cost:
                return 0.0
            if r == 0.0:
                return np.inf
            denom = Amax - A_now
            num = Amax - cost
            if denom <= 0.0 or num <= 0.0:
                return np.inf
            return (1.0 / r) * np.log(denom / num)

        def propagate_from_times(t):
            if not perQ:
                g = np.empty(n, float)
                src = np.empty(n, np.int32)
                g[0], src[0] = t[0], 0
                for k in range(1, n):
                    cand = g[k - 1] + _edge_forward(g[k - 1], k)
                    if t[k] <= cand:
                        g[k], src[k] = t[k], k
                    else:
                        g[k], src[k] = cand, src[k - 1]
                for k in range(n - 2, -1, -1):
                    cand = g[k + 1] + _edge_backward(g[k + 1], k)
                    if cand < g[k]:
                        g[k], src[k] = cand, src[k + 1]
                return g, src
            g = t.copy()
            src = np.arange(n, dtype=np.int32)
            visited = np.zeros(n, bool)
            pq = [(g[i], i) for i in range(n) if np.isfinite(g[i])]
            heapq.heapify(pq)
            while pq:
                time_u, u = heapq.heappop(pq)
                if visited[u] or time_u > g[u]:
                    continue
                visited[u] = True
                v = (u + 1) % n
                w = _edge_forward(time_u, v)
                if np.isfinite(w):
                    cand = time_u + w
                    if cand < g[v]:
                        g[v] = cand
                        src[v] = src[u]
                        heapq.heappush(pq, (cand, v))
                v = (u - 1) % n
                w = _edge_backward(time_u, v)
                if np.isfinite(w):
                    cand = time_u + w
                    if cand < g[v]:
                        g[v] = cand
                        src[v] = src[u]
                        heapq.heappush(pq, (cand, v))
            return g, src

        def schedule_with_releases(t_base, release_guess):
            t_adj = np.full(n, np.inf, float)
            A = Amax
            current_time = 0.0
            cand_idx = np.flatnonzero(np.isfinite(t_base)).astype(np.int64, copy=False)
            cand_heap = [(t_base[i], int(i)) for i in cand_idx]
            heapq.heapify(cand_heap)
            rel_heap = []
            firedQ = np.zeros(n, bool)
            while cand_heap or rel_heap:
                next_cand_t = cand_heap[0][0] if cand_heap else np.inf
                next_rel_t = rel_heap[0][0] if rel_heap else np.inf
                if next_rel_t <= next_cand_t:
                    te, add = heapq.heappop(rel_heap)
                    if te > current_time:
                        A = recover(A, te - current_time)
                        current_time = te
                    A = min(Amax, A + add)
                    continue
                ti, i = heapq.heappop(cand_heap)
                if firedQ[i]:
                    continue
                if ti > current_time:
                    A = recover(A, ti - current_time)
                    current_time = ti
                if A >= cost:
                    A -= cost
                    firedQ[i] = True
                    t_adj[i] = current_time
                    rt = release_guess[i]
                    if np.isfinite(rt) and rt >= current_time:
                        heapq.heappush(rel_heap, (rt, cost))
                else:
                    dtw = wait_time_to_cost(A)
                    if not np.isfinite(dtw):
                        firedQ[i] = True
                        t_adj[i] = np.inf
                        continue
                    heapq.heappush(cand_heap, (current_time + dtw, i))
            return t_adj

        if not ffiring_forkQ:
            t = t0.copy()
            heap_idx = np.flatnonzero(np.isfinite(t)).astype(np.int64, copy=False)
            heap = [(t[i], int(i)) for i in heap_idx]
            heapq.heapify(heap)
            done = np.zeros(n, bool)
            A = Amax
            current_time = 0.0
            t_adj = t.copy()
            while heap:
                ti, i = heapq.heappop(heap)
                if done[i]:
                    continue
                if ti > current_time:
                    A = recover(A, ti - current_time)
                    current_time = ti
                if A >= cost:
                    A -= cost
                    done[i] = True
                    t_adj[i] = current_time
                else:
                    dtw = wait_time_to_cost(A)
                    if not np.isfinite(dtw):
                        t_adj[i] = np.inf
                        done[i] = True
                        continue
                    current_time = current_time + dtw
                    A = recover(A, dtw)
                    heapq.heappush(heap, (current_time, i))
            t = t_adj
        else:
            t = t0.copy()
            release_guess = np.full(n, np.inf, float)
            t = schedule_with_releases(t0, release_guess)
            for _ in range(3):
                rep_time_tmp, src_tmp = propagate_from_times(t)
                if not perQ:
                    changes = np.flatnonzero(np.r_[True, src_tmp[1:] != src_tmp[:-1], True])
                    starts = changes[:-1]
                    ends = changes[1:] - 1
                    labels = src_tmp[starts].astype(np.int64, copy=False)
                    release_guess.fill(np.inf)
                    release_guess[labels] = np.maximum(rep_time_tmp[starts], rep_time_tmp[ends])
                else:
                    release_guess.fill(np.inf)
                    fired = np.flatnonzero((src_tmp == np.arange(n, dtype=np.int32)) & np.isfinite(t)).astype(np.int64, copy=False)
                    for ori in fired:
                        seg = np.flatnonzero(src_tmp == ori)
                        if seg.size:
                            release_guess[ori] = np.nanmax(rep_time_tmp[seg])
                t_new = schedule_with_releases(t0, release_guess)
                if np.allclose(np.nan_to_num(t_new, nan=np.inf, posinf=np.inf),
                               np.nan_to_num(t, nan=np.inf, posinf=np.inf),
                               rtol=0, atol=1e-12):
                    t = t_new
                    break
                t = t_new
    else:
        t = t0

    # propagation with prev
    if not perQ:
        g = np.empty(n, float)
        src = np.empty(n, np.int32)
        prev = np.full(n, -1, np.int32)
        g[0], src[0], prev[0] = t[0], 0, 0
        for k in range(1, n):
            cand = g[k - 1] + _edge_forward(g[k - 1], k)
            if t[k] <= cand:
                g[k], src[k], prev[k] = t[k], k, k
            else:
                g[k], src[k], prev[k] = cand, src[k - 1], k - 1
        for k in range(n - 2, -1, -1):
            cand = g[k + 1] + _edge_backward(g[k + 1], k)
            if cand < g[k]:
                g[k], src[k], prev[k] = cand, src[k + 1], k + 1
    else:
        g = t.copy()
        src = np.arange(n, dtype=np.int32)
        prev = np.arange(n, dtype=np.int32)
        visited = np.zeros(n, bool)
        pq = [(g[i], i) for i in range(n) if np.isfinite(g[i])]
        heapq.heapify(pq)
        while pq:
            time_u, u = heapq.heappop(pq)
            if visited[u] or time_u > g[u]:
                continue
            visited[u] = True
            v = (u + 1) % n
            w = _edge_forward(time_u, v)
            if np.isfinite(w):
                cand = time_u + w
                if cand < g[v]:
                    g[v] = cand
                    src[v] = src[u]
                    prev[v] = u
                    heapq.heappush(pq, (cand, v))
            v = (u - 1) % n
            w = _edge_backward(time_u, v)
            if np.isfinite(w):
                cand = time_u + w
                if cand < g[v]:
                    g[v] = cand
                    src[v] = src[u]
                    prev[v] = u
                    heapq.heappush(pq, (cand, v))

    idx = np.arange(n, dtype=np.int32)
    fired_mask = (src == idx) & np.isfinite(t)
    fired_idx = np.flatnonzero(fired_mask).astype(np.int64, copy=False)

    fork_directionality = np.zeros(n, float)
    if not perQ:
        fork_directionality[prev == (idx - 1)] = 1.0
        fork_directionality[prev == (idx + 1)] = -1.0
    else:
        fork_directionality[prev == ((idx - 1) % n)] = 1.0
        fork_directionality[prev == ((idx + 1) % n)] = -1.0

    fired_times_min = t[fired_idx]
    fired_pos_kb = fired_idx.astype(float) * resolution_space

    if fired_idx.size >= 2:
        fired_sorted = np.sort(fired_idx)
        diffs = np.diff(fired_sorted)
        if perQ:
            wrap = n - fired_sorted[-1] + fired_sorted[0]
            interorigin_dist_kb = np.concatenate([diffs, [wrap]]).astype(float) * resolution_space
            if interorigin_dist_kb.size != fired_sorted.size:
                raise RuntimeError("perQ=True: expected len(IODs) == number of fired origins.")
        else:
            interorigin_dist_kb = diffs.astype(float) * resolution_space
    else:
        interorigin_dist_kb = np.array([], float)

    replication_time_min = g
    if max_rep_time is not None:
        replication_time_min = np.minimum(replication_time_min, float(max_rep_time))

    # --- time statistics (1D only, stays here) ---
    time_stats = None
    if time_statsQ:
        if time_grid is None:
            raise ValueError("time_statsQ=True requires a time_grid array.")
        time_grid = np.asarray(time_grid, dtype=float)
        t_max_grid = float(time_grid[-1])

        def count_from_intervals(starts, ends):
            starts = np.asarray(starts, float)
            ends = np.asarray(ends, float)
            if starts.size == 0:
                return np.zeros_like(time_grid, float)
            starts.sort()
            ends.sort()
            started = np.searchsorted(starts, time_grid, side="right")
            ended = np.searchsorted(ends, time_grid, side="left")
            return (started - ended).astype(float)

        # build fork/stall intervals via traversal edges (cheap vs mÃ—n)
        pk = prev.astype(np.int64, copy=False)
        good = (pk >= 0) & np.isfinite(replication_time_min) & np.isfinite(replication_time_min[pk])
        cols = np.flatnonzero(good).astype(np.int64, copy=False)
        edge_start = replication_time_min[pk[cols]]
        edge_end = replication_time_min[cols]

        fork_ok = np.isfinite(edge_start) & np.isfinite(edge_end) & (edge_start < edge_end) & (edge_start <= t_max_grid)
        fork_starts = edge_start[fork_ok].tolist()
        fork_ends = np.minimum(edge_end[fork_ok], t_max_grid).tolist()

        d = fork_directionality[cols]
        delay = np.where(d > 0, stall_delay_left[cols], stall_delay_right[cols])
        stall_ok = np.isfinite(delay) & (delay > 0.0) & np.isfinite(edge_end) & (edge_end <= t_max_grid)
        stall_starts = (edge_end[stall_ok] - delay[stall_ok]).tolist()
        stall_ends = np.minimum(edge_end[stall_ok], t_max_grid).tolist()

        total_forks = count_from_intervals(fork_starts, fork_ends)
        stalled_forks = count_from_intervals(stall_starts, stall_ends)
        active_forks = total_forks - stalled_forks

        # firing_factors 1D (as before)
        firing_factors = None
        if use_factors:
            Amax = float(ffiring_nr)
            cost = 2.0
            firing_factors = np.empty_like(time_grid, float)

            if ffiring_forkQ:
                release = np.full(n, np.inf, float)
                if not perQ:
                    changes = np.flatnonzero(np.r_[True, src[1:] != src[:-1], True])
                    starts = changes[:-1]
                    ends = changes[1:] - 1
                    labels = src[starts].astype(np.int64, copy=False)
                    release[labels] = np.maximum(replication_time_min[starts], replication_time_min[ends])
                else:
                    for ori in fired_idx:
                        seg = np.flatnonzero(src == ori)
                        if seg.size:
                            release[ori] = np.nanmax(replication_time_min[seg])

                fire_events = np.sort(fired_times_min[np.isfinite(fired_times_min)])
                rel_events = np.sort(release[fired_idx][np.isfinite(release[fired_idx])])
                A = Amax
                fi = 0
                ri = 0
                for j, tj in enumerate(time_grid):
                    while fi < fire_events.size and fire_events[fi] <= tj:
                        A = max(0.0, A - cost)
                        fi += 1
                    while ri < rel_events.size and rel_events[ri] <= tj:
                        A = min(Amax, A + cost)
                        ri += 1
                    firing_factors[j] = A
            else:
                r = float(ffiring_recycle)
                A = Amax
                last_t = 0.0
                events = np.sort(fired_times_min[np.isfinite(fired_times_min)])

                def _recover(A_now, dt):
                    if r == 0.0 or dt <= 0.0:
                        return A_now
                    return Amax - (Amax - A_now) * np.exp(-r * dt)

                ei = 0
                for j, tj in enumerate(time_grid):
                    while ei < events.size and events[ei] <= tj:
                        te = events[ei]
                        A = _recover(A, te - last_t)
                        A = max(0.0, A - cost)
                        last_t = te
                        ei += 1
                    firing_factors[j] = _recover(A, tj - last_t)

        time_stats = dict(
            time_grid=time_grid,
            total_forks=total_forks,
            active_forks=active_forks,
            stalled_forks=stalled_forks,
            firing_factors=firing_factors,
        )

    return (
        fired_idx.astype(np.int64, copy=False),
        fired_pos_kb,
        fired_times_min,
        replication_time_min,
        interorigin_dist_kb,
        time_stats,
        fork_directionality,
        # NEW minimal extra return for post dens compilation (no mÃ—n arrays):
        t, src, prev,
    )

def rsim(
    firing_rate=None,
    fork_speed=1.4,
    sim_number=50,
    resolution_space=1.0,
    resolution_time=1.0,
    perQ=False,
    stall_rate=0.0,
    tau=np.inf,
    ffiring_nr=None,
    ffiring_recycle=0.0,
    ffiring_forkQ=False,
    time_statsQ=False,
    time_stats_xtQ=False,
    time_stats_densQ=None,  # deprecated alias; kept for backwards compatibility
    time_grid=np.arange(0.0, 1500.0 + 1.0, 1.0),
    max_rep_time=1200.0,
    seed=None,
    verbose=True,
    print_every=1,
    dens_block_n=4096,
    dens_memmap_dir=None,
    n_bins=None,
    ori_rate=None,
    initiation_rate=None,
    geometry=None,
    resolution=None,
):
    """Run replicate stochastic simulations and return averaged features.

    ``firing_rate`` is the preferred name for origin-initiation propensity.
    ``ori_rate`` and ``initiation_rate`` are accepted as compatibility aliases.
    Firing rate and fork speed support four field forms: scalar, ``(n,)``,
    ``(1, T)``, and ``(T, n)``. Use ``n_bins`` when both kinetic inputs lack a
    spatial axis.
    """
    if time_stats_densQ is not None:
        time_stats_xtQ = bool(time_stats_densQ)

    firing_rate = _normalise_firing_rate(
        firing_rate=firing_rate,
        ori_rate=ori_rate,
        initiation_rate=initiation_rate,
    )

    if resolution is not None:
        resolution_space = resolution

    if geometry is not None:
        if not isinstance(geometry, str):
            raise ValueError("geometry must be 'linear' or 'periodic'.")
        g = geometry.strip().lower()
        if g in {"linear", "chromosome", "open"}:
            perQ = False
        elif g in {"periodic", "circular", "circle"}:
            perQ = True
        else:
            raise ValueError("geometry must be 'linear' or 'periodic'.")

    n = _resolve_n_bins(firing_rate, fork_speed, n_bins=n_bins)

    sim_number = int(sim_number)

    ss = np.random.SeedSequence(seed) if seed is not None else np.random.SeedSequence()
    generators = [np.random.default_rng(s) for s in ss.spawn(sim_number)]

    sum_rep_time = np.zeros(n, dtype=float)
    count_rep_time = np.zeros(n, dtype=np.int64)

    rep_times_per_sim = np.empty((sim_number, n), dtype=float)
    nr_oris = np.empty(sim_number, dtype=np.int64)

    all_iods = []
    fire_counts = np.zeros(n, dtype=np.int64)

    sum_fdir = np.zeros(n, dtype=float)
    count_fdir = np.zeros(n, dtype=np.int64)

    need_time_grid = time_statsQ or time_stats_xtQ
    if need_time_grid:
        if time_grid is None:
            raise ValueError("time_statsQ/time_stats_xtQ require a time_grid array.")
        time_grid = np.asarray(time_grid, dtype=float)
        m = int(time_grid.size)
        dt_grid_out = float(time_grid[1] - time_grid[0]) if m >= 2 else 1.0
        sum_replicated_fraction = np.zeros(m, float)
    else:
        time_grid = None
        dt_grid_out = None
        m = None
        sum_replicated_fraction = None

    if time_statsQ:
        sum_total_forks = np.zeros(m, float)
        sum_active_forks = np.zeros(m, float)
        sum_stalled_forks = np.zeros(m, float)
        sum_firing_factors = np.zeros(m, float)
        count_firing_factors = 0
    else:
        sum_total_forks = None
        sum_active_forks = None
        sum_stalled_forks = None
        sum_firing_factors = None
        count_firing_factors = 0

    # --- allocate space-time accumulators (disk-backed) ---
    if time_stats_xtQ:
        if dens_memmap_dir is None:
            dens_memmap_dir = tempfile.gettempdir()
        os.makedirs(dens_memmap_dir, exist_ok=True)

        def _mm(name, shape, dtype):
            path = os.path.join(dens_memmap_dir, f"{name}_{int(time.time()*1e6)}.dat")
            return np.memmap(path, mode="w+", dtype=dtype, shape=shape)

        # store sums as float32 (you can change to uint16 if sim_number <= 65535)
        sum_replicated_fraction_xt = _mm("sum_replicated_fraction_xt", (m, n), np.float32)
        sum_initiation_events_xt = _mm("sum_initiation_events_xt", (m, n), np.float32)
        sum_coalescence_events_xt = _mm("sum_coalescence_events_xt", (m, n), np.float32)
        sum_right_moving_fork_density_xt = _mm("sum_right_moving_fork_density_xt", (m, n), np.float32)
        sum_left_moving_fork_density_xt = _mm("sum_left_moving_fork_density_xt", (m, n), np.float32)

        sum_replicated_fraction_xt[:] = 0.0
        sum_initiation_events_xt[:] = 0.0
        sum_coalescence_events_xt[:] = 0.0
        sum_right_moving_fork_density_xt[:] = 0.0
        sum_left_moving_fork_density_xt[:] = 0.0

        dens_block_n = int(dens_block_n)
        if dens_block_n <= 0:
            raise ValueError("dens_block_n must be a positive integer.")
    else:
        sum_replicated_fraction_xt = None
        sum_initiation_events_xt = None
        sum_coalescence_events_xt = None
        sum_right_moving_fork_density_xt = None
        sum_left_moving_fork_density_xt = None

    start_time = time.time()

    for i, g in enumerate(generators, start=1):
        fired_idx, _, _, rep_time, iod, time_stats, fdir, t_fire, src, prev = simulate_replication(
            firing_rate=firing_rate,
            fork_speed=fork_speed,
            resolution_space=resolution_space,
            resolution_time=resolution_time,
            rng=g,
            perQ=perQ,
            stall_rate=stall_rate,
            tau=tau,
            ffiring_nr=ffiring_nr,
            ffiring_recycle=ffiring_recycle,
            ffiring_forkQ=ffiring_forkQ,
            time_statsQ=time_statsQ,
            time_stats_xtQ=time_stats_xtQ,
            time_stats_densQ=time_stats_densQ,
            time_grid=time_grid,
            max_rep_time=max_rep_time,
            n_bins=n,
        )

        rep_times_per_sim[i - 1, :] = rep_time
        nr_oris[i - 1] = int(fired_idx.size)

        finite_mask = np.isfinite(rep_time)
        sum_rep_time[finite_mask] += rep_time[finite_mask]
        count_rep_time[finite_mask] += 1

        if iod.size:
            all_iods.append(iod)
        if fired_idx.size:
            fire_counts[fired_idx] += 1

        sum_fdir[finite_mask] += fdir[finite_mask]
        count_fdir[finite_mask] += 1

        # --- 1D replicated-fraction time series (RAM) ---
        if time_grid is not None:
            rep_sorted = np.sort(rep_time[finite_mask])
            if rep_sorted.size:
                sum_replicated_fraction += (
                    np.searchsorted(rep_sorted, time_grid, side="right") / float(n)
                )

        # --- 1D time series accumulate (RAM) ---
        if time_statsQ and time_stats is not None:
            sum_total_forks += time_stats["total_forks"]
            sum_active_forks += time_stats["active_forks"]
            sum_stalled_forks += time_stats["stalled_forks"]
            ff = time_stats.get("firing_factors", None)
            if ff is not None:
                sum_firing_factors += ff
                count_firing_factors += 1

        # --- 2D space-time stats accumulate (disk memmap, block-by-block, no giant temporaries) ---
        if time_stats_xtQ:
            t_max_grid = float(time_grid[-1])

            # (a) replicated fraction in (t,x)
            for x0 in range(0, n, dens_block_n):
                x1 = min(n, x0 + dens_block_n)
                blk = (rep_time[x0:x1][None, :] <= time_grid[:, None])
                sum_replicated_fraction_xt[:, x0:x1] += blk.astype(np.float32)

            # (b) initiation events: sparse binning
            tf = t_fire
            ok = np.isfinite(tf) & (tf <= t_max_grid)
            if np.any(ok):
                cols = np.flatnonzero(ok).astype(np.int64, copy=False)
                bins = np.searchsorted(time_grid, tf[ok], side="left").astype(np.int64, copy=False)
                np.add.at(sum_initiation_events_xt, (bins, cols), 1.0)

            # (c) coalescence events: sparse binning on boundaries
            if not perQ:
                boundary_k = (np.flatnonzero(src[1:] != src[:-1]) + 1).astype(np.int64, copy=False)
                if boundary_k.size:
                    km1 = boundary_k - 1
                    t_meet = np.maximum(rep_time[km1], rep_time[boundary_k])
                    okm = np.isfinite(t_meet) & (t_meet <= t_max_grid)
                    if np.any(okm):
                        bins = np.searchsorted(time_grid, t_meet[okm], side="left").astype(np.int64, copy=False)
                        cols = boundary_k[okm].astype(np.int64, copy=False)
                        np.add.at(sum_coalescence_events_xt, (bins, cols), 1.0)
            else:
                boundary_k = np.flatnonzero(src[np.r_[1:n, 0]] != src).astype(np.int64, copy=False)
                if boundary_k.size:
                    km1 = (boundary_k - 1) % n
                    t_meet = np.maximum(rep_time[km1], rep_time[boundary_k])
                    okm = np.isfinite(t_meet) & (t_meet <= t_max_grid)
                    if np.any(okm):
                        bins = np.searchsorted(time_grid, t_meet[okm], side="left").astype(np.int64, copy=False)
                        cols = boundary_k[okm].astype(np.int64, copy=False)
                        np.add.at(sum_coalescence_events_xt, (bins, cols), 1.0)

            # (d) fork densities in (t,x)
            pk = prev.astype(np.int64, copy=False)
            good = (pk >= 0) & np.isfinite(rep_time) & np.isfinite(rep_time[pk])
            cols0 = np.flatnonzero(good).astype(np.int64, copy=False)
            if cols0.size:
                t_end = rep_time[cols0]
                t_start = rep_time[pk[cols0]]
                ok2 = (t_start < t_end) & (t_start <= t_max_grid)
                cols0 = cols0[ok2]
                if cols0.size:
                    t_start = t_start[ok2]
                    t_end = np.minimum(t_end[ok2], t_max_grid)
                    j0 = np.searchsorted(time_grid, t_start, side="left").astype(np.int64, copy=False)
                    j1 = np.searchsorted(time_grid, t_end, side="left").astype(np.int64, copy=False)
                    ok3 = j1 > j0
                    cols0 = cols0[ok3]
                    if cols0.size:
                        j0 = j0[ok3]
                        j1 = j1[ok3]
                        d = fdir[cols0]
                        Rm = d > 0
                        Lm = d < 0

                        def _range_add(target_mm, cols, a, b):
                            for kk in range(cols.size):
                                c = int(cols[kk])
                                target_mm[a[kk]:b[kk], c] += 1.0

                        if np.any(Rm):
                            _range_add(sum_right_moving_fork_density_xt, cols0[Rm], j0[Rm], j1[Rm])
                        if np.any(Lm):
                            _range_add(sum_left_moving_fork_density_xt, cols0[Lm], j0[Lm], j1[Lm])

        if verbose and (i % int(print_every) == 0 or i == int(sim_number)):
            etaf(it=i, maxiter=int(sim_number), start_time=start_time)

    all_iods_kb = np.concatenate(all_iods) if all_iods else np.array([], dtype=float)

    avg_rep_time_min = np.full(n, np.inf, dtype=float)
    nonzero = count_rep_time > 0
    avg_rep_time_min[nonzero] = sum_rep_time[nonzero] / count_rep_time[nonzero].astype(float)

    fork_directionality = np.full(n, np.nan, float)
    fd_ok = count_fdir > 0
    fork_directionality[fd_ok] = sum_fdir[fd_ok] / count_fdir[fd_ok].astype(float)

    s_phase_durations_min = rep_times_per_sim.max(axis=1)
    efficiencies = fire_counts.astype(float) / float(sim_number)

    avg_replicated_fraction = (
        sum_replicated_fraction / float(sim_number)
        if sum_replicated_fraction is not None
        else None
    )

    # --- finalise outputs ---
    if time_statsQ:
        invS = 1.0 / float(sim_number)
        avg_total_forks = sum_total_forks * invS
        avg_active_forks = sum_active_forks * invS
        avg_stalled_forks = sum_stalled_forks * invS
        avg_firing_factors = (sum_firing_factors / float(count_firing_factors)) if count_firing_factors > 0 else None
    else:
        avg_total_forks = None
        avg_active_forks = None
        avg_stalled_forks = None
        avg_firing_factors = None

    if time_stats_xtQ:
        post_start = time.time()
        for j, x0 in enumerate(range(0, n, dens_block_n), start=1):
            x1 = min(n, x0 + dens_block_n)
            sum_replicated_fraction_xt[:, x0:x1] /= float(sim_number)
            sum_initiation_events_xt[:, x0:x1] /= float(sim_number)
            sum_coalescence_events_xt[:, x0:x1] /= float(sim_number)
            sum_right_moving_fork_density_xt[:, x0:x1] /= float(sim_number)
            sum_left_moving_fork_density_xt[:, x0:x1] /= float(sim_number)
            if verbose:
                etaf(it=j, maxiter=(n + dens_block_n - 1) // dens_block_n, start_time=post_start)

        avg_replicated_fraction_xt = sum_replicated_fraction_xt
        avg_initiation_events_xt = sum_initiation_events_xt
        avg_coalescence_events_xt = sum_coalescence_events_xt
        avg_right_moving_fork_density_xt = sum_right_moving_fork_density_xt
        avg_left_moving_fork_density_xt = sum_left_moving_fork_density_xt
        dt_grid_out = float(time_grid[1] - time_grid[0]) if time_grid is not None and time_grid.size >= 2 else 1.0
    else:
        avg_replicated_fraction_xt = None
        avg_initiation_events_xt = None
        avg_coalescence_events_xt = None
        avg_right_moving_fork_density_xt = None
        avg_left_moving_fork_density_xt = None

    return dict(
        replication_timing=avg_rep_time_min,
        rep_times_per_sim=rep_times_per_sim,
        s_phase_duration=s_phase_durations_min,
        completion_time=s_phase_durations_min,
        num_oris=nr_oris.tolist(),
        num_origins=nr_oris.tolist(),
        inter_origin_distances=all_iods_kb,
        inter_origin_distance=all_iods_kb,
        efficiency=efficiencies,
        origin_efficiency=efficiencies,
        fork_directionality=fork_directionality,
        time_grid=time_grid,
        dt_grid=dt_grid_out,
        replicated_fraction=avg_replicated_fraction,
        replicated_fraction_xt=avg_replicated_fraction_xt,
        initiation_events_xt=avg_initiation_events_xt,
        coalescence_events_xt=avg_coalescence_events_xt,
        right_moving_fork_density_xt=avg_right_moving_fork_density_xt,
        right_fork_density_xt=avg_right_moving_fork_density_xt,
        left_moving_fork_density_xt=avg_left_moving_fork_density_xt,
        left_fork_density_xt=avg_left_moving_fork_density_xt,
        total_forks=avg_total_forks,
        active_forks=avg_active_forks,
        stalled_forks=avg_stalled_forks,
        firing_factors=avg_firing_factors,
    )
