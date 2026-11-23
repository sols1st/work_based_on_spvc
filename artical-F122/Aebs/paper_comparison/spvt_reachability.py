"""Controlled SPVT-style finite-horizon reachability, not native SPVT training.

This backend freezes the observation contract and controller supplied by tasks.py.
Every accepted step encloses ALL allowed observation errors and process noise.
Linear tasks use unreduced zonotopes (exact Minkowski propagation in real
arithmetic). Other tasks use interval enclosures with adaptive predecessor/error
partitioning. An enclosure touching danger is UNKNOWN, not a counterexample.

Floating-point enclosure reliability is inherited from the task adapter; this
module deliberately does not advertise a machine-checked formal proof.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReachabilityConfig:
    mode: str = "auto"
    timeout_seconds: float = 5.0
    max_nodes: int = 20000
    max_frontier: int = 256
    max_depth: int = 16
    split_lookahead: bool = True
    falsification_paths: int = 24
    seed: int = 20261118


class BudgetExhausted(Exception):
    pass


class Budget:
    def __init__(self, config: ReachabilityConfig):
        self.config = config
        self.start = time.perf_counter()
        self.nodes = 0

    def check(self, count: int = 0):
        if self.config.timeout_seconds > 0 and self.elapsed >= self.config.timeout_seconds:
            raise BudgetExhausted("timeout")
        if self.config.max_nodes > 0 and self.nodes + count > self.config.max_nodes:
            raise BudgetExhausted("node_budget")
        self.nodes += count

    @property
    def elapsed(self):
        return time.perf_counter() - self.start


def _array(value):
    return np.asarray(value, dtype=np.float64)


def _classification(task, lo, hi, t, context):
    # The common adapter API has no context on this method; optionally allow it.
    return task.classify_safe(lo, hi, t=t)


def _step_bound(task, lo, hi, elo, ehi, nlo, nhi, context, t, budget):
    budget.check(1)
    left, right = task.interval_step(
        lo, hi, error_lo=elo, error_hi=ehi, noise_lo=nlo, noise_hi=nhi,
        context=context, t=t,
    )
    left, right = _array(left), _array(right)
    if left.shape != lo.shape or right.shape != hi.shape:
        raise ValueError("interval_step returned the wrong state shape")
    if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
        raise ValueError("interval_step returned nonfinite bounds")
    if np.any(left > right):
        raise ValueError("interval_step returned reversed bounds")
    return left, right


def _make_result(task, initial, context, horizon, status, certified_steps, budget, **extra):
    return dict(
        task=task.name, initial_state=_array(initial).tolist(), context=context,
        horizon=int(horizon), requested_seconds=float(horizon * task.dt),
        status=status, certified_steps=int(certified_steps),
        certified_seconds=float(certified_steps * task.dt),
        interval_evaluations=budget.nodes, bound_evaluations=budget.nodes, wall_seconds=budget.elapsed,
        **extra,
    )


def _facet_safe(task, center, generators, context, t):
    """Use exact linear support rather than boxing coupled safety constraints."""
    if hasattr(task, "classify_zonotope"):
        return task.classify_zonotope(center, generators, context=context, t=t)
    if hasattr(task, "safe_linear_constraints"):
        constraints = task.safe_linear_constraints(context=context, t=t)
        if constraints is not None:
            H, b = map(_array, constraints)
            upper = H @ center + np.abs(H @ generators).sum(axis=1)
            # Some tasks define equality as dangerous. Strict comparison is
            # conservative for both conventions and avoids changing the task.
            return "safe" if np.all(upper < b) else "unknown"
    radius = np.abs(generators).sum(axis=1)
    return _classification(task, center - radius, center + radius, t, context)


def _verify_zonotope(task, initial, context, horizon, config, budget, initial_hi=None):
    initial_hi = _array(initial if initial_hi is None else initial_hi)
    center = (_array(initial) + initial_hi) / 2
    generators = np.diag((initial_hi - _array(initial)) / 2)
    generators = generators[:, np.any(generators != 0, axis=0)]
    elo, ehi = _array(task.error_lo), _array(task.error_hi)
    nlo, nhi = _array(task.noise_lo), _array(task.noise_hi)
    ec, er = (elo + ehi) / 2, (ehi - elo) / 2
    nc, nr = (nlo + nhi) / 2, (nhi - nlo) / 2
    certified, max_generators = 0, 0
    try:
        for t in range(horizon):
            budget.check(1)
            radius = np.abs(generators).sum(axis=1)
            if hasattr(task, "affine_matrices_zonotope"):
                matrices = task.affine_matrices_zonotope(center, generators, context=context, t=t)
            else:
                matrices = task.affine_matrices(center - radius, center + radius, context=context, t=t)
            if matrices is None:
                return _make_result(task, initial, context, horizon, "unknown_nonlinear_region", certified,
                                    budget, method="unreduced_zonotope", max_generators=max_generators)
            A, B, D, c = map(_array, matrices)
            center = A @ center + B @ ec + D @ nc + c
            new_generators = [A @ generators, B * er[None, :], D * nr[None, :]]
            generators = np.concatenate(new_generators, axis=1)
            # Removing exactly zero columns is exact, unlike order reduction.
            generators = generators[:, np.any(generators != 0, axis=0)]
            max_generators = max(max_generators, generators.shape[1])
            if _facet_safe(task, center, generators, context, t + 1) != "safe":
                return _make_result(task, initial, context, horizon, "unknown_enclosure", certified,
                                    budget, method="unreduced_zonotope", max_generators=max_generators,
                                    first_unresolved_step=t + 1)
            certified = t + 1
    except BudgetExhausted as exc:
        return _make_result(task, initial, context, horizon, "unknown_" + str(exc), certified,
                            budget, method="unreduced_zonotope", max_generators=max_generators)
    return _make_result(task, initial, context, horizon, "certified", certified, budget,
                        method="unreduced_zonotope", max_generators=max_generators)


def _split_candidates(node):
    # node is (state lower/upper, error lower/upper, noise lower/upper, depth).
    for pair in (0, 2, 4):
        lo, hi = node[pair], node[pair + 1]
        for axis in np.flatnonzero(hi > lo):
            mid = lo[axis] + (hi[axis] - lo[axis]) / 2
            if mid == lo[axis] or mid == hi[axis]:
                continue
            first, second = list(node), list(node)
            first[pair + 1] = hi.copy()
            first[pair + 1][axis] = mid
            second[pair] = lo.copy()
            second[pair][axis] = mid
            first[6] = second[6] = node[6] + 1
            yield tuple(first), tuple(second)


def _coalesce(task, boxes, cap, t, context):
    """Hull adjacent boxes only; union containment is maintained after merging."""
    if len(boxes) <= cap:
        return boxes, 0
    merges = 0
    while len(boxes) > cap:
        centers = np.asarray([(a + b) / 2 for a, b in boxes])
        scale = np.maximum(_array(task.domain_hi) - _array(task.domain_lo), 1e-12)
        axis = int(np.argmax(np.ptp(centers / scale, axis=0)))
        order = np.argsort(centers[:, axis])
        next_boxes = []
        for index in range(0, len(order), 2):
            if index + 1 == len(order):
                next_boxes.append(boxes[int(order[index])])
                continue
            a, b = boxes[int(order[index])], boxes[int(order[index + 1])]
            lo, hi = np.minimum(a[0], b[0]), np.maximum(a[1], b[1])
            # All actual children are safe already. Requiring the hull safe also
            # keeps subsequent calls inside the adapter's verified safe domain.
            if _classification(task, lo, hi, t, context) != "safe":
                return None, merges
            next_boxes.append((lo, hi))
            merges += 1
        boxes = next_boxes
    return boxes, merges


def _verify_boxes(task, initial, context, horizon, config, budget, initial_hi=None):
    initial_hi = initial if initial_hi is None else initial_hi
    frontier = [(_array(initial).copy(), _array(initial_hi).copy())]
    elo, ehi = _array(task.error_lo), _array(task.error_hi)
    nlo, nhi = _array(task.noise_lo), _array(task.noise_hi)
    scales = np.maximum(_array(task.domain_hi) - _array(task.domain_lo), 1e-12)
    certified, splits, merges, max_frontier = 0, 0, 0, 1
    try:
        for t in range(horizon):
            next_frontier = []
            for lo, hi in frontier:
                pending = [((lo, hi, elo, ehi, nlo, nhi, 0), None)]
                while pending:
                    budget.check()
                    node, cached_bound = pending.pop()
                    bound = cached_bound or _step_bound(task, *node[:6], context, t, budget)
                    if _classification(task, *bound, t + 1, context) == "safe":
                        next_frontier.append(bound)
                        continue
                    if node[6] >= config.max_depth:
                        return _make_result(task, initial, context, horizon, "unknown_partition_depth", certified,
                                            budget, method="partitioned_intervals", splits=splits, merges=merges,
                                            max_frontier=max_frontier, first_unresolved_step=t + 1)
                    best = None
                    for children in _split_candidates(node):
                        child_bounds = [_step_bound(task, *child[:6], context, t, budget) for child in children]
                        safe_count = sum(_classification(task, *b, t + 1, context) == "safe" for b in child_bounds)
                        width = max(float(np.sum((b[1] - b[0]) / scales)) for b in child_bounds)
                        score = (safe_count, -width)
                        if best is None or score > best[0]:
                            best = (score, children, child_bounds)
                        if safe_count == 2 or not config.split_lookahead:
                            break
                    if best is None:
                        return _make_result(task, initial, context, horizon, "unknown_unsplittable", certified,
                                            budget, method="partitioned_intervals", splits=splits, merges=merges,
                                            max_frontier=max_frontier, first_unresolved_step=t + 1)
                    splits += 1
                    pending.extend(zip(best[1], best[2]))
            max_frontier = max(max_frontier, len(next_frontier))
            certified = t + 1
            frontier, added_merges = _coalesce(task, next_frontier, config.max_frontier, t + 1, context)
            merges += added_merges
            if frontier is None:
                return _make_result(task, initial, context, horizon, "unknown_frontier_budget", certified,
                                    budget, method="partitioned_intervals", splits=splits, merges=merges,
                                    max_frontier=max_frontier)
    except BudgetExhausted as exc:
        return _make_result(task, initial, context, horizon, "unknown_" + str(exc), certified,
                            budget, method="partitioned_intervals", splits=splits, merges=merges,
                            max_frontier=max_frontier)
    return _make_result(task, initial, context, horizon, "certified", certified, budget,
                        method="partitioned_intervals", splits=splits, merges=merges, max_frontier=max_frontier)


def find_counterexample(task, initial, context, horizon, paths, seed, max_seconds=1.0):
    """Replay valid disturbances; only actual point violations are witnesses.

    This is a falsifier, never a safety proof. It runs after, and outside, the
    enclosure budget; its measured time is included separately and in total.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    el, eh = _array(task.error_lo), _array(task.error_hi)
    nl, nh = _array(task.noise_lo), _array(task.noise_hi)
    dims = len(el) + len(nl)
    if not paths:
        return None, time.perf_counter() - start
    corners = list(itertools.islice(itertools.product((-1.0, 1.0), repeat=dims), max(0, paths - 1)))
    for p in range(paths):
        x = _array(initial).copy()
        error_trace, noise_trace = [], []
        for t in range(horizon):
            if max_seconds > 0 and time.perf_counter() - start >= max_seconds:
                return None, time.perf_counter() - start
            if p == 0:
                signs = np.zeros(dims)
            elif p <= len(corners):
                signs = np.asarray(corners[p - 1])
            else:
                signs = rng.choice((-1.0, 1.0), size=dims)
            error = (el + eh) / 2 + (eh - el) / 2 * signs[:len(el)]
            noise = (nl + nh) / 2 + (nh - nl) / 2 * signs[len(el):]
            x = _array(task.step(x, error=error, noise=noise, context=context, t=t))
            error_trace.append(error.tolist())
            noise_trace.append(noise.tolist())
            if _classification(task, x, x, t + 1, context) == "unsafe":
                return dict(step=t + 1, state=x.tolist(), errors=error_trace, noise=noise_trace,
                            note="Point replay in the controlled low-dimensional error model, not a real-camera violation."), time.perf_counter() - start
    return None, time.perf_counter() - start


def verify_initial(task, initial, context=None, horizon=None, config=None):
    config = config or ReachabilityConfig()
    horizon = task.horizon if horizon is None else int(horizon)
    if horizon < 0 or config.max_frontier < 1 or config.max_depth < 0:
        raise ValueError("Invalid horizon or reachability budget")
    budget = Budget(config)
    if _classification(task, _array(initial), _array(initial), 0, context) != "safe":
        return _make_result(task, initial, context, horizon, "initial_not_safe", -1, budget)
    mode = config.mode
    if mode not in ("auto", "zonotope", "interval"):
        raise ValueError("mode must be auto, zonotope or interval")
    if mode == "zonotope" and not hasattr(task, "affine_matrices"):
        raise ValueError("Task does not provide affine_matrices")
    if mode != "interval" and hasattr(task, "affine_matrices"):
        result = _verify_zonotope(task, initial, context, horizon, config, budget)
        if mode == "auto" and result["status"] == "unknown_nonlinear_region":
            prefix = result["certified_steps"]
            result = _verify_boxes(task, initial, context, horizon, config, budget)
            result["method"] = "zonotope_then_partitioned_intervals"
            result["zonotope_prefix_steps"] = prefix
            result["certified_steps"] = max(result["certified_steps"], prefix)
            result["certified_seconds"] = result["certified_steps"] * task.dt
    else:
        result = _verify_boxes(task, initial, context, horizon, config, budget)
    if result["status"] != "certified":
        remaining = max(0.0, config.timeout_seconds - budget.elapsed) if config.timeout_seconds > 0 else 1.0
        witness, elapsed = find_counterexample(task, initial, context, horizon,
                                               config.falsification_paths if remaining > 0 else 0,
                                               config.seed, max_seconds=remaining)
        result["enclosure_status"] = result["status"]
        result["falsification_seconds"] = elapsed
        result["wall_seconds"] += elapsed
        if witness is not None:
            if witness["step"] <= result["certified_steps"]:
                raise AssertionError("A replay contradicts a certified prefix: adapter or bound bug")
            result["status"] = "counterexample"
            result["counterexample"] = witness
    return result


def verify_initial_box(task, lo=None, hi=None, context=None, horizon=None, config=None):
    """Separate whole-domain experiment, never counted as N independent trials."""
    config = config or ReachabilityConfig(timeout_seconds=300.0)
    lo, hi = _array(task.initial_lo if lo is None else lo), _array(task.initial_hi if hi is None else hi)
    horizon = task.horizon if horizon is None else int(horizon)
    budget = Budget(config)
    if np.any(lo > hi):
        raise ValueError("Reversed initial box")
    if _classification(task, lo, hi, 0, context) != "safe":
        result = _make_result(task, lo, context, horizon, "unknown_initial_domain", -1, budget)
    elif config.mode != "interval" and hasattr(task, "affine_matrices"):
        result = _verify_zonotope(task, lo, context, horizon, config, budget, initial_hi=hi)
        if config.mode == "auto" and result["status"] == "unknown_nonlinear_region":
            prefix = result["certified_steps"]
            result = _verify_boxes(task, lo, context, horizon, config, budget, initial_hi=hi)
            result["method"] = "zonotope_then_partitioned_intervals"
            result["zonotope_prefix_steps"] = prefix
            result["certified_steps"] = max(prefix, result["certified_steps"])
            result["certified_seconds"] = result["certified_steps"] * task.dt
    else:
        result = _verify_boxes(task, lo, context, horizon, config, budget, initial_hi=hi)
    result["initial_lo"], result["initial_hi"] = lo.tolist(), hi.tolist()
    result["population"] = "entire_initial_box_separate_from_iid_point_budget"
    return result


def hoeffding_lower_bound(verified, n, delta):
    if n <= 0 or not 0 < delta < 1:
        raise ValueError("Need n > 0 and 0 < delta < 1")
    return max(0.0, verified / n - math.sqrt(math.log(2 / delta) / (2 * n)))


def summarize_results(results, horizons, delta=0.05):
    """Pointwise initial-state bounds, conditional on the common error contract."""
    n = len(results)
    if not n:
        return []
    output = []
    for horizon in horizons:
        count = sum(row["certified_steps"] >= horizon for row in results)
        output.append(dict(horizon=int(horizon), n_initial_states=n, certified=count,
                           certified_fraction=count / n,
                           confidence=1 - delta,
                           hoeffding_lower=hoeffding_lower_bound(count, n, delta),
                           simultaneous_over_horizons_lower=hoeffding_lower_bound(count, n, delta / len(horizons))))
    return output


def evaluate_task(task, n=100, seed=20261118, horizons=None, config=None,
                  total_timeout_seconds=300.0, context_mode="sampled_iid", screening_pass=True):
    config = config or ReachabilityConfig()
    horizons = sorted(set(horizons or [task.horizon]))
    initial = _array(task.sample_initial(n, seed))
    if initial.shape != (n, task.state_dim):
        raise ValueError("sample_initial must return (n, state_dim)")
    start = time.perf_counter()
    if context_mode not in ("sampled_iid", "per_context"):
        raise ValueError("context_mode must be sampled_iid or per_context")
    context_rng = np.random.default_rng(np.random.SeedSequence([seed, 417]))
    if context_mode == "sampled_iid":
        context_indices = context_rng.integers(0, len(task.contexts), n)
        populations = [("sampled_iid", [task.contexts[i] for i in context_indices])]
    else:
        populations = [(context, [context] * n) for context in task.contexts]
    all_contexts = []
    for population, contexts in populations:
        rows = []
        for x, context in zip(initial, contexts):
            remaining = total_timeout_seconds - (time.perf_counter() - start)
            if total_timeout_seconds > 0 and remaining <= 0:
                rows.append(dict(task=task.name, initial_state=x.tolist(), context=context,
                                 horizon=max(horizons), status="unknown_unprocessed_batch_budget",
                                 certified_steps=0, certified_seconds=0, interval_evaluations=0, wall_seconds=0))
                continue
            config_dict = asdict(config)
            if screening_pass:
                # First reach all initial samples cheaply. Expensive refinement
                # must not consume the entire budget on the first few samples.
                config_dict["max_depth"] = 0
                config_dict["falsification_paths"] = 0
            if total_timeout_seconds > 0:
                config_dict["timeout_seconds"] = min(remaining, config.timeout_seconds) if config.timeout_seconds > 0 else remaining
            rows.append(verify_initial(task, x, context, max(horizons), ReachabilityConfig(**config_dict)))
        if screening_pass:
            for i, (x, context) in enumerate(zip(initial, contexts)):
                before = rows[i]
                if before["status"] == "certified":
                    before["screening_only"] = True
                    continue
                remaining = total_timeout_seconds - (time.perf_counter() - start)
                if total_timeout_seconds > 0 and remaining <= 0:
                    before["refinement_skipped_batch_budget"] = True
                    continue
                options = asdict(config)
                if total_timeout_seconds > 0:
                    options["timeout_seconds"] = min(remaining, config.timeout_seconds) if config.timeout_seconds > 0 else remaining
                refined = verify_initial(task, x, context, max(horizons), ReachabilityConfig(**options))
                refined["screening_certified_steps"] = before["certified_steps"]
                refined["screening_wall_seconds"] = before["wall_seconds"]
                refined["certified_steps"] = max(refined["certified_steps"], before["certified_steps"])
                refined["certified_seconds"] = refined["certified_steps"] * task.dt
                refined["wall_seconds"] += before["wall_seconds"]
                refined["interval_evaluations"] += before["interval_evaluations"]
                refined["bound_evaluations"] = refined["interval_evaluations"]
                if refined.get("counterexample", {}).get("step", max(horizons) + 1) <= refined["certified_steps"]:
                    raise AssertionError("Refinement witness contradicts screening certificate")
                rows[i] = refined
        all_contexts.append(dict(context=population, summary=summarize_results(rows, horizons), results=rows))
    joint = []
    if len(all_contexts) > 1:
        joint_rows = [dict(certified_steps=min(c["results"][i]["certified_steps"] for c in all_contexts)) for i in range(n)]
        joint = summarize_results(joint_rows, horizons)
    return dict(
        schema="controlled_spvt_reachability_v1", task=task.name,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation="Frozen low-dimensional front end, controller, dynamics and uncertainty contract. Not native image-generator SPVT retraining.",
        confidence_scope="Bounds concern sampled initial states conditional on the error contract. They are not additional CP safety probabilities.",
        context_scope="Each context is fixed for a trajectory; all-context intersection does not certify arbitrary switching.",
        floating_point_scope="Reliability inherits task adapter; no automatic outward-rounding guarantee.",
        config=asdict(config), seed=seed, metadata=task.metadata,
        total_timeout_seconds=total_timeout_seconds, context_mode=context_mode,
        screening_pass=screening_pass,
        unprocessed_count=sum(row["status"] == "unknown_unprocessed_batch_budget" for c in all_contexts for row in c["results"]),
        initial_states_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
        contexts=all_contexts, all_contexts_joint_summary=joint,
        total_wall_seconds=time.perf_counter() - start,
    )


def evaluate_global_task(task, config=None, total_timeout_seconds=300.0, horizon=None):
    config = config or ReachabilityConfig(timeout_seconds=300.0)
    start, rows = time.perf_counter(), []
    for context in task.contexts:
        remaining = total_timeout_seconds - (time.perf_counter() - start)
        if total_timeout_seconds > 0 and remaining <= 0:
            rows.append(dict(task=task.name, context=context, status="unknown_unprocessed_batch_budget",
                             certified_steps=0, wall_seconds=0))
            continue
        options = asdict(config)
        if total_timeout_seconds > 0:
            options["timeout_seconds"] = min(remaining, config.timeout_seconds) if config.timeout_seconds > 0 else remaining
        rows.append(verify_initial_box(task, context=context, horizon=horizon, config=ReachabilityConfig(**options)))
    return dict(schema="controlled_spvt_global_box_v1", task=task.name,
                implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                metadata=task.metadata, config=asdict(config), total_timeout_seconds=total_timeout_seconds,
                contexts=rows, all_contexts_certified=all(r["status"] == "certified" for r in rows),
                total_wall_seconds=time.perf_counter() - start,
                scope="Separate universal initial-box check; no reuse of this result as independently computed point timings.")


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def self_test():
    """Small semantic checks: propagation, safe split, and genuine violation."""
    class LinearTask:
        name, state_dim, error_dim, horizon, dt = "linear_semantic_check", 2, 2, 40, 0.1
        domain_lo, domain_hi = np.full(2, -1.0), np.full(2, 1.0)
        error_lo, error_hi = np.full(2, -0.05), np.full(2, 0.05)
        noise_lo, noise_hi = np.empty(0), np.empty(0)
        contexts, metadata = (None,), {}
        A = np.array([[0.4, 0.3], [-0.3, 0.4]])
        def classify_safe(self, lo, hi, t=0):
            return "safe" if np.all(lo >= -1) and np.all(hi <= 1) else ("unsafe" if np.array_equal(lo, hi) else "unknown")
        def affine_matrices(self, lo, hi, context=None, t=0):
            return self.A, np.eye(2), np.empty((2, 0)), np.zeros(2)
        def step(self, x, error=None, noise=None, context=None, t=0):
            return self.A @ x + error
        def interval_step(self, lo, hi, error_lo=None, error_hi=None, **kwargs):
            center, radius = (lo + hi) / 2, (hi - lo) / 2
            return self.A @ center - np.abs(self.A) @ radius + error_lo, self.A @ center + np.abs(self.A) @ radius + error_hi
    config = ReachabilityConfig(timeout_seconds=10, max_nodes=10000, falsification_paths=8)
    task = LinearTask()
    z = verify_initial(task, np.array([0.5, -0.5]), config=config)
    assert z["status"] == "certified" and z["max_generators"] == 80, z
    intervals = verify_initial(task, np.array([0.5, -0.5]), config=ReachabilityConfig(mode="interval"))
    assert intervals["status"] == "certified", intervals
    task.A = np.eye(2) * 1.2
    unstable = verify_initial(task, np.array([0.5, 0.5]), config=config)
    assert unstable["status"] == "counterexample", unstable

    class ProductTask:
        name, state_dim, error_dim, horizon, dt = "partition_semantic_check", 1, 1, 1, 1
        domain_lo, domain_hi = np.array([-0.1]), np.array([0.3])
        error_lo, error_hi = np.array([0.0]), np.array([1.0])
        noise_lo, noise_hi = np.empty(0), np.empty(0)
        def classify_safe(self, lo, hi, t=0):
            return "safe" if lo[0] >= -0.1 and hi[0] <= 0.3 else ("unsafe" if np.array_equal(lo, hi) else "unknown")
        def step(self, x, error=None, **kwargs):
            return error * (1 - error)
        def interval_step(self, lo, hi, error_lo=None, error_hi=None, **kwargs):
            return error_lo * (1 - error_hi), error_hi * (1 - error_lo)
    shallow = verify_initial(ProductTask(), [0.0], config=ReachabilityConfig(max_depth=0, falsification_paths=4))
    refined = verify_initial(ProductTask(), [0.0], config=config)
    assert shallow["status"].startswith("unknown"), shallow
    assert refined["status"] == "certified" and refined["splits"] > 0, refined
    assert abs(hoeffding_lower_bound(2000, 2000, .05) - .9696319269045848) < 1e-14
    return dict(status="passed", linear_zonotope=z, linear_interval=intervals,
                unstable_counterexample=unstable, unsplit=shallow, adaptive_partition=refined)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["aebs", "lane", "braking", "hover", "crane"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--variant")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--horizons", type=int, nargs="+")
    parser.add_argument("--mode", choices=["auto", "zonotope", "interval"], default="auto")
    parser.add_argument("--timeout", type=float, default=5)
    parser.add_argument("--total-timeout", type=float, default=300)
    parser.add_argument("--context-mode", choices=["sampled_iid", "per_context"], default="sampled_iid")
    parser.add_argument("--global-box", action="store_true")
    parser.add_argument("--max-nodes", type=int, default=20000)
    parser.add_argument("--max-frontier", type=int, default=256)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--falsification-paths", type=int, default=24)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        report = self_test()
    else:
        if not args.task:
            parser.error("--task is required unless --self-test is used")
        from Aebs.paper_comparison.tasks import load_task
        task = load_task(args.task, root=args.root, variant=args.variant)
        config = ReachabilityConfig(mode=args.mode, timeout_seconds=args.timeout, max_nodes=args.max_nodes,
                                    max_frontier=args.max_frontier, max_depth=args.max_depth,
                                    falsification_paths=args.falsification_paths, seed=args.seed)
        if args.global_box:
            report = evaluate_global_task(task, config=config, total_timeout_seconds=args.total_timeout,
                                          horizon=max(args.horizons) if args.horizons else None)
        else:
            report = evaluate_task(task, n=args.n, seed=args.seed, horizons=args.horizons, config=config,
                                   total_timeout_seconds=args.total_timeout, context_mode=args.context_mode)
    content = json.dumps(report, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False)
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"Refusing to replace existing result: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n")
        print(json.dumps(dict(output=str(args.output), status=report.get("status", "completed"))))
    else:
        print(content)


if __name__ == "__main__":
    main()
