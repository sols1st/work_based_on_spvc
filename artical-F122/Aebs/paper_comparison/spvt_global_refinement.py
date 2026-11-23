"""Strengthen finite-horizon reachability by partitioning initial states.

Each child is propagated again from time zero. This can remove wrapping from
earlier steps, unlike subdividing a late reachable box that already contains
spurious states. The proof uses no barrier or cached current-certificate result.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import heapq
from pathlib import Path
import time

import numpy as np

from .spvt_reachability import ReachabilityConfig, verify_initial_box


def verify_partitioned_initial_box(task, context=None, horizon=None, timeout_seconds=300.0,
                                   max_initial_depth=28, max_boxes=100000,
                                   local_partition_depth=0):
    horizon = task.horizon if horizon is None else int(horizon)
    root_lo, root_hi = np.asarray(task.initial_lo), np.asarray(task.initial_hi)
    widths = root_hi - root_lo
    active = widths > 0
    start = time.perf_counter()
    queue = [(-1.0, 0, root_lo.copy(), root_hi.copy(), 0)]
    serial, evaluated, bound_calls, deepest, global_prefix = 0, 0, 0, 0, 0
    certified, unresolved = [], []
    while queue:
        remaining = timeout_seconds - (time.perf_counter() - start)
        if remaining <= 0 or evaluated >= max_boxes:
            reason = "timeout" if remaining <= 0 else "box_budget"
            unresolved.extend(dict(lo=a.tolist(), hi=b.tolist(), depth=d,
                                   geometric_fraction=-priority, reason=reason)
                              for priority, _, a, b, d in queue)
            queue.clear()
            break
        neg_mass, _, lo, hi, depth = heapq.heappop(queue)
        config = ReachabilityConfig(mode="auto", timeout_seconds=remaining,
                                    max_nodes=200000, max_frontier=512,
                                    max_depth=local_partition_depth, falsification_paths=0)
        check = verify_initial_box(task, lo=lo, hi=hi, context=context, horizon=horizon, config=config)
        evaluated += 1
        bound_calls += check.get("bound_evaluations", 0)
        deepest = max(deepest, depth)
        if evaluated == 1:
            global_prefix = check["certified_steps"]
        if check["status"] == "certified":
            certified.append(dict(lo=lo.tolist(), hi=hi.tolist(), depth=depth,
                                  geometric_fraction=-neg_mass, bound_calls=check.get("bound_evaluations", 0)))
            continue
        if depth >= max_initial_depth or not np.any(hi > lo):
            unresolved.append(dict(lo=lo.tolist(), hi=hi.tolist(), depth=depth,
                                   geometric_fraction=-neg_mass, reason="initial_depth_or_point_limit",
                                   last_status=check["status"], certified_steps=check["certified_steps"]))
            continue
        normalized = np.divide(hi - lo, widths, out=np.zeros_like(widths), where=active)
        axis = int(np.argmax(normalized))
        midpoint = lo[axis] + (hi[axis] - lo[axis]) / 2
        if midpoint == lo[axis] or midpoint == hi[axis]:
            unresolved.append(dict(lo=lo.tolist(), hi=hi.tolist(), depth=depth,
                                   geometric_fraction=-neg_mass, reason="floating_point_split_limit"))
            continue
        first_hi, second_lo = hi.copy(), lo.copy()
        first_hi[axis] = midpoint
        second_lo[axis] = midpoint
        for a, b in ((lo, first_hi), (second_lo, hi)):
            serial += 1
            fraction = float(np.prod((b[active] - a[active]) / widths[active]))
            heapq.heappush(queue, (-fraction, serial, a, b, depth + 1))
    return dict(
        schema="controlled_spvt_initial_partition_v1", task=task.name, context=context,
        status="certified" if not unresolved else "unknown_partition_incomplete",
        horizon=horizon, certified_steps=horizon if not unresolved else global_prefix,
        initial_lo=root_lo.tolist(), initial_hi=root_hi.tolist(),
        method="initial-domain branch and bound with complete reachability recomputation",
        evaluated_initial_boxes=evaluated, bound_evaluations=bound_calls,
        max_initial_depth_reached=deepest, certified_leaves=len(certified), unresolved_leaves=len(unresolved),
        certified_geometric_volume_fraction=sum(row["geometric_fraction"] for row in certified),
        volume_scope="Geometric fraction of initial bounding box; not probability under a nonuniform initial law.",
        wall_seconds=time.perf_counter() - start, timeout_seconds=timeout_seconds,
        config=dict(max_initial_depth=max_initial_depth, max_boxes=max_boxes,
                    local_partition_depth=local_partition_depth),
        certificate_cells=certified, unresolved_cells=unresolved,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        metadata=task.metadata,
        scope="Controlled finite-horizon numerical reachability; no current barrier used; no directed rounding claimed.",
    )


def evaluate_strengthened_global(task, total_timeout_seconds=300.0):
    start, results = time.perf_counter(), []
    for index, context in enumerate(task.contexts):
        remaining = total_timeout_seconds - (time.perf_counter() - start)
        allowance = max(1e-9, remaining / (len(task.contexts) - index))
        results.append(verify_partitioned_initial_box(task, context=context, timeout_seconds=allowance))
    return dict(schema="controlled_spvt_strengthened_global_v1", task=task.name,
                contexts=results, all_contexts_certified=all(r["status"] == "certified" for r in results),
                total_wall_seconds=time.perf_counter() - start, total_timeout_seconds=total_timeout_seconds,
                additional_experiment=True,
                note="Separate strengthening run, not included in the original 300-second baseline cost; do not sum advantages without reporting this cost.")


def self_test():
    class DependencyTask:
        name, state_dim, error_dim, horizon, dt = "initial_partition_semantic_check", 1, 0, 1, 1
        initial_lo, initial_hi = np.array([0.]), np.array([1.])
        domain_lo, domain_hi = np.array([-1.]), np.array([1.])
        error_lo, error_hi = np.empty(0), np.empty(0)
        noise_lo, noise_hi = np.empty(0), np.empty(0)
        contexts, metadata = (None,), {}
        def classify_safe(self, lo, hi, t=0):
            return "safe" if t == 0 or (lo[0] >= -.05 and hi[0] <= .05) else "unknown"
        def interval_step(self, lo, hi, **kwargs):
            return lo - hi, hi - lo  # A sound but dependency-losing x-x bound.
    result = verify_partitioned_initial_box(DependencyTask(), timeout_seconds=10)
    assert result["status"] == "certified", result
    assert result["certified_leaves"] == 32 and result["certified_geometric_volume_fraction"] == 1
    return dict(status="passed", evaluated_initial_boxes=result["evaluated_initial_boxes"],
                certified_leaves=result["certified_leaves"])
