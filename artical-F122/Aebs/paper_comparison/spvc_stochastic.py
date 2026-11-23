"""Controlled SPVC-inspired expectation checks on the shared task interface.

This is NOT a reproduction of native cGAN training.  The primary comparison
keeps the controller, barrier and uncertainty support fixed, and replaces a
universal successor condition with a uniform-noise expectation condition.
Its induced transition law may depend on state, unlike Assumption 1 of the
provided SafePVC manuscript.  A second, explicitly model-dependent diagnostic
fits one state-independent additive uniform box on development transitions.

Expectations are enclosed by integrating barrier interval bounds over a
partition of the complete noise support.  Monte Carlo is never used to accept
a certificate.  State boxes are subdivided over the full safe sublevel set,
not merely a thin shell.  Float64 interval operations are numerical evidence;
they are not advertised as outward-rounded formal proofs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import heapq
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass
class Budget:
    max_state_boxes: int = 4000
    max_seconds: float = 90.0
    noise_cells: int = 32
    max_depth: int = 22
    witness_states: int = 64
    witness_seed: int = 181123
    strict_epsilon: float = 1e-6
    region_boxes: int = 256
    witness_relative_tolerance: float = 1e-9


def _array(x):
    return np.asarray(x, dtype=np.float64)


def _metadata(task):
    return getattr(task, "metadata", {})


def _scalar(x):
    return float(np.asarray(x).reshape(-1)[0])


def _barrier_bounds(task, lo, hi, t):
    lower, upper = task.barrier_bounds(_array(lo), _array(hi), t=t)
    return _array(lower).reshape(-1), _array(upper).reshape(-1)


def uniform_partition(lo, hi, requested_cells):
    """Equal-mass product partition; zero-width coordinates are Dirac masses.

    The actual count is the largest power of two not exceeding the budget.
    Coordinates are split in cyclic order rather than compared in mixed units.
    """
    lo, hi = _array(lo), _array(hi)
    if lo.shape != hi.shape or np.any(hi < lo) or not np.isfinite([lo, hi]).all():
        raise ValueError("finite ordered noise bounds required")
    counts = np.ones(len(lo), dtype=int)
    active = np.flatnonzero(hi > lo)
    i = 0
    while len(active) and np.prod(counts) * 2 <= max(1, requested_cells):
        counts[active[i % len(active)]] *= 2
        i += 1
    coordinates = []
    for a, b, n in zip(lo, hi, counts):
        edges = np.linspace(a, b, n + 1)
        coordinates.append(list(zip(edges[:-1], edges[1:])))
    pairs = list(itertools.product(*coordinates))
    if not pairs:
        return np.zeros((1, 0)), np.zeros((1, 0)), np.ones(1)
    array = np.asarray(pairs, dtype=np.float64)
    return array[:, :, 0], array[:, :, 1], np.full(len(pairs), 1 / len(pairs))


def fit_additive_mu(task, *, seed=181121, trajectories=64, steps=12):
    """Register an assumed pooled uniform law using development samples.

    Samples use uniform draws within the SAME observation and plant-noise
    support as the robust task.  Min/max are an engineering fit, not CP or a
    proof of real-world support.  The fitted law deliberately forgets the
    state correlation; all downstream claims are conditional on that law.
    """
    rng = np.random.default_rng(seed)
    deltas, states = [], []
    contexts = list(task.contexts) or [None]
    for ci, context in enumerate(contexts):
        state = _array(task.sample_initial(trajectories, seed + ci))
        for t in range(min(int(task.horizon), steps)):
            err = rng.uniform(task.error_lo, task.error_hi, (len(state), task.error_dim))
            no = rng.uniform(task.noise_lo, task.noise_hi, (len(state), len(task.noise_lo)))
            base = _array(task.nominal_step(state, context=context, t=t))
            nxt = _array(task.step(state, error=err, noise=no, context=context, t=t))
            deltas.append(nxt - base)
            states.append(state)
            # State sampling follows the perturbed closed loop, not independent
            # draws from an unrelated state box.
            state = nxt
    delta = np.concatenate(deltas, axis=0)
    seen = np.concatenate(states, axis=0)
    lo, hi = delta.min(axis=0), delta.max(axis=0)
    return {
        "kind": "assumed_state_independent_additive_uniform",
        "development_seed": seed,
        "trajectories_per_context": trajectories,
        "steps_per_trajectory": min(int(task.horizon), steps),
        "transition_count": len(delta),
        "lower": lo.tolist(), "upper": hi.tolist(),
        "delta_mean": delta.mean(axis=0).tolist(),
        "delta_std": delta.std(axis=0).tolist(),
        "state_min": seen.min(axis=0).tolist(), "state_max": seen.max(axis=0).tolist(),
        "observation_support_lower": _array(task.error_lo).tolist(),
        "observation_support_upper": _array(task.error_hi).tolist(),
        "plant_noise_lower": _array(task.noise_lo).tolist(),
        "plant_noise_upper": _array(task.noise_hi).tolist(),
        "coverage_guarantee": None,
        "real_transition_distribution_validated": False,
        "note": "Pooled sample extrema define an assumed law, not a certified residual envelope. No real-camera probability is inferred.",
    }


class ExpectationOracle:
    def __init__(self, task, noise_cells=32, model="observation_uniform", mu=None):
        self.task, self.model, self.calls = task, model, 0
        if model == "observation_uniform":
            low = np.concatenate([task.error_lo, task.noise_lo])
            high = np.concatenate([task.error_hi, task.noise_hi])
        elif model == "additive_uniform":
            if mu is None:
                raise ValueError("additive mode requires a registered mu")
            low, high = mu["lower"], mu["upper"]
        else:
            raise ValueError(f"unknown model {model}")
        self.low, self.high, self.mass = uniform_partition(low, high, noise_cells)

    def evaluate(self, lo, hi, *, context=None, t=0):
        task = self.task
        if hasattr(task, "terminal_status"):
            terminal = task.terminal_status(lo, hi)
        elif np.array_equal(lo, hi) and bool(np.asarray(getattr(task, "is_terminal", lambda _: False)(lo)).all()):
            terminal = "terminal"
        else:
            terminal = "mixed" if hasattr(task, "_terminal") else "active"
        if terminal == "terminal":
            lower, upper = _barrier_bounds(task, lo, hi, t + 1)
            self.calls += 1
            return {"lower": float(lower[0]), "upper": float(upper[0]),
                    "domain_enclosed": bool(np.all(_array(lo) >= task.domain_lo) and np.all(_array(hi) <= task.domain_hi)),
                    "successor_lo": _array(lo), "successor_hi": _array(hi),
                    "absorbing_identity": not _metadata(task).get("barrier_time_varying", False)}
        next_lows, next_highs = [], []
        if self.model == "additive_uniform":
            base_lo, base_hi = task.interval_step(
                lo, hi, error_lo=np.zeros(task.error_dim), error_hi=np.zeros(task.error_dim),
                noise_lo=np.zeros(len(task.noise_lo)), noise_hi=np.zeros(len(task.noise_lo)),
                context=context, t=t,
            )
            next_lows = _array(base_lo)[None, :] + self.low
            next_highs = _array(base_hi)[None, :] + self.high
            if terminal == "mixed":
                # The stopped process applies the same active-state law but
                # freezes terminal states before adding any disturbance.
                next_lows = np.minimum(next_lows, _array(lo)[None, :])
                next_highs = np.maximum(next_highs, _array(hi)[None, :])
        else:
            d = task.error_dim
            for low, high in zip(self.low, self.high):
                a, b = task.interval_step(
                    lo, hi, error_lo=low[:d], error_hi=high[:d],
                    noise_lo=low[d:], noise_hi=high[d:], context=context, t=t,
                )
                next_lows.append(a)
                next_highs.append(b)
            next_lows, next_highs = _array(next_lows), _array(next_highs)
        # Evaluate every noise cell, including successors outside the working
        # domain.  Such an integral is diagnostic only and cannot accept a proof.
        lower, upper = _barrier_bounds(task, next_lows, next_highs, t + 1)
        self.calls += len(self.mass)
        closed = bool(np.all(next_lows >= _array(task.domain_lo)) and
                      np.all(next_highs <= _array(task.domain_hi)))
        return {
            "lower": math.fsum(float(p) * float(v) for p, v in zip(self.mass, lower)),
            "upper": math.fsum(float(p) * float(v) for p, v in zip(self.mass, upper)),
            "domain_enclosed": closed,
            "successor_lo": next_lows.min(axis=0),
            "successor_hi": next_highs.max(axis=0),
        }


def stochastic_inheritance(task, model="observation_uniform"):
    """Preserve the strong stochastic baseline implied by a robust proof.

    An incomplete independent integration check cannot overturn this logical
    implication.  Its proof quality and domain remain exactly those of the
    source robust verification; this function does not strengthen arithmetic.
    """
    explanation = ("If B(next)<=B(now) for every disturbance in the same support, "
                   "then E[B(next)|state]<=B(state) for every probability law on "
                   "that support. No uniformity or independent-coordinate assumption is needed.")
    if model != "observation_uniform":
        return {"status": "not_inherited_for_different_transition_support",
                "reason": "A pooled additive box need not equal the state-conditioned observation-error image.",
                "logical_implication": explanation}
    try:
        source = task.current_baseline()
    except Exception as exc:
        return {"status": "source_verification_unavailable", "error": repr(exc),
                "logical_implication": explanation}
    if source.get("passed", False):
        return {"status": "nonincrease_inherited_from_matching_robust_certificate",
                "strict_epsilon_inherited": False,
                "proof_scope": "Same modeled support, candidate and domain as the source robust result; does not certify real-camera probabilities.",
                "source_robust_result": source, "logical_implication": explanation}
    return {"status": "source_robust_proof_not_rerun_or_not_passed",
            "source_robust_result": source, "logical_implication": explanation}


def _split(lo, hi, scale):
    axis = int(np.argmax((hi - lo) / np.maximum(scale, 1e-15)))
    middle = (lo[axis] + hi[axis]) / 2
    left_hi, right_lo = hi.copy(), lo.copy()
    left_hi[axis] = middle
    right_lo[axis] = middle
    return [(lo.copy(), left_hi), (right_lo, hi.copy())]


def _initial_upper(task, max_boxes):
    queue = [(_array(task.initial_lo), _array(task.initial_hi), 0)]
    target = min(1.0, float(_metadata(task).get("unsafe_barrier_threshold", 20.0)))
    resolved, remaining = [], []
    count = 0
    scale = _array(task.initial_hi) - _array(task.initial_lo)
    while queue and count < max_boxes:
        lo, hi, depth = queue.pop()
        _, upper = _barrier_bounds(task, lo, hi, 0)
        count += 1
        if upper[0] <= target or depth >= 12:
            resolved.append(float(upper[0]))
        else:
            queue.extend((a, b, depth + 1) for a, b in _split(lo, hi, scale))
    for lo, hi, _ in queue:
        _, upper = _barrier_bounds(task, lo, hi, 0)
        remaining.append(float(upper[0]))
    value = max(resolved + remaining, default=float("inf"))
    return {"upper": value, "boxes_evaluated": count,
            "all_initial_states_enclosed": True, "target": target,
            "passed": value <= target}


def _point_witness(task, oracle, x, context, t, threshold):
    x = _array(x)
    if task.classify_safe(x, x, t=t) != "safe":
        return None
    b = _scalar(task.barrier(x[None, :], t=t))
    if b >= threshold or b < 0:
        return None
    exp = oracle.evaluate(x, x, context=context, t=t)
    blo, bhi = _barrier_bounds(task, x, x, t)
    lower = exp["lower"] - float(bhi[0])
    upper = exp["upper"] - float(blo[0])
    if exp.get("absorbing_identity", False):
        lower = upper = 0.0
    return {
        "state": x.tolist(), "time": t, "context": context,
        "barrier": b, "expected_barrier_lower": exp["lower"],
        "expected_barrier_upper": exp["upper"],
        "current_barrier_lower": float(blo[0]), "current_barrier_upper": float(bhi[0]),
        "drift_lower": lower, "drift_upper": upper,
        "absorbing_identity": exp.get("absorbing_identity", False),
        "domain_enclosed": exp["domain_enclosed"],
        "successor_lo": exp["successor_lo"].tolist(),
        "successor_hi": exp["successor_hi"].tolist(),
    }


def _significant_positive(witness, budget):
    scale = 1 + abs(witness["barrier"]) + max(abs(witness["expected_barrier_lower"]), abs(witness["expected_barrier_upper"]))
    return witness["drift_lower"] > budget.witness_relative_tolerance * scale


def run_spvc(task, budget=None, *, model="observation_uniform", mu=None):
    """Run fixed-candidate expectation diagnostics and continuous box checks.

    A successful numerical drift check is NOT a complete certificate unless
    nonnegativity, initial/unsafe separation, model closure and the time
    semantics are all independently justified.  Unknown obligations remain
    unknown in the returned record.
    """
    budget = budget or Budget()
    if isinstance(budget, dict):
        budget = Budget(**budget)
    started = time.monotonic()
    meta = _metadata(task)
    inheritance = stochastic_inheritance(task, model)
    threshold = float(meta.get("unsafe_barrier_threshold", 20.0))
    varying = bool(meta.get("barrier_time_varying", False))
    if model == "additive_uniform" and mu is None:
        mu = fit_additive_mu(task)
    oracle = ExpectationOracle(task, budget.noise_cells, model, mu)
    contexts = list(task.contexts) or [None]
    times = range(task.horizon) if varying else [0]
    initial = _initial_upper(task, budget.region_boxes)
    witnesses, strict_only = [], []
    rng = np.random.default_rng(budget.witness_seed)
    x0 = _array(task.sample_initial(max(1, budget.witness_states // 2), budget.witness_seed))
    domain = rng.uniform(task.domain_lo, task.domain_hi,
                         size=(max(1, budget.witness_states // 2), task.state_dim))
    points = np.concatenate([x0, domain,
                             ((_array(task.domain_lo) + _array(task.domain_hi)) / 2)[None, :]])
    # These points can disprove a candidate using expectation LOWER bounds.
    # They can never establish universal acceptance.
    for ci, context in enumerate(contexts):
        for ti, t in enumerate(times):
            for point in points:
                if time.monotonic() - started >= budget.max_seconds * .25:
                    break
                w = _point_witness(task, oracle, point, context, t, threshold)
                if w and w["domain_enclosed"]:
                    if _significant_positive(w, budget):
                        witnesses.append(w)
                    elif w["drift_lower"] + budget.strict_epsilon > 0:
                        strict_only.append(w)
            if witnesses or time.monotonic() - started >= budget.max_seconds * .25:
                break
        if witnesses or time.monotonic() - started >= budget.max_seconds * .25:
            break

    serial = itertools.count()
    queue = []
    dlo, dhi = _array(task.domain_lo), _array(task.domain_hi)
    scale = dhi - dlo
    for ci, context in enumerate(contexts):
        for t in times:
            heapq.heappush(queue, (0, next(serial), dlo.copy(), dhi.copy(), ci, t, 0))
    counts = {"processed": 0, "drift_passed": 0, "strict_drift_passed": 0,
              "outside_relevant_sublevel": 0, "unsafe_only": 0,
              "unresolved": 0, "domain_not_enclosed": 0}
    unresolved = []
    max_drift_upper = -float("inf")
    # Once a point with a positive lower bound exists, this fixed candidate is
    # refuted for the registered modeled law; spending all boxes is unnecessary.
    while queue and not witnesses:
        if counts["processed"] >= budget.max_state_boxes or time.monotonic() - started >= budget.max_seconds:
            break
        _, _, lo, hi, ci, t, depth = heapq.heappop(queue)
        counts["processed"] += 1
        blo, bhi = _barrier_bounds(task, lo, hi, t)
        if blo[0] > threshold:
            counts["outside_relevant_sublevel"] += 1
            continue
        safety = task.classify_safe(lo, hi, t=t)
        if safety == "unsafe":
            counts["unsafe_only"] += 1
            continue
        exp = oracle.evaluate(lo, hi, context=contexts[ci], t=t)
        delta = exp["upper"] - float(blo[0])
        if exp.get("absorbing_identity", False):
            # Identity is a symbolic equality for every state in this box,
            # so dependency loss B(hi)-B(lo) must not turn it into a failure.
            delta = 0.0
        passed = exp["domain_enclosed"] and delta <= 0
        if passed:
            counts["drift_passed"] += 1
            counts["strict_drift_passed"] += int(delta <= -budget.strict_epsilon)
            max_drift_upper = max(max_drift_upper, delta)
            continue
        point = (lo + hi) / 2
        w = _point_witness(task, oracle, point, contexts[ci], t, threshold)
        if w and w["domain_enclosed"]:
            if _significant_positive(w, budget):
                witnesses.append(w)
                break
            if w["drift_lower"] + budget.strict_epsilon > 0 and len(strict_only) < 5:
                strict_only.append(w)
        if depth >= budget.max_depth or np.max((hi - lo) / np.maximum(scale, 1e-15)) < 1e-8:
            counts["unresolved"] += 1
            counts["domain_not_enclosed"] += int(not exp["domain_enclosed"])
            if len(unresolved) < 8:
                unresolved.append({"lo": lo.tolist(), "hi": hi.tolist(), "time": t,
                                   "context": contexts[ci], "drift_upper": delta,
                                   "domain_enclosed": exp["domain_enclosed"]})
            continue
        for a, b in _split(lo, hi, scale):
            heapq.heappush(queue, (depth + 1, next(serial), a, b, ci, t, depth + 1))

    complete = not queue and counts["unresolved"] == 0 and not witnesses
    if witnesses:
        status = "numerical_positive_expected_drift_witness"
    elif complete:
        status = "continuous_domain_numerical_nonincrease_pass"
    else:
        status = "inconclusive_within_budget"
    if witnesses or strict_only:
        strict_status = "fixed_candidate_does_not_meet_strict_epsilon"
    elif complete and counts["strict_drift_passed"] == counts["drift_passed"]:
        strict_status = "continuous_domain_numerical_strict_drift_pass"
    else:
        strict_status = "inconclusive_or_zero_drift_cells"
    separation = meta.get("unsafe_separation_evidence", meta.get("barrier_separation_evidence"))
    return {
        "task": task.name, "method": "SPVC-inspired controlled expectation backend",
        "comparison_scope": "same_frozen_controller_and_barrier_candidate; not native cGAN retraining",
        "stochastic_model": model,
        "model_description": (
            "Uniform independent observation and plant errors on the registered robust support; induced transition law depends on state."
            if model == "observation_uniform" else
            "Assumed pooled state-independent additive uniform transition law, fitted once on development trajectories."
        ),
        "paper_assumption_1_preserved": False,
        "paper_assumption_1_scope": ("state-independent additive law on active states; shared absorbing-stop convention makes the full stopped kernel state-dependent"
                                     if model == "additive_uniform" else "state-conditioned transition law induced by uniform observation and plant errors"),
        "stopping_semantics": "terminal states absorb before any disturbance is applied, for both stochastic models",
        "real_visual_probability_claim": None,
        "mu": mu,
        "nonincrease_status": status, "strict_status": strict_status,
        "strong_stochastic_baseline": inheritance,
        "interpretation_of_inconclusive": "An incomplete standalone interval integration is not failure of a stochastic method; a matching robust certificate already implies its nonincrease expectation condition.",
        "strict_epsilon": budget.strict_epsilon,
        "strict_comparison_warning": "Zero plateaus and absorbing terminal states can make strict epsilon impossible for this transferred candidate; this is not evidence that SPVC cannot synthesize a different certificate.",
        "expectation_bound_method": "equal-mass complete-support interval integration",
        "noise_partition_cells": len(oracle.mass),
        "interval_barrier_evaluations": oracle.calls,
        "initial_bound": initial, "unsafe_threshold": threshold,
        "nonnegative_by_construction": meta.get("barrier_nonnegative_by_construction", False),
        "unsafe_separation_evidence": separation,
        "complete_certificate_claim": False,
        "certificate_obligations": "Numerical drift results alone do not establish unsafe separation, reliable arithmetic, real visual distribution validity, or all closure/terminal assumptions.",
        "barrier_time_varying": varying,
        "theoretical_time_scope": f"finite horizon {task.horizon}" if varying else "stationary candidate; any infinite-horizon claim additionally requires all global SBC obligations",
        "counts": {**counts, "queued_unresolved": len(queue)},
        "largest_accepted_cell_drift_upper": max_drift_upper if np.isfinite(max_drift_upper) else None,
        "positive_drift_witnesses": sorted(witnesses, key=lambda w: w["drift_lower"], reverse=True)[:5],
        "strict_only_witnesses": strict_only[:5], "unresolved_examples": unresolved,
        "arithmetic": "float64 interval operations; no directed outward rounding",
        "witness_significance_guard": "drift lower must exceed relative tolerance times (1+|B|+|E upper/lower|); smaller positive drift is never used as a counterexample. Acceptance is not relaxed by this tolerance.",
        "outward_rounded": False, "elapsed_seconds": time.monotonic() - started,
        "budget": asdict(budget), "task_metadata": meta,
        "source_binding": {"path": str(Path(__file__).resolve()),
                           "sha256": SOURCE_SHA256},
    }


run_task = run_spvc


def self_test():
    class Quadratic:
        error_dim = 1
        error_lo, error_hi = np.array([-.2]), np.array([.2])
        noise_lo, noise_hi = np.empty(0), np.empty(0)
        domain_lo, domain_hi = np.array([-3.]), np.array([3.])

        def interval_step(self, lo, hi, error_lo, error_hi, **kwargs):
            return .5 * _array(lo) + error_lo, .5 * _array(hi) + error_hi

        def barrier_bounds(self, lo, hi, **kwargs):
            lo, hi = _array(lo), _array(hi)
            lower = np.where((lo <= 0) & (hi >= 0), 0, np.minimum(lo**2, hi**2))
            return lower.sum(axis=-1), np.maximum(lo**2, hi**2).sum(axis=-1)

    task = Quadratic()
    exact = .25 + .2**2 / 3
    widths = []
    for cells in [1, 4, 64]:
        oracle = ExpectationOracle(task, cells)
        result = oracle.evaluate(np.array([1.]), np.array([1.]))
        assert result["lower"] <= exact <= result["upper"], result
        widths.append(result["upper"] - result["lower"])
    assert widths[2] < widths[1] < widths[0]
    # Nonzero process noise causes positive expected x^2 drift at the origin.
    zero = ExpectationOracle(task, 64).evaluate(np.zeros(1), np.zeros(1))
    assert zero["lower"] > 0
    low, high, weights = uniform_partition([0, -1], [0, 1], 8)
    assert len(weights) == 8 and np.all(low[:, 0] == high[:, 0])
    assert weights.sum() == 1
    return {"passed": True, "analytic_expectation": exact, "interval_widths": widths,
            "positive_drift_lower_at_origin": zero["lower"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="aebs")
    parser.add_argument("--variant")
    parser.add_argument("--model", choices=["observation_uniform", "additive_uniform"], default="observation_uniform")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mu-json", type=Path)
    parser.add_argument("--max-seconds", type=float, default=90)
    parser.add_argument("--max-boxes", type=int, default=4000)
    parser.add_argument("--noise-cells", type=int, default=32)
    parser.add_argument("--witness-states", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        result = self_test()
    else:
        from .tasks import load_task
        task = load_task(args.task, variant=args.variant)
        mu = json.loads(args.mu_json.read_text()) if args.mu_json else None
        result = run_spvc(task, Budget(max_seconds=args.max_seconds, max_state_boxes=args.max_boxes,
                                      noise_cells=args.noise_cells, witness_states=args.witness_states),
                          model=args.model, mu=mu)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
