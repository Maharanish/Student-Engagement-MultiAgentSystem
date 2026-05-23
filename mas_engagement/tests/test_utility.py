import math
import random

import numpy as np
import pytest

from mas_engagement.config import (
    ACTIONS,
    HIDDEN_STATES,
    MIN_GAP_SEC,
    UTILITY_MATRIX,
    WARMUP_SEC,
)
from mas_engagement.reasoning import utility as U


def _belief(values):
    return {s: float(v) for s, v in zip(HIDDEN_STATES, values)}


def _snapshot(
    *,
    interventions=None,
    intervention_count=0,
    silent_mode=False,
    session_start=0.0,
):
    return {
        "interventions": list(interventions or []),
        "intervention_count": intervention_count,
        "silent_mode": silent_mode,
        "session_start": session_start,
    }


def _post_warmup_now(session_start=0.0):
    return session_start + WARMUP_SEC + 1.0


def test_choose_action_returns_do_nothing_when_belief_concentrated_on_engaged():
    # belief = 1.0 on engaged
    b = {s: 0.0 for s in HIDDEN_STATES}
    b["engaged"] = 1.0
    snap = _snapshot(session_start=0.0)
    now = _post_warmup_now()
    action, eus = U.choose_action(b, snap, now)
    assert action == "do_nothing"
    assert all(eus["do_nothing"] >= eus[a] for a in ACTIONS)


def test_all_intervention_eus_negative_inf_during_warmup():
    b = {s: 0.25 for s in HIDDEN_STATES}
    snap = _snapshot(session_start=1000.0)
    now = 1000.0 + WARMUP_SEC * 0.5  # still warming up
    _, eus = U.choose_action(b, snap, now)
    for a in ACTIONS:
        if a == "do_nothing":
            assert math.isfinite(eus[a])
        else:
            assert eus[a] == float("-inf")


def test_all_intervention_eus_negative_inf_within_cooldown():
    b = {s: 0.25 for s in HIDDEN_STATES}
    now = _post_warmup_now()
    # Last intervention ~1s ago — well inside MIN_GAP_SEC.
    snap = _snapshot(
        session_start=0.0,
        intervention_count=1,
        interventions=[{"ts": now - 1.0, "tier": "2", "msg_id": "x", "response": None}],
    )
    _, eus = U.choose_action(b, snap, now)
    for a in ACTIONS:
        if a == "do_nothing":
            assert math.isfinite(eus[a])
        else:
            assert eus[a] == float("-inf"), f"action {a} should be blocked, got {eus[a]}"

    # After MIN_GAP_SEC elapses, at least some intervention EU is finite again.
    later = now + MIN_GAP_SEC + 1.0
    _, eus_after = U.choose_action(b, snap, later)
    assert any(math.isfinite(eus_after[a]) for a in ACTIONS if a != "do_nothing")


def test_fatigue_cost_monotone_non_decreasing():
    prev = U.fatigue_cost(0)
    for n in range(1, 50):
        cur = U.fatigue_cost(n)
        assert cur >= prev - 1e-12, f"fatigue not monotone at n={n}: {prev} -> {cur}"
        prev = cur


def test_utility_perturbation_decision_stability():
    """Across 20 random ±20% perturbations of UTILITY_MATRIX, argmax should
    flip on fewer than 30% of test scenarios. Uses 5 belief scenarios × 20 trials."""
    rng = random.Random(20260515)
    np_rng = np.random.default_rng(20260515)

    scenarios = [
        _belief([0.10, 0.20, 0.60, 0.10]),  # default-like
        _belief([0.50, 0.30, 0.10, 0.10]),  # disengaged-leaning
        _belief([0.10, 0.10, 0.10, 0.70]),  # frustrated
        _belief([0.25, 0.50, 0.15, 0.10]),  # drifting
        _belief([0.05, 0.10, 0.80, 0.05]),  # engaged
    ]
    snap = _snapshot(session_start=0.0)
    now = _post_warmup_now()

    baseline = [U.choose_action(b, snap, now)[0] for b in scenarios]

    flips = 0
    total = 0
    original = UTILITY_MATRIX.copy()
    try:
        for _ in range(20):
            perturb = np_rng.uniform(0.8, 1.2, size=UTILITY_MATRIX.shape)
            UTILITY_MATRIX[:] = original * perturb
            for i, b in enumerate(scenarios):
                total += 1
                action, _ = U.choose_action(b, snap, now)
                if action != baseline[i]:
                    flips += 1
    finally:
        UTILITY_MATRIX[:] = original

    flip_rate = flips / total
    assert flip_rate < 0.30, f"decision unstable under ±20% utility perturbation: {flip_rate:.2%}"
