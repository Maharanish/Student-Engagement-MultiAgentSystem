"""Pure-function utility scoring and action selection."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from mas_engagement.config import (
    ACTIONS,
    FATIGUE_MIDPOINT,
    FATIGUE_SCALE,
    FATIGUE_STEEPNESS,
    HIDDEN_STATES,
    MIN_GAP_SEC,
    UTILITY_MATRIX,
    WARMUP_SEC,
)

_DO_NOTHING = "do_nothing"


def fatigue_cost(n_interventions: int) -> float:
    """Sigmoid in n: rises from ~0 to FATIGUE_SCALE around FATIGUE_MIDPOINT."""
    x = (float(n_interventions) - FATIGUE_MIDPOINT) / FATIGUE_STEEPNESS
    return FATIGUE_SCALE / (1.0 + math.exp(-x))


def _cooldown_blocked(snapshot: dict, now: float) -> bool:
    if snapshot.get("silent_mode"):
        return True
    # Tier-3 break window: while now < break_until, the system is paused
    # waiting for the student to click the return widget.
    if now < float(snapshot.get("break_until", 0.0)):
        return True
    session_start = float(snapshot.get("session_start", now))
    if (now - session_start) < WARMUP_SEC:
        return True
    interventions = snapshot.get("interventions", []) or []
    if interventions:
        last_ts = max(float(i["ts"]) for i in interventions)
        if (now - last_ts) < MIN_GAP_SEC:
            return True
    return False


def expected_utility(
    action: str, belief: Dict[str, float], snapshot: dict, now: float
) -> float:
    """EU(a) = Σ_s b[s]·U[a,s] − fatigue(n) − cooldown_penalty.

    Returns -inf for intervention actions when blocked (warmup/silent/cooldown).
    do_nothing has zero fatigue and zero cooldown penalty.
    """
    if action not in ACTIONS:
        raise ValueError(f"unknown action: {action!r}")
    a_idx = ACTIONS.index(action)

    b = np.array([belief[s] for s in HIDDEN_STATES], dtype=np.float64)
    base = float(UTILITY_MATRIX[a_idx] @ b)

    if action == _DO_NOTHING:
        return base

    if _cooldown_blocked(snapshot, now):
        return float("-inf")

    n = int(snapshot.get("intervention_count", 0))
    return base - fatigue_cost(n)


def choose_action(
    belief: Dict[str, float], snapshot: dict, now: float
) -> Tuple[str, Dict[str, float]]:
    """Return (argmax_action, full EU dict over ACTIONS)."""
    eus = {a: expected_utility(a, belief, snapshot, now) for a in ACTIONS}
    best = max(eus, key=lambda a: eus[a])
    return best, eus
