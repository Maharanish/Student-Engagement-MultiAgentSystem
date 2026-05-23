"""Pure-function Bayesian belief updates over HIDDEN_STATES."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from mas_engagement.config import (
    BELIEF_DECAY_RATE,
    DEFAULT_PRIOR,
    HIDDEN_STATES,
    LIKELIHOOD_CLICK,
    LIKELIHOOD_DETECTION,
)

_FLOOR = 1e-6


def normalize(belief: Dict[str, float]) -> Dict[str, float]:
    """Floor each entry at 1e-6, then renormalize. All-zero input yields a uniform."""
    vals = np.array([max(float(belief.get(s, 0.0)), _FLOOR) for s in HIDDEN_STATES])
    total = vals.sum()
    if total <= 0.0 or not np.isfinite(total):
        u = 1.0 / len(HIDDEN_STATES)
        return {s: u for s in HIDDEN_STATES}
    vals = vals / total
    return {s: float(vals[i]) for i, s in enumerate(HIDDEN_STATES)}


def update_with_detection(belief: Dict[str, float], softmax: List[float]) -> Dict[str, float]:
    """Soft observation Bayes update: posterior[s] ∝ prior[s] · Σ_c softmax[c]·P(c|s)."""
    sm = np.asarray(softmax, dtype=np.float64)
    if sm.shape[0] != LIKELIHOOD_DETECTION.shape[1]:
        raise ValueError(
            f"softmax length {sm.shape[0]} != likelihood cols {LIKELIHOOD_DETECTION.shape[1]}"
        )
    # P(obs | s) marginalised over the soft observation
    weighted = LIKELIHOOD_DETECTION @ sm  # shape (|S|,)
    prior = np.array([belief[s] for s in HIDDEN_STATES])
    post = prior * weighted
    return normalize({s: float(post[i]) for i, s in enumerate(HIDDEN_STATES)})


def update_with_response(belief: Dict[str, float], clicked: bool) -> Dict[str, float]:
    """Bayes update from a single binary response. P(ignored|s) = 1 - P(clicked|s)."""
    post = {}
    for s in HIDDEN_STATES:
        p_click = LIKELIHOOD_CLICK[s]
        lik = p_click if clicked else (1.0 - p_click)
        post[s] = belief[s] * lik
    return normalize(post)


def decay_toward_uniform(belief: Dict[str, float], dt_seconds: float) -> Dict[str, float]:
    """Linear interpolation toward DEFAULT_PRIOR: decayed = (1-λdt)·b + λdt·prior."""
    alpha = max(0.0, min(1.0, BELIEF_DECAY_RATE * float(dt_seconds)))
    out = {
        s: (1.0 - alpha) * belief[s] + alpha * DEFAULT_PRIOR[s]
        for s in HIDDEN_STATES
    }
    return normalize(out)
