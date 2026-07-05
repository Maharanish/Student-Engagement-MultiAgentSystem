"""Pure-function Bayesian belief updates over HIDDEN_STATES."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from mas_engagement.config import (
    BELIEF_DECAY_RATE,
    DEFAULT_PRIOR,
    EWMA_GAMMA,
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


def _evidence_over_states(softmax: List[float]) -> np.ndarray:
    """Map a detection-class softmax to a P(O_t|s) distribution over HIDDEN_STATES.

    weighted[s] = Σ_c softmax[c]·P(c|s) via LIKELIHOOD_DETECTION, then normalized
    so it is a proper distribution over the hidden states (the detection classes
    and the hidden states are *not* the same axis — see config.LIKELIHOOD_DETECTION).
    A uniform softmax maps to a uniform P(O_t|s) because each likelihood row sums to 1.
    """
    sm = np.asarray(softmax, dtype=np.float64)
    if sm.shape[0] != LIKELIHOOD_DETECTION.shape[1]:
        raise ValueError(
            f"softmax length {sm.shape[0]} != likelihood cols {LIKELIHOOD_DETECTION.shape[1]}"
        )
    weighted = LIKELIHOOD_DETECTION @ sm  # shape (|S|,), P(obs | s) marginalised
    total = weighted.sum()
    if total <= 0.0 or not np.isfinite(total):
        return np.full(len(HIDDEN_STATES), 1.0 / len(HIDDEN_STATES))
    return weighted / total


def update_with_ewma(
    belief: Dict[str, float],
    softmax: List[float],
    gamma: float = EWMA_GAMMA,
) -> Dict[str, float]:
    """EWMA belief update: B_t(s) = (1-γ)·P(O_t|s) + γ·B_{t-1}(s).

    Combines forgetting and evidence assimilation in a single step (replaces the
    separate decay_toward_prior + update_with_detection pipeline). γ (gamma) is
    the forgetting factor in [0, 1]: higher γ → slower updates (more weight on
    the belief history B_{t-1}).

    P(O_t|s) is the detection evidence mapped onto the hidden states via
    LIKELIHOOD_DETECTION (see _evidence_over_states). Because (1-γ)+γ = 1 and
    both P(O_t|s) and B_{t-1} are distributions summing to 1, the output always
    sums to 1 — the belief can never saturate to a one-hot state.

    No-face case: pass a uniform softmax (e.g. [0.25, 0.25, 0.25, 0.25]); this
    maps to a uniform P(O_t|s), so repeated no-face evidence decays the belief
    toward the uniform distribution (the role formerly played by DEFAULT_PRIOR).
    """
    g = float(gamma)
    p_obs = _evidence_over_states(softmax)
    prior = np.array([belief[s] for s in HIDDEN_STATES])
    post = (1.0 - g) * p_obs + g * prior
    result = normalize({s: float(post[i]) for i, s in enumerate(HIDDEN_STATES)})
    return result


def update_with_detection(belief: Dict[str, float], softmax: List[float]) -> Dict[str, float]:
    """DEPRECATED — superseded by update_with_ewma; no longer called by the
    orchestrator detection path. Retained for reference / backward compatibility.

    Soft observation Bayes update: posterior[s] ∝ prior[s] · Σ_c softmax[c]·P(c|s)."""
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


def decay_toward_prior(belief: Dict[str, float], dt_seconds: float) -> Dict[str, float]:
    """DEPRECATED — forgetting is now folded into update_with_ewma; no longer
    called by the orchestrator. Retained for reference / backward compatibility.

    Linear interpolation toward DEFAULT_PRIOR: decayed = (1-λdt)·b + λdt·prior."""
    alpha = max(0.0, min(1.0, BELIEF_DECAY_RATE * float(dt_seconds)))
    out = {
        s: (1.0 - alpha) * belief[s] + alpha * DEFAULT_PRIOR[s]
        for s in HIDDEN_STATES
    }
    return normalize(out)
