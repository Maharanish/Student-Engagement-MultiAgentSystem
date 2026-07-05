import math

import pytest

from mas_engagement.config import HIDDEN_STATES
from mas_engagement.reasoning import belief as B


def _uniform():
    u = 1.0 / len(HIDDEN_STATES)
    return {s: u for s in HIDDEN_STATES}


def _sums_to_one(b: dict) -> bool:
    return math.isclose(sum(b.values()), 1.0, abs_tol=1e-9)


def _l1(a: dict, b: dict) -> float:
    return sum(abs(a[s] - b[s]) for s in HIDDEN_STATES)


# ── EWMA detection update (replaces decay_toward_prior + update_with_detection) ─

def test_all_updates_preserve_sum_to_one():
    b = _uniform()
    b1 = B.update_with_ewma(b, [0.7, 0.2, 0.07, 0.03])
    b2 = B.update_with_response(b1, clicked=True)
    b3 = B.update_with_response(b2, clicked=False)
    b4 = B.normalize(b3)
    for post in (b1, b2, b3, b4):
        assert _sums_to_one(post)
        assert set(post.keys()) == set(HIDDEN_STATES)


def test_ewma_preserves_sum_to_one():
    """Output is always a valid distribution, for any gamma and any evidence."""
    b = {"actively_dis": 0.1, "drifting": 0.2, "engaged": 0.6, "frustrated": 0.1}
    for gamma in (0.0, 0.1, 0.5, 0.85, 1.0):
        for sm in ([0.95, 0.03, 0.01, 0.01], [0.25, 0.25, 0.25, 0.25],
                   [0.0, 0.0, 0.0, 1.0]):
            out = B.update_with_ewma(b, sm, gamma=gamma)
            assert _sums_to_one(out), f"gamma={gamma} sm={sm}: sum={sum(out.values())}"
            assert set(out.keys()) == set(HIDDEN_STATES)


def test_ewma_high_gamma_slow_update():
    """High gamma keeps most weight on history → belief barely moves per step."""
    b = _uniform()
    sm = [0.95, 0.03, 0.01, 0.01]
    out = B.update_with_ewma(b, sm, gamma=0.95)
    # Change is small relative to a low-gamma update on the same evidence.
    slow_change = _l1(out, b)
    fast_change = _l1(B.update_with_ewma(b, sm, gamma=0.1), b)
    assert slow_change < fast_change
    assert slow_change < 0.2  # a single high-gamma step moves very little


def test_ewma_low_gamma_fast_update():
    """Low gamma puts most weight on new evidence → belief tracks it quickly."""
    b = _uniform()
    sm = [0.95, 0.03, 0.01, 0.01]
    out = B.update_with_ewma(b, sm, gamma=0.1)
    # With gamma→0 the belief approaches the (normalized) evidence distribution.
    p_obs = B.update_with_ewma(b, sm, gamma=0.0)
    assert _l1(out, p_obs) < _l1(b, p_obs)  # moved toward the evidence
    # Disengagement-like mass should dominate engaged after a class-0-peaked obs.
    assert out["actively_dis"] > out["engaged"]


def test_ewma_no_face_decays_toward_uniform():
    """A uniform (no-face) softmax drives the belief toward the uniform dist."""
    b = {"actively_dis": 0.97, "drifting": 0.01, "engaged": 0.01, "frustrated": 0.01}
    uniform_sm = [0.25, 0.25, 0.25, 0.25]
    for _ in range(200):
        b = B.update_with_ewma(b, uniform_sm, gamma=0.85)
    target = 1.0 / len(HIDDEN_STATES)
    for s in HIDDEN_STATES:
        assert math.isclose(b[s], target, abs_tol=1e-3), f"{s}={b[s]:.4f}"


def test_ewma_repeated_evidence_no_saturation():
    """Same evidence 100× must NOT push any state to 1.0 (anti-saturation)."""
    b = _uniform()
    sm = [0.95, 0.03, 0.01, 0.01]  # strongly class-0-peaked detection
    for _ in range(100):
        b = B.update_with_ewma(b, sm, gamma=0.85)
    assert _sums_to_one(b)
    # The EWMA fixed point is the (normalized) evidence over states, which is a
    # genuine mixture — no state can reach 1.0 no matter how many repetitions.
    assert max(b.values()) < 0.99
    assert b["actively_dis"] == max(b.values())  # still points at disengagement


def test_ewma_uniform_softmax_maps_to_uniform_evidence():
    """Sanity: uniform softmax → uniform P(O_t|s) because likelihood rows sum to 1."""
    u = _uniform()
    # gamma=0 returns pure evidence; uniform softmax must yield the uniform dist.
    out = B.update_with_ewma(u, [0.25, 0.25, 0.25, 0.25], gamma=0.0)
    target = 1.0 / len(HIDDEN_STATES)
    for s in HIDDEN_STATES:
        assert math.isclose(out[s], target, abs_tol=1e-9)


# ── Response update (unchanged) ────────────────────────────────────────────────

def test_ignored_raises_frustrated_more_than_clicked():
    b = _uniform()
    post_ignored = B.update_with_response(b, clicked=False)
    post_clicked = B.update_with_response(b, clicked=True)
    assert post_ignored["frustrated"] > post_clicked["frustrated"]


def test_normalize_handles_all_zero_input_as_uniform():
    z = {s: 0.0 for s in HIDDEN_STATES}
    out = B.normalize(z)
    assert _sums_to_one(out)
    target = 1.0 / len(HIDDEN_STATES)
    for s in HIDDEN_STATES:
        assert math.isclose(out[s], target, abs_tol=1e-6)
