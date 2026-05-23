import math

import pytest

from mas_engagement.config import DEFAULT_PRIOR, HIDDEN_STATES
from mas_engagement.reasoning import belief as B


def _uniform():
    u = 1.0 / len(HIDDEN_STATES)
    return {s: u for s in HIDDEN_STATES}


def _sums_to_one(b: dict) -> bool:
    return math.isclose(sum(b.values()), 1.0, abs_tol=1e-9)


def test_all_updates_preserve_sum_to_one():
    b = _uniform()
    b1 = B.update_with_detection(b, [0.7, 0.2, 0.07, 0.03])
    b2 = B.update_with_response(b1, clicked=True)
    b3 = B.update_with_response(b2, clicked=False)
    b4 = B.decay_toward_uniform(b3, dt_seconds=5.0)
    b5 = B.normalize(b4)
    for post in (b1, b2, b3, b4, b5):
        assert _sums_to_one(post)
        assert set(post.keys()) == set(HIDDEN_STATES)


def test_detection_peaked_on_class0_shifts_to_dis_and_frustrated():
    b = _uniform()
    posterior = B.update_with_detection(b, [0.95, 0.03, 0.01, 0.01])
    # Both disengagement-like states must rise vs uniform.
    assert posterior["actively_dis"] > b["actively_dis"]
    assert posterior["frustrated"] > b["frustrated"]
    # They should dominate engaged.
    assert posterior["actively_dis"] > posterior["engaged"]
    assert posterior["frustrated"] > posterior["engaged"]


def test_decay_dt_zero_is_identity():
    b = {"actively_dis": 0.1, "drifting": 0.2, "engaged": 0.6, "frustrated": 0.1}
    out = B.decay_toward_uniform(b, dt_seconds=0.0)
    for s in HIDDEN_STATES:
        assert math.isclose(out[s], b[s], abs_tol=1e-9)


def test_decay_large_dt_approaches_default_prior():
    b = {"actively_dis": 0.97, "drifting": 0.01, "engaged": 0.01, "frustrated": 0.01}
    out = B.decay_toward_uniform(b, dt_seconds=1e9)
    for s in HIDDEN_STATES:
        assert math.isclose(out[s], DEFAULT_PRIOR[s], abs_tol=1e-6)


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
