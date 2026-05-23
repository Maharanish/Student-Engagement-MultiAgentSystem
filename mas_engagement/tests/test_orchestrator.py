"""Trace-driven tests for the utility-based Orchestrator.

Each of the 10 named traces in fixtures/engagement_traces.json is replayed at
1 Hz through the orchestrator (see _sim.replay). For every trace the test
asserts the exact non-do-nothing action sequence and all trace-specific
invariants (final belief properties, silent_mode, intervention_count, ...).
"""
import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from mas_engagement.agents.orchestrator import decide_action
from mas_engagement.config import HIDDEN_STATES
from mas_engagement.tests._sim import replay

_FIXTURES = Path(__file__).parent / "fixtures" / "engagement_traces.json"


def _load_traces() -> List[tuple]:
    with open(_FIXTURES, encoding="utf-8") as fh:
        data = json.load(fh)
    return [(name, trace) for name, trace in data.items()]


_TRACES = _load_traces()


def _check_invariants(
    name: str,
    invariants: Dict[str, Any],
    actions_log: List[tuple],
    final: Dict[str, Any],
) -> None:
    """Assert every trace-specific invariant; each failure names the trace + key."""
    actions = [a for _t, a in actions_log]
    times = [t for t, _a in actions_log]
    belief = final["belief_state"]

    for key, val in invariants.items():
        ctx = f"[{name}] invariant {key}={val!r}"

        if key == "intervention_count":
            assert final["intervention_count"] == val, (
                f"{ctx}: got {final['intervention_count']}")

        elif key == "silent_mode":
            assert final["silent_mode"] is val, (
                f"{ctx}: got {final['silent_mode']}")

        elif key == "final_belief_engaged_min":
            assert belief["engaged"] >= val, (
                f"{ctx}: got engaged={belief['engaged']:.3f}")

        elif key == "final_frustrated_min":
            assert belief["frustrated"] >= val, (
                f"{ctx}: got frustrated={belief['frustrated']:.3f}")

        elif key == "no_action_before_t":
            early = [t for t in times if t < val]
            assert not early, f"{ctx}: actions fired early at {early}"

        elif key == "first_action_t_min":
            assert times, f"{ctx}: no actions fired"
            assert times[0] >= val, f"{ctx}: first action at t={times[0]}"

        elif key == "first_action_t_max":
            assert times, f"{ctx}: no actions fired"
            assert times[0] <= val, f"{ctx}: first action at t={times[0]}"

        elif key == "first_action_is":
            assert actions, f"{ctx}: no actions fired"
            assert actions[0] == val, f"{ctx}: first action is {actions[0]}"

        elif key == "second_action_is":
            assert len(actions) >= 2, f"{ctx}: fewer than 2 actions"
            assert actions[1] == val, f"{ctx}: second action is {actions[1]}"

        elif key == "no_tier3":
            assert "tier_3" not in actions, f"{ctx}: tier_3 fired in {actions}"

        elif key == "has_tier3":
            assert ("tier_3" in actions) is val, f"{ctx}: actions={actions}"

        elif key == "max_total_actions":
            assert len(actions) <= val, f"{ctx}: got {len(actions)} actions"

        elif key == "min_total_actions":
            assert len(actions) >= val, f"{ctx}: got {len(actions)} actions"

        elif key == "tier3_response_none":
            tier3s = [iv for iv in final["interventions"]
                      if str(iv.get("tier")) == "3"]
            assert tier3s, f"{ctx}: no tier_3 intervention found"
            assert all(iv.get("response") is None for iv in tier3s) is val, (
                f"{ctx}: tier_3 responses are "
                f"{[iv.get('response') for iv in tier3s]}")

        else:
            raise AssertionError(f"[{name}] unknown invariant key: {key}")


@pytest.mark.parametrize("name,trace", _TRACES, ids=[t[0] for t in _TRACES])
def test_trace(name: str, trace: Dict[str, Any]) -> None:
    """Replay one trace; assert the exact action sequence and all invariants."""
    actions_log, final = replay(trace)
    actions = [a for _t, a in actions_log]

    expected = trace["expected"]

    # Exact non-do-nothing action sequence.
    if expected.get("actions") is not None:
        assert actions == expected["actions"], (
            f"[{name}] action sequence mismatch:\n"
            f"  expected: {expected['actions']}\n"
            f"  got:      {actions}\n"
            f"  timeline: {actions_log}"
        )

    # Belief stays a normalized distribution over HIDDEN_STATES.
    belief = final["belief_state"]
    assert set(belief.keys()) == set(HIDDEN_STATES), f"[{name}] belief keys wrong"
    assert abs(sum(belief.values()) - 1.0) < 1e-6, (
        f"[{name}] belief not normalized: sum={sum(belief.values())}")

    _check_invariants(name, expected.get("invariants", {}), actions_log, final)


def test_all_ten_traces_present() -> None:
    """The fixture must define exactly the 10 required traces."""
    required = {
        "steady_engaged", "warmup_violation", "gradual_drift",
        "rapid_disengagement", "full_escalation", "low_confidence",
        "ignored_interventions", "responsive_recovery", "max_budget",
        "tier3_never_clicked",
    }
    names = {n for n, _ in _TRACES}
    assert names == required, f"trace set mismatch: {names ^ required}"


def test_decide_action_is_pure() -> None:
    """decide_action must not mutate the snapshot it is given."""
    _actions, final = replay(dict(_TRACES)["gradual_drift"])
    snapshot = copy.deepcopy(final)
    snapshot["_last_poll_time"] = snapshot["session_start"]
    before = copy.deepcopy(snapshot)

    action, belief, utilities = decide_action(snapshot, now=snapshot["warmup_until"] + 10)

    assert snapshot == before, "decide_action mutated its snapshot argument"
    assert isinstance(action, str)
    assert set(belief.keys()) == set(HIDDEN_STATES)
    assert abs(sum(belief.values()) - 1.0) < 1e-6
    assert set(utilities.keys()) == {"tier_1", "tier_2", "tier_3", "do_nothing"}
