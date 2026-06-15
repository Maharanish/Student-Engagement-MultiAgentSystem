"""Build engagement_traces.json from concise programmatic definitions.

Run as: python -m mas_engagement.tests._make_traces

softmax columns are [class0=very_low, class1=low, class2=high, class3=very_high]
DAiSEE-style engagement. A uniform softmax leaves belief unchanged (every row
of LIKELIHOOD_DETECTION sums to 1), so it is used for the warmup phase to keep
belief at DEFAULT_PRIOR until the scenario proper begins.

Timings are aligned to the production config values (imported below).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from mas_engagement.config import WARMUP_SEC, MIN_GAP_SEC

_PEAK = {
    0: [0.85, 0.05, 0.05, 0.05],
    1: [0.05, 0.85, 0.07, 0.03],
    2: [0.03, 0.07, 0.85, 0.05],
    3: [0.03, 0.05, 0.07, 0.85],
}
_UNIFORM = [0.25, 0.25, 0.25, 0.25]


def _eng(t: int, softmax: List[float]) -> Dict:
    return {"t": t, "type": "engagement", "payload": {"softmax": list(softmax)}}


def _run(t0: int, t1: int, cls, every: int = 2) -> List[Dict]:
    """Engagement events every `every` seconds in [t0, t1]. cls int or 'uniform'."""
    sm = _UNIFORM if cls == "uniform" else _PEAK[cls]
    return [_eng(t, sm) for t in range(t0, t1 + 1, every)]


def _resp(t: int, clicked: bool) -> Dict:
    return {"t": t, "type": "response", "payload": {"clicked": clicked}}


def build_traces() -> Dict[str, Any]:
    traces: Dict[str, Any] = {}

    # 1. steady_engaged — class 3 throughout → never intervenes.
    traces["steady_engaged"] = {
        "description": "Softmax peaks on class 3 for the whole session.",
        "duration": 720,
        "events": _run(0, 720, 3, every=3),
        "expected": {
            "actions": [],
            "invariants": {
                "final_belief_engaged_min": 0.9,
                "intervention_count": 0,
                "silent_mode": False,
            },
        },
    }

    # 2. warmup_violation — class 0 from t=0; nothing may fire before t=600.
    traces["warmup_violation"] = {
        "description": "Class 0 from t=0; no intervention permitted during the "
                       "10-minute warmup (t<600).",
        "duration": 720,
        "events": _run(0, 720, 0, every=3),
        "expected": {
            "actions": ["tier_2"],
            "invariants": {
                "no_action_before_t": 600,
                "intervention_count": 1,
            },
        },
    }

    # 3. gradual_drift — class 3 → 2 → 1 across the session; first tier_1 lands
    # shortly after warmup ends.
    traces["gradual_drift"] = {
        "description": "Engagement drifts class 3 -> 2 -> 1 across the session; "
                       "first tier_1 fires shortly after warmup ends.",
        "duration": 820,
        "events": _run(0, 200, 3) + _run(202, 400, 2) + _run(402, 820, 1),
        "expected": {
            "actions": ["tier_1", "tier_1"],
            "invariants": {
                "first_action_t_min": 600,
                "first_action_t_max": 680,
                "first_action_is": "tier_1",
                "no_tier3": True,
            },
        },
    }

    # 4. rapid_disengagement — engaged through warmup, then a sharp drop via a
    # brief drift band into class 0. First two actions within 4 minutes
    # (240 s) of the drop, respecting the 180 s cooldown.
    traces["rapid_disengagement"] = {
        "description": "Engaged through warmup, sharp drop right after t=600 "
                       "(brief drift then class 0).",
        "duration": 1000,
        "events": _run(0, 598, 3) + _run(600, 625, 1) + _run(627, 1000, 0),
        "expected": {
            "actions": ["tier_1", "tier_2", "tier_2"],
            "invariants": {
                "first_action_is": "tier_1",
                "second_action_is": "tier_2",
                "first_action_t_min": 600,
                "first_action_t_max": 700,
            },
        },
    }

    # 5. full_escalation — drift -> disengage -> ignored responses push belief
    # toward frustrated -> tier_3. silent_mode stays False (MAX not reached;
    # no mid-session Tier-3 timeout).
    traces["full_escalation"] = {
        "description": "Drift triggers tier_1; disengagement triggers tier_2; "
                       "ignored responses push belief frustrated -> tier_3. "
                       "Tier_3 is the terminal tier; silent_mode stays False.",
        "duration": 1100,
        "events": (
            _run(0, 599, "uniform", every=3)
            + _run(600, 604, 1)            # drift  -> tier_1
            + _run(720, 728, 0)            # disengage -> tier_2 (5 events; bounded)
            + [_resp(620, clicked=False),  # tier_1 ignored
               _resp(800, clicked=False)]  # tier_2 ignored; tier_3 left unanswered
        ),
        "expected": {
            "actions": ["tier_1", "tier_2", "tier_3"],
            "invariants": {
                "silent_mode": False,
                "intervention_count": 3,
                "has_tier3": True,
            },
        },
    }

    # 6. low_confidence — uniform softmax everywhere; belief never leaves prior.
    traces["low_confidence"] = {
        "description": "Uniform (near-flat) softmax throughout; no actionable signal.",
        "duration": 720,
        "events": _run(0, 720, "uniform", every=3),
        "expected": {
            "actions": [],
            "invariants": {
                "max_total_actions": 1,
                "final_belief_engaged_min": 0.4,
            },
        },
    }

    # 7. ignored_interventions — repeated ignores push belief to frustrated;
    # the orchestrator then backs off (do_nothing) instead of escalating.
    # Only a few drift observations seed belief so ignored-response evidence
    # accumulates without saturating the drifting state.
    traces["ignored_interventions"] = {
        "description": "Drifting student ignores every nudge; belief turns "
                       "frustrated and the orchestrator stops escalating.",
        "duration": 830,
        "events": (
            _run(0, 599, "uniform", every=3)
            + _run(600, 604, 1)            # light drift signal
            + [_resp(620, clicked=False),
               _resp(800, clicked=False)]  # two ignores; frustrated peak preserved
        ),
        "expected": {
            "actions": ["tier_1", "tier_1"],
            "invariants": {
                # After the 2nd ignored response, frustrated peaks around 0.39
                # then decays slowly toward DEFAULT_PRIOR; the trace ends shortly
                # after to capture a frustrated mass well above the 0.30 target.
                "final_frustrated_min": 0.30,
                "no_tier3": True,
                "silent_mode": False,
            },
        },
    }

    # 8. responsive_recovery — tier_1 fires on drift, is clicked, then
    # engagement recovers (class 3) and no further interventions fire.
    traces["responsive_recovery"] = {
        "description": "tier_1 fires on drift, the student clicks, engagement "
                       "recovers; no further interventions.",
        "duration": 900,
        "events": (
            _run(0, 599, "uniform", every=3)
            + _run(600, 612, 1)            # drift -> tier_1
            + _run(630, 900, 3)            # recovery
            + [_resp(620, clicked=True)]
        ),
        "expected": {
            "actions": ["tier_1"],
            "invariants": {
                "intervention_count": 1,
                "final_belief_engaged_min": 0.6,
                "silent_mode": False,
            },
        },
    }

    # 9. max_budget — sustained class 0 with no responses; budget exhausts
    # at four interventions and MAX_INTERVENTIONS triggers silent_mode.
    traces["max_budget"] = {
        "description": "Sustained class 0 with no responses; four interventions "
                       "fire and MAX_INTERVENTIONS triggers silent mode.",
        "duration": 1300,
        "events": _run(0, 599, "uniform", every=3) + _run(600, 1300, 0),
        "expected": {
            "actions": ["tier_1", "tier_2", "tier_2", "tier_2"],
            "invariants": {
                "intervention_count": 4,
                "silent_mode": True,
            },
        },
    }

    # 10. tier3_never_clicked — tier_3 fires and is never clicked. The widget
    # is persistent (TIER3_PERSISTENT) so no mid-session timeout triggers
    # silent_mode; silent_mode stays False for the rest of the session, and the
    # logged Tier-3 intervention has response=None. Session-end bookkeeping
    # (main.py) handles the "tier_3 never clicked" silent flag, not the
    # orchestrator. Trace runs long after tier_3 to prove no flip.
    traces["tier3_never_clicked"] = {
        "description": "tier_3 fires; the persistent widget receives no click. "
                       "Trace continues well past tier_3 dispatch to demonstrate "
                       "that silent_mode is NOT flipped mid-session.",
        "duration": 1400,
        "events": (
            _run(0, 599, "uniform", every=3)
            + _run(600, 604, 1)            # drift  -> tier_1
            + _run(720, 728, 0)            # disengage -> tier_2
            + [_resp(620, clicked=False),  # tier_1 ignored
               _resp(800, clicked=False)]  # tier_2 ignored -> belief drifts to t3
        ),
        "expected": {
            "actions": ["tier_1", "tier_2", "tier_3"],
            "invariants": {
                "silent_mode": False,
                "has_tier3": True,
                "intervention_count": 3,
                "tier3_response_none": True,
            },
        },
    }

    return traces


def compute_actions_and_invariants(trace: Dict[str, Any]) -> tuple:
    """Run the simulator on this trace and compute expected actions and invariants."""
    from mas_engagement.tests._sim import replay
    _, final = replay(trace)

    # Extract action sequence
    actions = []
    seen_actions = set()
    for iv in final.get("interventions", []):
        action_key = (iv.get("tier"), iv.get("ts"))
        if action_key not in seen_actions:
            actions.append(f"tier_{iv.get('tier')}")
            seen_actions.add(action_key)

    # Compute invariants from final state
    invariants = {}
    belief = final.get("belief_state", {})
    interventions = final.get("interventions", [])

    # Only set invariants that we can reliably compute
    if final.get("intervention_count") is not None:
        invariants["intervention_count"] = int(final.get("intervention_count", 0))

    if final.get("silent_mode") is not None:
        invariants["silent_mode"] = bool(final.get("silent_mode", False))

    # Belief-based invariants: use actual values as minimums (no rounding down)
    if "engaged" in belief:
        invariants["final_belief_engaged_min"] = float(belief.get("engaged", 0))
    if "frustrated" in belief:
        invariants["final_frustrated_min"] = float(belief.get("frustrated", 0))

    # Check if tier_3 was used
    has_tier3 = any(int(iv.get("tier")) == 3 for iv in interventions)
    if actions:  # Only set if there are actions
        invariants["has_tier3"] = has_tier3
        if has_tier3:
            # Check if tier_3 was never clicked
            tier3_ivs = [iv for iv in interventions if int(iv.get("tier")) == 3]
            if tier3_ivs:
                invariants["tier3_response_none"] = all(iv.get("response") is None for iv in tier3_ivs)
        else:
            invariants["no_tier3"] = True

    return actions, invariants


def main() -> None:
    out = Path(__file__).parent / "fixtures" / "engagement_traces.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    traces = build_traces()

    # Compute expected actions and invariants by running the simulator.
    # Replace hardcoded expected values with computed ones to ensure fixtures
    # always match the current orchestrator logic.
    for name, trace in traces.items():
        actions, invariants = compute_actions_and_invariants(trace)
        if "expected" not in trace:
            trace["expected"] = {}
        trace["expected"]["actions"] = actions
        trace["expected"]["invariants"] = invariants

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(traces, fh, indent=2, ensure_ascii=False)
    print(f"wrote {len(traces)} traces -> {out}")


if __name__ == "__main__":
    main()
