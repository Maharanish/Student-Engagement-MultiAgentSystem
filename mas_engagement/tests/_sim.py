"""Shared simulator for orchestrator trace tests.

Replays a trace's event list against a SharedState blackboard, ticking the
orchestrator at 1 Hz across the trace's full duration, and returns the
recorded action sequence plus the final blackboard snapshot.

Response events are matched to interventions FIFO: the k-th response event
(once its timestamp has passed) is recorded against the k-th intervention
the orchestrator dispatches. This decouples trace authoring from the exact
tick on which an intervention fires.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Tuple

from mas_engagement.agents.orchestrator import (
    _should_silent_max,
    decide_action,
)
from mas_engagement.blackboard import SharedState
from mas_engagement.config import WARMUP_SEC

# Trace fixtures are authored on the assumption that the orchestrator's warmup
# elapses at trace-t = 600 s. We insulate the simulator from whatever value
# production WARMUP_SEC currently holds (it can be raised, lowered, or even
# disabled to 0 by user_config.json or a manual edit) by shifting session_start
# so the utility-gate elapses exactly at trace-t = _TRACE_WARMUP_SEC.
#
# Derivation: the gate is `(now - session_start) < WARMUP_SEC`. We want the
# gate to flip at now = _TRACE_WARMUP_SEC, so we pick
#     session_start = _TRACE_WARMUP_SEC - WARMUP_SEC.
# Verify:
#   WARMUP_SEC=0   → session_start=600; flip at (600 - 600) = 0 < 0 → no  ✓
#   WARMUP_SEC=600 → session_start=0;   flip at (600 - 0)   = 600 < 600 → no ✓
#   WARMUP_SEC=900 → session_start=-300; flip at (600+300)  = 900 < 900 → no ✓
_TRACE_WARMUP_SEC = 600.0


def replay(trace: Dict[str, Any]) -> Tuple[List[Tuple[float, str]], Dict]:
    """Replay a trace at 1 Hz; return (actions_log, final_snapshot).

    actions_log entries are (timestamp, action_string) for every tick that
    dispatched a non-do-nothing action.
    """
    bb = SharedState(warmup_duration=0)
    # Insulate trace replay from the production WARMUP_SEC value (see comment
    # above). Trace fixtures still see "warmup ends at t=600" regardless of
    # what is configured in config.py at the moment.
    bb._state["session_start"] = _TRACE_WARMUP_SEC - WARMUP_SEC
    bb._state["warmup_until"] = _TRACE_WARMUP_SEC

    events = sorted(trace["events"], key=lambda e: e["t"])
    end_t = int(trace.get("duration") or (max(e["t"] for e in events) + 120))

    # Engagement events grouped by integer second.
    eng_by_t: Dict[int, list] = {}
    # Response events as a FIFO queue ordered by availability time.
    response_q: deque = deque()
    for ev in events:
        if ev["type"] == "engagement":
            eng_by_t.setdefault(int(ev["t"]), []).append(ev)
        elif ev["type"] == "response":
            response_q.append((float(ev["t"]), bool(ev["payload"]["clicked"])))
    response_q = deque(sorted(response_q, key=lambda r: r[0]))

    last_poll = -1.0
    actions_log: List[Tuple[float, str]] = []

    for t in range(0, end_t + 1):
        now = float(t)

        # 1) Materialize engagement events scheduled for this tick.
        for ev in eng_by_t.get(t, []):
            sm = list(ev["payload"]["softmax"])
            level = int(max(range(len(sm)), key=lambda i: sm[i]))
            conf = float(max(sm))
            bb.append_engagement(level, conf, now, softmax=sm)

        # 2) Materialize the intervention emitted on the previous tick.
        consumed = bb.consume_pending_action()
        if consumed is not None and consumed["action"] != "do_nothing":
            tier = consumed["action"].split("_")[1]
            bb.append_intervention(
                tier=tier, msg_id=f"sim_{int(now)}", ts=now, response=None
            )
            bb.inc_intervention_count()
            actions_log.append((now, consumed["action"]))

        # 2b) Attach the next queued response to the latest unanswered
        #     intervention (FIFO: response k -> intervention k).
        if response_q and response_q[0][0] <= now:
            ivs = bb.snapshot()["interventions"]
            if ivs and ivs[-1].get("response") is None:
                _, clicked = response_q.popleft()
                bb.record_response(now, "clicked" if clicked else "ignored")

        # 3) Silent-mode trigger (mid-session): only MAX_INTERVENTIONS.
        snap = bb.snapshot()
        if not snap["silent_mode"] and _should_silent_max(snap):
            bb.set_silent_mode(True)

        # 4) Orchestrator decision tick.
        snap = bb.snapshot()
        snap["_last_poll_time"] = last_poll
        action, belief, _utilities = decide_action(snap, now)
        bb.set_belief_state(belief)
        if action != "do_nothing":
            bb.set_pending_action(action, now)
            bb.set_last_intervention_ts(now)
        last_poll = now

    # Drain a final pending action so the last decision is reflected.
    consumed = bb.consume_pending_action()
    if consumed is not None and consumed["action"] != "do_nothing":
        bb.append_intervention(
            tier=consumed["action"].split("_")[1],
            msg_id=f"sim_{end_t + 1}",
            ts=float(end_t + 1),
            response=None,
        )
        bb.inc_intervention_count()
        actions_log.append((float(end_t + 1), consumed["action"]))
        snap = bb.snapshot()
        if not snap["silent_mode"] and _should_silent_max(snap):
            bb.set_silent_mode(True)

    return actions_log, bb.snapshot()
