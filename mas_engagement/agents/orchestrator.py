"""Utility-based Orchestrator agent (Phase-2).

Per-tick responsibility:
  1. Snapshot blackboard.
  2. dt = now - last_poll_time
  3. belief = blackboard.get_belief_state()
  4. decay toward DEFAULT_PRIOR by dt
  5. apply update_with_detection for each new engagement record
  6. apply update_with_response for each new response record
  7. write belief back to blackboard
  8. choose_action; if not 'do_nothing' set pending_action
  9. log decision
 10. apply silent-mode triggers (mid-session: MAX_INTERVENTIONS only)

decide_action(snapshot, now) is a pure function: it consumes a snapshot that
includes a '_last_poll_time' field and applies decay + all evidence newer than
that, then returns (action, belief, utilities). No mutation, no I/O.

Tier-3 mid-session silent trigger has been removed: TIER3_PERSISTENT means the
widget waits indefinitely for a click, so there is no "ignored timeout" to
detect. Silent mode from an unanswered Tier-3 widget is applied at session
end (see main.py) when response_history has no entry for that intervention.
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple

_log = logging.getLogger(__name__)

_PKG_ROOT = Path(__file__).resolve().parent.parent

from mas_engagement.config import (  # noqa: E402
    DEFAULT_PRIOR,
    LOG_DIR,
    MAX_INTERVENTIONS,
    ORCHESTRATOR_POLL_HZ,
    WARMUP_DURATION,
)
from mas_engagement.reasoning.belief import (  # noqa: E402
    decay_toward_uniform,
    update_with_detection,
    update_with_response,
)
from mas_engagement.reasoning.utility import choose_action  # noqa: E402


_LAST_POLL_KEY = "_last_poll_time"


# ── Pure decision function ────────────────────────────────────────────────────

def decide_action(
    snapshot: Dict, now: float
) -> Tuple[str, Dict[str, float], Dict[str, float]]:
    """Pure: derive (action, belief, utilities) from a blackboard snapshot.

    The snapshot must include '_last_poll_time' (seconds, same clock as 'now').
    Engagement records with timestamp > _last_poll_time are applied as soft
    observations; response_history records with ts > _last_poll_time as
    binary click/ignore evidence.
    """
    last_poll = float(snapshot.get(_LAST_POLL_KEY, snapshot.get("session_start", now)))
    dt = max(0.0, now - last_poll)

    belief = snapshot.get("belief_state") or deepcopy(DEFAULT_PRIOR)
    belief = dict(belief)

    if dt > 0.0:
        belief = decay_toward_uniform(belief, dt)

    new_engagement = [
        r for r in snapshot.get("engagement_history", []) or []
        if "softmax" in r and float(r["timestamp"]) > last_poll
    ]
    for r in sorted(new_engagement, key=lambda r: r["timestamp"]):
        belief = update_with_detection(belief, list(r["softmax"]))

    # Pull response evidence from interventions list directly so we include
    # 'ignored' outcomes (response_history mirrors only real clicks).
    new_response = [
        r for r in snapshot.get("interventions", []) or []
        if r.get("response") is not None
        and float(r.get("response_ts", r["ts"])) > last_poll
    ]
    for r in sorted(new_response, key=lambda r: float(r.get("response_ts", r["ts"]))):
        clicked = r.get("response") not in (None, "ignored")
        belief = update_with_response(belief, clicked)

    action, utilities = choose_action(belief, snapshot, now)
    return action, belief, utilities


# ── Silent-mode trigger checks (pure) ─────────────────────────────────────────
#
# Mid-session, the orchestrator only enters silent mode when the intervention
# budget is exhausted. Unanswered Tier-3 widgets no longer trigger silent
# mode mid-session — TIER3_PERSISTENT means the widget waits for a click and
# Delivery handles the lifecycle. Any session-end "Tier-3 was never clicked"
# silent flagging is applied by main.py at shutdown time.

def _should_silent_max(snapshot: Dict) -> bool:
    return int(snapshot.get("intervention_count", 0)) >= MAX_INTERVENTIONS


# ── OrchestratorAgent ─────────────────────────────────────────────────────────

class OrchestratorAgent:
    """Utility-based orchestrator. Polls at ORCHESTRATOR_POLL_HZ."""

    def __init__(self, blackboard, stop_event: threading.Event, logger=None):
        self._bb = blackboard
        self._stop = stop_event
        self._logger = logger
        self._last_poll_time: float = -1.0  # so first tick consumes ts >= 0

    def run(self) -> None:
        poll_interval = 1.0 / ORCHESTRATOR_POLL_HZ
        while not self._stop.is_set():
            self.tick(time.time())
            time.sleep(poll_interval)

    def stop(self) -> None:
        self._stop.set()

    # Single tick — separated for testability.
    def tick(self, now: float) -> Tuple[str, Dict[str, float], Dict[str, float]]:
        # Silent-mode trigger (mid-session): only MAX_INTERVENTIONS now.
        # Tier-3 widgets are persistent (TIER3_PERSISTENT); silent mode from an
        # unanswered Tier-3 is applied at session end by main.py, not here.
        snap = self._bb.snapshot()
        if not snap.get("silent_mode") and _should_silent_max(snap):
            self._bb.set_silent_mode(True)
            self._log_event(now, "silent_mode_on", reason="max_interventions")

        snap = self._bb.snapshot()
        snap[_LAST_POLL_KEY] = self._last_poll_time

        action, belief, utilities = decide_action(snap, now)
        self._bb.set_belief_state(belief)

        if action != "do_nothing":
            self._bb.set_pending_action(action, now)

        n_eng = sum(
            1 for r in snap.get("engagement_history", []) or []
            if "softmax" in r and float(r["timestamp"]) > self._last_poll_time
        )
        n_resp = sum(
            1 for r in snap.get("response_history", []) or []
            if float(r["ts"]) > self._last_poll_time
        )

        self._log_event(
            now,
            "decision",
            belief=belief,
            utilities=utilities,
            action=action,
            n_engagement_evidence=n_eng,
            n_response_evidence=n_resp,
        )

        self._last_poll_time = now
        return action, belief, utilities

    def _log_event(self, ts: float, event: str, **kwargs) -> None:
        if self._logger is not None:
            self._logger.log("orchestrator", event, ts=ts, **kwargs)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OrchestratorAgent standalone runner")
    p.add_argument("--mock-class", type=int, choices=[0, 1, 2, 3], default=0,
                   help="Synthetic engagement softmax peak class (0=very low, 3=high)")
    p.add_argument("--duration", type=float, default=300.0,
                   help="Seconds to run before stopping (default 300)")
    return p.parse_args()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s – %(message)s")
    args = _parse_args()

    sys.path.insert(0, str(_PKG_ROOT.parent))
    from mas_engagement.blackboard import SharedState
    from mas_engagement.logger import JsonlLogger

    bb = SharedState(warmup_duration=WARMUP_DURATION)
    log = JsonlLogger(log_dir=str(LOG_DIR))
    stop = threading.Event()
    agent = OrchestratorAgent(bb, stop, logger=log)

    def _feeder():
        peak = [0.05] * 4
        peak[args.mock_class] = 0.85
        while not stop.is_set():
            bb.append_engagement(args.mock_class, 0.85, time.time(), softmax=peak)
            time.sleep(1.0)

    threading.Thread(target=_feeder, daemon=True).start()
    _log.info("OrchestratorAgent running for %.0f s (mock_class=%d)",
              args.duration, args.mock_class)
    t = threading.Thread(target=agent.run, daemon=True)
    t.start()
    try:
        t.join(timeout=args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        _log.info("Stopped. Log: %s", log.path)
