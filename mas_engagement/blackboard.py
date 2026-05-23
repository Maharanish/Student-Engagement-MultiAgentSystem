import copy
import threading
import time
from typing import Any, Dict, List, Optional

from mas_engagement.config import DEFAULT_PRIOR


class SharedState:
    """Thread-safe blackboard for the cooperative multi-agent engagement system."""

    def __init__(
        self,
        warmup_duration: float = 5.0,
        initial_belief: Optional[Dict[str, float]] = None,
    ):
        self._lock = threading.RLock()
        self._warmup_duration = warmup_duration
        now = time.time()
        self._state: Dict[str, Any] = {
            "engagement_history": [],
            "pending_action": None,
            "pending_message": None,
            "interventions": [],
            "response_history": [],
            "belief_state": (
                copy.deepcopy(initial_belief)
                if initial_belief is not None
                else copy.deepcopy(DEFAULT_PRIOR)
            ),
            "silent_mode": False,
            "intervention_count": 0,
            "warmup_until": now + warmup_duration,
            "session_start": now,
        }

    def set_session_start_time(self, now: float) -> None:
        """Reset the session clock; warmup window is re-anchored to `now`."""
        with self._lock:
            self._state["session_start"] = now
            self._state["warmup_until"] = now + self._warmup_duration

    # ── Engagement history ────────────────────────────────────────────────────

    def append_engagement(
        self,
        level: int,
        confidence: float,
        timestamp: float,
        softmax: Optional[List[float]] = None,
    ) -> None:
        """Append a single engagement observation; softmax vector is optional metadata."""
        record: Dict[str, Any] = {
            "level": level,
            "confidence": confidence,
            "timestamp": timestamp,
        }
        if softmax is not None:
            record["softmax"] = list(softmax)
        with self._lock:
            self._state["engagement_history"].append(record)

    def get_recent_engagement(self, window_sec: float) -> List[Dict]:
        """Return all engagement records within the last window_sec seconds."""
        with self._lock:
            cutoff = time.time() - window_sec
            return [e for e in self._state["engagement_history"] if e["timestamp"] >= cutoff]

    def snapshot(self) -> Dict[str, Any]:
        """Return a deep copy of the entire state dict."""
        with self._lock:
            return copy.deepcopy(self._state)

    # ── Pending orchestrator action (replaces pending_tier) ───────────────────

    def set_pending_action(self, action: str, ts: float) -> None:
        """Write a pending orchestrator action (e.g. 'tier_2', 'do_nothing').

        action must be one of mas_engagement.config.ACTIONS.
        Overwrites any previously unread value.
        """
        with self._lock:
            self._state["pending_action"] = {"action": action, "ts": ts}

    def consume_pending_action(self) -> Optional[Dict]:
        """Atomically read and clear the pending action; returns None if none was set."""
        with self._lock:
            value = self._state["pending_action"]
            self._state["pending_action"] = None
            return value

    # ── Pending outbound message ──────────────────────────────────────────────

    def set_pending_message(
        self,
        msg: str,
        tier: str,
        ts: float,
        msg_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None,
    ) -> None:
        """Write a pending outbound message; overwrites any previously unread value.

        `msg_id` identifies the specific message in the bank (tier_1/tier_2 picks).
        `payload` carries structured tier_3 content (prompt/continue_label/
        break_label/break_followup) so Delivery can render the persistent widget.
        `intent` is the message's purpose code (e.g. "link_to_material") used by
        Delivery to render a "Maksud: <bahasa label>" footer line on the card.
        """
        with self._lock:
            self._state["pending_message"] = {
                "msg": msg,
                "tier": tier,
                "ts": ts,
                "msg_id": msg_id,
                "payload": copy.deepcopy(payload) if payload is not None else None,
                "intent": intent,
            }

    def consume_pending_message(self) -> Optional[Dict]:
        """Atomically read and clear the pending message, returning None if none was set."""
        with self._lock:
            value = self._state["pending_message"]
            self._state["pending_message"] = None
            return value

    # ── Interventions and response history ────────────────────────────────────

    def append_intervention(self, tier: str, msg_id: str, ts: float, response: Any) -> None:
        """Append a completed intervention; mirror to response_history when responded.

        A response is treated as "responded" when it is not None and not the string
        'ignored' (which signals a toast that timed out without user interaction).
        """
        record = {"tier": tier, "msg_id": msg_id, "ts": ts, "response": response}
        with self._lock:
            self._state["interventions"].append(record)
            if response is not None and response != "ignored":
                self._state["response_history"].append(copy.deepcopy(record))

    def get_response_history(self, since_ts: float) -> List[Dict]:
        """Return responded interventions with ts >= since_ts (for Bayesian likelihood update)."""
        with self._lock:
            return [
                copy.deepcopy(r) for r in self._state["response_history"]
                if r["ts"] >= since_ts
            ]

    def record_response(self, ts: float, response: Any) -> None:
        """Record a response (including 'ignored') against the most recent intervention.

        Updates the intervention in-place with a 'response_ts' field so callers
        can distinguish dispatch time from response time. Mirrors to
        response_history when the response is a real user action (matches
        append_intervention semantics).
        """
        with self._lock:
            ivs = self._state["interventions"]
            if not ivs:
                return
            last = ivs[-1]
            last["response"] = response
            last["response_ts"] = ts
            if response is not None and response != "ignored":
                self._state["response_history"].append(copy.deepcopy(last))

    # ── Belief state ──────────────────────────────────────────────────────────

    def get_belief_state(self) -> Dict[str, float]:
        """Return a deep copy of the current belief distribution over HIDDEN_STATES."""
        with self._lock:
            return copy.deepcopy(self._state["belief_state"])

    def set_belief_state(self, belief: Dict[str, float]) -> None:
        """Replace the belief distribution wholesale (caller owns normalisation)."""
        with self._lock:
            self._state["belief_state"] = copy.deepcopy(belief)

    # ── Session flags ─────────────────────────────────────────────────────────

    def set_silent_mode(self, active: bool) -> None:
        """Enable or disable silent mode (suppresses outbound messages when True)."""
        with self._lock:
            self._state["silent_mode"] = active

    def inc_intervention_count(self) -> None:
        """Increment the session-level intervention counter by one."""
        with self._lock:
            self._state["intervention_count"] += 1

    def is_warmup_active(self, now: float) -> bool:
        """Return True if the warmup period has not yet elapsed relative to now."""
        with self._lock:
            return now < self._state["warmup_until"]
