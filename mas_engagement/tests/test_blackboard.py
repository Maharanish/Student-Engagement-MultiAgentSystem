import json
import threading
import time

import pytest

from mas_engagement.blackboard import SharedState
from mas_engagement.config import DEFAULT_PRIOR, MESSAGES_PATH


def test_append_and_get_recent_engagement():
    state = SharedState(warmup_duration=0)
    now = time.time()
    state.append_engagement(2, 0.9, now)
    state.append_engagement(1, 0.7, now - 60)

    recent = state.get_recent_engagement(window_sec=10)

    assert len(recent) == 1
    assert recent[0]["level"] == 2


def test_append_engagement_persists_softmax_when_given():
    state = SharedState(warmup_duration=0)
    now = time.time()
    state.append_engagement(2, 0.85, now, softmax=[0.05, 0.05, 0.85, 0.05])

    snap = state.snapshot()
    record = snap["engagement_history"][0]
    assert record["softmax"] == [0.05, 0.05, 0.85, 0.05]
    assert record["level"] == 2
    assert record["confidence"] == 0.85


def test_consume_pending_action_is_atomic():
    state = SharedState(warmup_duration=0)
    state.set_pending_action("tier_2", time.time())

    first = state.consume_pending_action()
    second = state.consume_pending_action()

    assert first is not None
    assert first["action"] == "tier_2"
    assert second is None


def test_pending_action_supports_do_nothing():
    state = SharedState(warmup_duration=0)
    state.set_pending_action("do_nothing", time.time())

    consumed = state.consume_pending_action()

    assert consumed["action"] == "do_nothing"
    assert state.consume_pending_action() is None


def test_consume_pending_message_is_atomic():
    state = SharedState(warmup_duration=0)
    state.set_pending_message("re-engage now", "tier1", time.time())

    first = state.consume_pending_message()
    second = state.consume_pending_message()

    assert first["msg"] == "re-engage now"
    assert second is None


def test_snapshot_is_independent_deep_copy():
    state = SharedState(warmup_duration=0)
    state.append_engagement(3, 0.95, time.time())

    snap = state.snapshot()
    state.append_engagement(1, 0.5, time.time())

    assert len(snap["engagement_history"]) == 1
    assert len(state.snapshot()["engagement_history"]) == 2


def test_belief_state_initialized_from_default_prior():
    state = SharedState(warmup_duration=0)

    belief = state.get_belief_state()

    assert belief == DEFAULT_PRIOR
    assert belief is not DEFAULT_PRIOR  # must be a copy


def test_belief_state_round_trip_and_isolation():
    state = SharedState(warmup_duration=0)

    new_belief = {"engaged": 0.1, "drifting": 0.4, "disengaged": 0.5}
    state.set_belief_state(new_belief)

    got = state.get_belief_state()
    assert got == new_belief

    # mutating returned copy must not affect stored state
    got["engaged"] = 0.99
    assert state.get_belief_state()["engaged"] == 0.1

    # mutating input dict after set must not affect stored state
    new_belief["disengaged"] = 0.0
    assert state.get_belief_state()["disengaged"] == 0.5


def test_response_history_only_includes_responded_interventions():
    state = SharedState(warmup_duration=0)
    t0 = time.time()
    state.append_intervention("1", "m1", t0, response="ignored")
    state.append_intervention("2", "m2", t0 + 1, response="clicked")
    state.append_intervention("3", "m3", t0 + 2, response=None)
    state.append_intervention("3", "m4", t0 + 3, response="Need help")

    responded = state.get_response_history(since_ts=0.0)
    assert [r["msg_id"] for r in responded] == ["m2", "m4"]

    # interventions list still contains all 4
    assert len(state.snapshot()["interventions"]) == 4


def test_response_history_filters_by_since_ts():
    state = SharedState(warmup_duration=0)
    t0 = time.time()
    state.append_intervention("1", "old", t0 - 100, response="clicked")
    state.append_intervention("2", "new", t0, response="clicked")

    recent = state.get_response_history(since_ts=t0 - 10)
    assert [r["msg_id"] for r in recent] == ["new"]


def test_concurrent_hammer_no_lost_writes():
    state = SharedState(warmup_duration=0)
    n_threads = 4
    n_ops = 1000

    def worker():
        for _ in range(n_ops):
            state.append_engagement(1, 0.8, time.time())
            state.inc_intervention_count()

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = state.snapshot()
    assert len(snap["engagement_history"]) == n_threads * n_ops
    assert snap["intervention_count"] == n_threads * n_ops


# ── messages.json startup validation ──────────────────────────────────────────

def test_messages_json_parses_with_validated_shape():
    """messages.json must parse and carry tier_1 (len 3), tier_2 (len 3),
    tier_3 (single object with the persistent-widget fields)."""
    with open(MESSAGES_PATH, encoding="utf-8") as fh:
        data = json.load(fh)

    assert isinstance(data.get("tier_1"), list) and len(data["tier_1"]) == 3
    assert isinstance(data.get("tier_2"), list) and len(data["tier_2"]) == 3

    for tier_key in ("tier_1", "tier_2"):
        for item in data[tier_key]:
            assert {"id", "intent", "text"} <= item.keys(), (
                f"{tier_key} item missing fields: {item}")

    t3 = data.get("tier_3")
    assert isinstance(t3, dict), "tier_3 must be an object, not a list"
    for field in ("prompt", "continue_label", "break_label", "break_followup"):
        assert field in t3, f"tier_3 missing {field!r}"


# ── Pending-message payload (tier_3 carries structured widget content) ────────

def test_break_until_round_trip_and_clear():
    """break_until pauses interventions via the utility-side cooldown gate;
    clear_break() resumes normal operation."""
    import math
    from mas_engagement.reasoning.utility import _cooldown_blocked

    state = SharedState(warmup_duration=0)
    snap = state.snapshot()
    assert snap["break_until"] == 0.0

    # set_break_until → cooldown blocks until that ts has passed.
    now = time.time()
    state.set_break_until(now + 120.0)
    assert state.get_break_until() == now + 120.0
    snap = state.snapshot()
    # Force the warmup/start gates out of the way so we only test break_until.
    snap["silent_mode"] = False
    snap["session_start"] = now - 1_000_000
    snap["interventions"] = []
    assert _cooldown_blocked(snap, now) is True
    assert _cooldown_blocked(snap, now + 200.0) is False    # window passed

    # math.inf is the "indefinite pause" mode (used while the return widget
    # is on screen).
    state.set_break_until(math.inf)
    snap = state.snapshot()
    snap["silent_mode"] = False
    snap["session_start"] = now - 1_000_000
    snap["interventions"] = []
    assert _cooldown_blocked(snap, now + 10_000_000) is True

    # clear_break resumes.
    state.clear_break()
    assert state.get_break_until() == 0.0
    snap = state.snapshot()
    snap["silent_mode"] = False
    snap["session_start"] = now - 1_000_000
    snap["interventions"] = []
    assert _cooldown_blocked(snap, now) is False


def test_set_pending_message_carries_intent_for_tier1():
    """Tier-1/Tier-2 pending messages must carry the message's intent code so
    Delivery can render a 'Maksud: <bahasa label>' line on the card."""
    state = SharedState(warmup_duration=0)
    state.set_pending_message(
        "halo", "1", time.time(), msg_id="t1_a", intent="link_to_material",
    )
    msg = state.consume_pending_message()
    assert msg["tier"] == "1"
    assert msg["msg_id"] == "t1_a"
    assert msg["intent"] == "link_to_material"


def test_set_pending_message_carries_payload_for_tier3():
    state = SharedState(warmup_duration=0)
    t3 = {
        "prompt": "p", "continue_label": "c",
        "break_label": "b", "break_followup": "f",
    }
    state.set_pending_message(
        t3["prompt"], "3", time.time(), msg_id="tier_3", payload=t3,
    )
    msg = state.consume_pending_message()
    assert msg["tier"] == "3"
    assert msg["msg_id"] == "tier_3"
    assert msg["payload"] == t3
    # Independent deep copy: caller mutation must not affect stored state.
    msg["payload"]["prompt"] = "MUTATED"
    state.set_pending_message(
        t3["prompt"], "3", time.time(), msg_id="tier_3", payload=t3,
    )
    assert state.consume_pending_message()["payload"]["prompt"] == "p"
