import argparse
import json
import logging
import random
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

_log = logging.getLogger(__name__)

_PKG_ROOT = Path(__file__).resolve().parent.parent

from mas_engagement.config import (  # noqa: E402
    INTERVENTION_POLL_HZ,
    MESSAGES_PATH,
    NO_REPEAT_WINDOW,
    LOG_DIR,
)

_REQUIRED_TIERS = frozenset({"tier_1", "tier_2", "tier_3"})
_POLL_INTERVAL = 1.0 / INTERVENTION_POLL_HZ


def _load_and_validate(path: Path) -> Dict[str, List[dict]]:
    """Load messages.json and raise ValueError immediately if the structure is malformed."""
    if not path.is_file():
        raise FileNotFoundError(f"messages.json not found at {path}")

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    missing = _REQUIRED_TIERS - data.keys()
    if missing:
        raise ValueError(f"messages.json missing tier keys: {sorted(missing)}")

    # tier_1 / tier_2 are non-empty lists of {id, intent, text}.
    for tier_key in ("tier_1", "tier_2"):
        msgs = data[tier_key]
        if not isinstance(msgs, list) or len(msgs) == 0:
            raise ValueError(f"messages.json: '{tier_key}' must be a non-empty list")
        for i, msg in enumerate(msgs):
            for field in ("id", "intent", "text"):
                if field not in msg:
                    raise ValueError(f"messages.json: {tier_key}[{i}] missing {field!r}")

    # tier_3 is a single object with the persistent-widget fields.
    t3 = data["tier_3"]
    if not isinstance(t3, dict):
        raise ValueError("messages.json: 'tier_3' must be an object")
    for field in ("prompt", "continue_label", "break_label", "break_followup"):
        if field not in t3:
            raise ValueError(f"messages.json: tier_3 missing {field!r}")

    _log.info(
        "Loaded messages.json — tier_1:%d  tier_2:%d  tier_3:object",
        len(data["tier_1"]), len(data["tier_2"]),
    )
    return data


class InterventionAgent:
    """Selects and posts intervention messages to the blackboard based on tier decisions."""

    def __init__(
        self,
        blackboard,
        stop_event: threading.Event,
        logger=None,
        messages_path: Optional[Path] = None,
    ):
        self._bb = blackboard
        self._stop = stop_event
        self._logger = logger
        self._messages = _load_and_validate(messages_path or MESSAGES_PATH)
        # Only tier_1/tier_2 have message pools; tier_3 is a single fixed widget.
        self._recent: Dict[str, deque] = {
            tier: deque(maxlen=NO_REPEAT_WINDOW) for tier in ("tier_1", "tier_2")
        }

    def run(self) -> None:
        """Event-driven loop; blocks on the action queue, wakes every 0.5 s to
        check stop_event so graceful shutdown is never delayed more than 0.5 s."""
        while not self._stop.is_set():
            item = self._bb.consume_pending_action(timeout=0.5)
            if item is not None:
                self._handle_action(str(item["action"]), float(item["ts"]))

    def _handle_action(self, action: str, ts: float) -> None:
        """Route an orchestrator action: dispatch tier_N as before, skip do_nothing."""
        if action == "do_nothing":
            _log.info("action=do_nothing — no message dispatched")
            if self._logger is not None:
                self._logger.log("intervention", "skip_do_nothing", ts=ts)
            return

        if not action.startswith("tier_"):
            _log.warning("Unknown action %r — ignoring", action)
            return

        tier = action.split("_", 1)[1]
        if tier not in {"1", "2", "3"}:
            _log.warning("Unsupported tier in action %r — ignoring", action)
            return

        self._handle_tier(tier, ts)

    def pick(self, tier: str) -> dict:
        """Pick a message for the given tier without touching the blackboard.

        For tier '1'/'2' this returns a random message dict honouring
        NO_REPEAT_WINDOW. For tier '3' it returns the single tier_3 widget
        object verbatim (there is no selection logic — only one Tier-3 exists).
        """
        tier_key = f"tier_{tier}"
        if tier_key not in self._messages:
            raise ValueError(f"Unknown tier: {tier!r}. Expected '1', '2', or '3'.")
        if tier == "3":
            return dict(self._messages["tier_3"])
        return self._select(tier_key)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _handle_tier(self, tier: str, ts: float) -> None:
        if tier == "3":
            self._dispatch_tier3(ts)
            return

        tier_key = f"tier_{tier}"
        chosen = self._select(tier_key)
        self._bb.set_pending_message(
            chosen["text"], tier, ts,
            msg_id=chosen["id"],
            intent=chosen.get("intent", ""),
        )
        if self._logger is not None:
            self._logger.log(
                "intervention", "message_selected",
                tier=tier, msg_id=chosen["id"], intent=chosen.get("intent", ""),
            )
        _log.info("tier=%s  selected msg_id=%s", tier, chosen["id"])

    def _dispatch_tier3(self, ts: float) -> None:
        """Tier_3 is a fixed persistent widget — no selection, just forward it."""
        t3 = self._messages["tier_3"]
        self._bb.set_pending_message(
            t3["prompt"], "3", ts, msg_id="tier_3", payload=dict(t3),
        )
        if self._logger is not None:
            self._logger.log(
                "intervention", "message_selected",
                tier="3", msg_id="tier_3", intent="persistent_check_in",
            )
        _log.info("tier=3  persistent widget dispatched")

    def _select(self, tier_key: str) -> dict:
        """Pick a random message from tier_key, honouring the NO_REPEAT_WINDOW."""
        pool = self._messages[tier_key]
        seen = set(self._recent[tier_key])
        candidates = [m for m in pool if m["id"] not in seen]

        if not candidates:
            # All messages are in the window; reset and use the full pool.
            _log.debug("%s repeat window exhausted — using full pool", tier_key)
            candidates = pool

        chosen = random.choice(candidates)
        self._recent[tier_key].append(chosen["id"])
        return chosen


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="InterventionAgent standalone runner")
    p.add_argument(
        "--test-tier",
        choices=["1", "2", "3"],
        metavar="{1,2,3}",
        help="Pick and print one message for the given tier, then exit",
    )
    return p.parse_args()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s – %(message)s")
    args = _parse_args()

    sys.path.insert(0, str(_PKG_ROOT.parent))
    from mas_engagement.blackboard import SharedState
    from mas_engagement.logger import JsonlLogger

    bb = SharedState(warmup_duration=0)
    stop = threading.Event()
    log = JsonlLogger(log_dir=str(LOG_DIR))
    agent = InterventionAgent(bb, stop, logger=log)

    if args.test_tier:
        chosen = agent.pick(args.test_tier)
        print(json.dumps(chosen, indent=2, ensure_ascii=False))
    else:
        _log.info("InterventionAgent running (Ctrl-C to stop)")
        try:
            agent.run()
        except KeyboardInterrupt:
            stop.set()
            _log.info("Stopped.")
