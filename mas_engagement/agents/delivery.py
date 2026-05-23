import argparse
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

_log = logging.getLogger(__name__)

_PKG_ROOT = Path(__file__).resolve().parent.parent

from mas_engagement.config import (  # noqa: E402
    DELIVERY_POLL_HZ,
    LOG_DIR,
    TIER2_TIMEOUT_SEC,
    TIER3_BREAK_NUDGE_SEC,
    TOAST_TIMEOUT_SEC,
)

try:
    from mas_engagement.overlay.tk_overlay import (
        intent_to_purpose,
        show_tier3_widget,
        show_toast,
    )
    _OVERLAY_AVAILABLE = True
except Exception:
    _OVERLAY_AVAILABLE = False

# Per-tier auto-dismiss timeout (seconds). Tier-3 is persistent — no entry here.
_TIER_TIMEOUT = {"1": TOAST_TIMEOUT_SEC, "2": TIER2_TIMEOUT_SEC}

try:
    from win10toast import ToastNotifier as _ToastNotifier
    _WIN10TOAST_AVAILABLE = True
except ImportError:
    _WIN10TOAST_AVAILABLE = False

_POLL_INTERVAL = 1.0 / DELIVERY_POLL_HZ

_TIER_TEST_MESSAGES: Dict[str, str] = {
    "1": "Quick check-in: are you still following along?",
    "2": "It looks like you might be losing focus — take a moment if needed.",
    "3": "Extended disengagement detected. Your instructor has been notified.",
}


def _fallback_toast(text: str, timeout: int = TOAST_TIMEOUT_SEC) -> None:
    """Deliver a notification via win10toast, or print to stderr as last resort."""
    if _WIN10TOAST_AVAILABLE:
        try:
            _ToastNotifier().show_toast(
                "Student Engagement",
                text,
                duration=timeout,
                threaded=True,
            )
            return
        except Exception as exc:
            _log.debug("win10toast failed: %s", exc)
    print(f"[DELIVERY FALLBACK] {text}", flush=True)


class DeliveryAgent:
    """Polls the blackboard for pending messages and delivers them via OS overlays."""

    def __init__(self):
        self._stop_event = threading.Event()

    def run(self, blackboard, logger) -> None:
        """Blocking 10 Hz poll loop; returns when stop() is called."""
        while not self._stop_event.is_set():
            item = blackboard.consume_pending_message()
            if item is not None:
                self._deliver(item, blackboard, logger)
            time.sleep(_POLL_INTERVAL)

    def stop(self) -> None:
        """Signal the poll loop to exit after the current sleep."""
        self._stop_event.set()

    # ── Delivery pipeline ─────────────────────────────────────────────────────

    def _deliver(self, item: Dict[str, Any], blackboard, logger) -> None:
        msg: str = item["msg"]
        tier: str = str(item["tier"])
        ts: float = item["ts"]
        msg_id: str = item.get("msg_id") or uuid.uuid4().hex[:8]
        payload: Optional[Dict[str, Any]] = item.get("payload")
        intent: Optional[str] = item.get("intent")

        t_show = time.time()
        response = self._show_overlay(msg, tier, payload, intent)
        latency_ms = (time.time() - t_show) * 1000.0

        blackboard.append_intervention(tier, msg_id, ts, response)
        blackboard.inc_intervention_count()
        # response is a structured action dict for tier_3, a string for tier_1/2.
        response_action = (
            response.get("action") if isinstance(response, dict) else response
        )
        logger.log(
            "delivery",
            "intervention_delivered",
            tier=tier,
            msg_id=msg_id,
            response=response,
            response_action=response_action,
            latency_ms=round(latency_ms, 1),
        )
        _log.info(
            "tier=%s msg_id=%s response=%r latency=%.0f ms",
            tier, msg_id, response_action, latency_ms,
        )

    def _show_overlay(
        self,
        msg: str,
        tier: str,
        payload: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None,
    ) -> Any:
        """Render the message. Tier-3 is persistent; tier-1/2 auto-dismissing toasts."""
        try:
            if not _OVERLAY_AVAILABLE:
                raise RuntimeError("tk_overlay unavailable")
            if tier == "3":
                p = payload or {}
                action = show_tier3_widget(
                    prompt=p.get("prompt", msg),
                    continue_label=p.get("continue_label", "Lanjutkan"),
                    break_label=p.get("break_label", "Istirahat sebentar"),
                    break_followup=p.get("break_followup", ""),
                    break_nudge_sec=TIER3_BREAK_NUDGE_SEC,
                )
                return {"action": action}
            # tier_1 / tier_2 auto-dismissing toast — purpose = "Maksud: …" line.
            timeout = _TIER_TIMEOUT.get(tier, TOAST_TIMEOUT_SEC)
            purpose = intent_to_purpose(intent)
            return show_toast(
                msg, timeout=timeout, tier=tier, purpose=purpose,
            )
        except Exception as exc:
            _log.warning("Overlay failed (%s) – using fallback toast", exc)
            _fallback_toast(msg)
            return "fallback"


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeliveryAgent standalone runner")
    p.add_argument(
        "--test-tier",
        choices=["1", "2", "3"],
        metavar="{1,2,3}",
        help="Inject a synthetic tier-N message and exit after delivery",
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
    log = JsonlLogger(log_dir=str(LOG_DIR))
    agent = DeliveryAgent()

    if args.test_tier:
        tier = args.test_tier
        ts = time.time()
        import json
        from mas_engagement.config import MESSAGES_PATH
        with open(MESSAGES_PATH, encoding="utf-8") as fh:
            bank = json.load(fh)

        if tier == "3":
            t3 = bank["tier_3"]
            bb.set_pending_message(
                t3["prompt"], "3", ts, msg_id="tier_3", payload=t3,
            )
        else:
            # Use the first validated Bahasa message of the requested tier so the
            # CLI demo exercises the same render path runtime uses.
            pool = bank[f"tier_{tier}"]
            chosen = pool[0]
            bb.set_pending_message(
                chosen["text"], tier, ts,
                msg_id=chosen["id"],
                intent=chosen.get("intent", ""),
            )
        _log.info("Injected tier-%s test message", tier)

        def _run():
            agent.run(bb, log)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        deadline = time.time() + 60.0
        while time.time() < deadline:
            if bb.snapshot()["intervention_count"] > 0:
                break
            time.sleep(0.05)

        agent.stop()
        t.join(timeout=3)
        snap = bb.snapshot()
        _log.info(
            "Done – interventions=%d  log=%s",
            snap["intervention_count"],
            log.path,
        )
    else:
        _log.info("DeliveryAgent running (Ctrl-C to stop)")
        try:
            agent.run(bb, log)
        except KeyboardInterrupt:
            agent.stop()
            _log.info("Stopped. Log: %s", log.path)
