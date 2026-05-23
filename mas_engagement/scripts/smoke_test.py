"""
smoke_test.py – Full MAS pipeline smoke test (no real camera or model required).

Usage (run from the mas_engagement/ directory):
    python scripts/smoke_test.py --inject-tier 1
    python scripts/smoke_test.py --inject-tier 2
    python scripts/smoke_test.py --inject-tier 3

What it does:
    1. Starts InterventionAgent + DeliveryAgent threads.
    2. After 2 s, writes pending_tier = <inject-tier> to the blackboard.
    3. Waits up to 15 s for one full delivery cycle to complete.
    4. Prints the intervention history and exits 0 on success, 1 on failure.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path

# Ensure the project root is on sys.path regardless of working directory.
_PKG_ROOT = Path(__file__).resolve().parent.parent          # mas_engagement/
_PROJECT_ROOT = _PKG_ROOT.parent                            # repo root
sys.path.insert(0, str(_PROJECT_ROOT))

from mas_engagement.blackboard import SharedState           # noqa: E402
from mas_engagement.logger import JsonlLogger               # noqa: E402
from mas_engagement.agents.intervention import InterventionAgent  # noqa: E402
from mas_engagement.agents.delivery import DeliveryAgent    # noqa: E402
from mas_engagement.config import LOG_DIR                   # noqa: E402

_INJECT_DELAY: float = 2.0
_RUN_TIMEOUT: float = 15.0
_WAIT_STEP: float = 0.1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger("smoke_test")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MAS full-pipeline smoke test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--inject-tier",
        choices=["1", "2", "3"],
        default="1",
        metavar="{1,2,3}",
        help="Tier to inject after 2 s (default: 1)",
    )
    return p.parse_args()


def _print_results(snap: dict, log_path: str) -> None:
    interventions = snap["interventions"]
    width = 60
    bar = "=" * width
    thin = "-" * width
    print()
    print(bar)
    print(f"  Intervention history ({len(interventions)} record(s))")
    print(bar)
    for iv in interventions:
        print(json.dumps(iv, indent=4, ensure_ascii=True))
    print(thin)
    print(f"  Total interventions : {snap['intervention_count']}")
    print(f"  Log file            : {log_path}")
    print(bar)


def main() -> None:
    args = _parse_args()
    tier = args.inject_tier

    # ── Shared infrastructure ─────────────────────────────────────────────────
    bb = SharedState(warmup_duration=0)
    log = JsonlLogger(log_dir=str(LOG_DIR))
    stop_event = threading.Event()

    # ── Agents ────────────────────────────────────────────────────────────────
    ia = InterventionAgent(bb, stop_event, logger=log)
    da = DeliveryAgent()

    t_ia = threading.Thread(target=ia.run, name="intervention", daemon=True)
    t_da = threading.Thread(target=da.run, args=(bb, log), name="delivery", daemon=True)

    # ── Start ─────────────────────────────────────────────────────────────────
    _log.info("Agents starting…")
    t_ia.start()
    t_da.start()

    # ── Inject tier after delay ───────────────────────────────────────────────
    _log.info("Waiting %.1f s before injecting tier-%s…", _INJECT_DELAY, tier)
    time.sleep(_INJECT_DELAY)
    _log.info(">>> set_pending_tier('%s')", tier)
    bb.set_pending_tier(tier, time.time())

    # ── Wait for delivery ─────────────────────────────────────────────────────
    deadline = time.time() + _RUN_TIMEOUT
    delivered = False
    while time.time() < deadline:
        if bb.snapshot()["intervention_count"] > 0:
            _log.info("Intervention delivered — stopping agents")
            delivered = True
            break
        time.sleep(_WAIT_STEP)

    if not delivered:
        _log.warning(
            "No intervention recorded within %.0f s timeout", _RUN_TIMEOUT
        )

    # ── Shutdown ──────────────────────────────────────────────────────────────
    stop_event.set()
    da.stop()
    t_ia.join(timeout=2)
    t_da.join(timeout=2)

    # ── Report ────────────────────────────────────────────────────────────────
    _print_results(bb.snapshot(), log.path)

    if not delivered:
        _log.error("SMOKE TEST FAILED")
        sys.exit(1)

    _log.info("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
