"""Session entry point: wires the four agents around a shared blackboard.

Startup seeds the belief state from DEFAULT_PRIOR and runs until SIGINT or the
duration timeout. `--dry-run` swaps real camera detection for a mock that posts
Dirichlet-sampled softmax vectors, so the pipeline runs in CI.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

import os
import sys
from export_log import transform_log_to_readable

if getattr(sys, 'frozen', False):
    _internal_dir = sys._MEIPASS
    _torch_lib_dir = os.path.join(_internal_dir, "torch", "lib")
    
    # Daftarkan folder _internal dan torch/lib ke dalam variabel environment PATH
    os.environ["PATH"] = _internal_dir + os.pathsep + _torch_lib_dir + os.pathsep + os.environ.get("PATH", "")
    
    # Untuk Python 3.8 ke atas di Windows, kita wajib mendaftarkan direktori secara eksplisit
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(_internal_dir)
        try:
            os.add_dll_directory(_torch_lib_dir)
        except FileNotFoundError:
            pass

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mas_engagement.agents.delivery import DeliveryAgent
from mas_engagement.agents.detection import DetectionAgent
from mas_engagement.agents.intervention import InterventionAgent
from mas_engagement.agents.orchestrator import OrchestratorAgent
from mas_engagement.blackboard import SharedState
from mas_engagement.config import (
    DEFAULT_USER_ID,
    LOG_DIR,
    NUM_CLASSES,
    SESSION_DURATION_SEC,
    USER_CONFIG_OVERRIDES,
    USER_CONFIG_PATH,
    USER_CONFIG_WARNINGS,
    WARMUP_DURATION,
)
from mas_engagement.logger import JsonlLogger

_log = logging.getLogger("main")

# LOG_DIR comes from config so it follows the PyInstaller-aware _BASE_DIR.
_LOG_DIR = LOG_DIR
_MOCK_DETECTION_INTERVAL = 10.0  # seconds between mock softmax posts


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Student-engagement multi-agent session")
    p.add_argument("--duration", type=float, default=SESSION_DURATION_SEC,
                   help="Session length in seconds (default: SESSION_DURATION_SEC)")
    p.add_argument("--user-id", default=DEFAULT_USER_ID,
                   help="Session label recorded in the log (session_start/session_end)")
    p.add_argument("--dry-run", action="store_true",
                   help="Replace camera detection with a mock softmax generator")
    p.add_argument("--mock-class", type=int, choices=[0, 1, 2, 3], default=None,
                   help="Force mock detector to peak at engagement class: "
                        "0=very_low (disengaged), 1=low, 2=high, 3=very_high (engaged). "
                        "If not set, uses random Dirichlet distribution (default CI behavior).")
    return p.parse_args()


def _mock_detection_loop(blackboard, logger, stop_event: threading.Event,
                         mock_class=None) -> None:
    """Dry-run detector: post a softmax every 10 seconds.

    When mock_class is None (default) the softmax is Dirichlet-sampled random
    noise. When mock_class is set (0-3) the softmax peaks at that class (0.85 at
    the chosen index, 0.05 elsewhere, then normalized) so a specific engagement
    trajectory can be forced for testing.
    """
    rng = np.random.default_rng()
    while not stop_event.is_set():
        if mock_class is not None:
            raw = [0.05] * NUM_CLASSES
            raw[mock_class] = 0.85
            total = sum(raw)
            softmax = [round(v / total, 6) for v in raw]
            level = mock_class
            confidence = float(softmax[mock_class])
        else:
            softmax = [round(float(p), 6) for p in rng.dirichlet(np.ones(NUM_CLASSES))]
            level = int(max(range(NUM_CLASSES), key=lambda i: softmax[i]))
            confidence = float(max(softmax))
        now = time.time()
        blackboard.append_engagement(level, confidence, now, softmax=softmax)
        # Mark when evidence finished being written, so Delivery can measure
        # end-to-end latency to the resulting notification (parity with the
        # real DetectionAgent).
        blackboard.set_last_evidence_ts(now)
        logger.log("detection", "engagement_posted",
                   level=level, confidence=confidence, softmax=softmax, mock=True)
        stop_event.wait(_MOCK_DETECTION_INTERVAL)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s – %(message)s")
    args = _parse_args()

    # ── Startup ───────────────────────────────────────────────────────────────
    logger = JsonlLogger(log_dir=str(_LOG_DIR))
    _log.info("Logging to %s", logger.path)

    # Surface any user_config.json overrides at the top of the session log.
    # config.py performed the actual override at module-load time; here we
    # only record what changed so the audit trail is preserved.
    if USER_CONFIG_OVERRIDES or USER_CONFIG_WARNINGS:
        logger.log(
            "main", "config_override",
            path=str(USER_CONFIG_PATH),
            overrides=USER_CONFIG_OVERRIDES,
            warnings=USER_CONFIG_WARNINGS,
        )
        if USER_CONFIG_OVERRIDES:
            _log.info("user_config.json applied: %s", USER_CONFIG_OVERRIDES)
        for w in USER_CONFIG_WARNINGS:
            _log.warning("user_config.json: %s", w)

    # Belief is seeded from DEFAULT_PRIOR (SharedState's default when no
    # initial_belief is supplied) — the per-user adaptive profile subsystem
    # has been removed.
    blackboard = SharedState(warmup_duration=WARMUP_DURATION)
    blackboard.set_session_start_time(time.time())
    logger.log("main", "session_start", user_id=args.user_id,
               duration=args.duration, dry_run=args.dry_run)

    stop_event = threading.Event()

    # ── Spawn agent threads (all daemon) ──────────────────────────────────────
    detection_agent = None
    delivery_agent = DeliveryAgent()
    intervention_agent = InterventionAgent(blackboard, stop_event, logger=logger)
    orchestrator_agent = OrchestratorAgent(blackboard, stop_event, logger=logger)

    threads: list[threading.Thread] = []

    if args.dry_run:
        threads.append(threading.Thread(
            target=_mock_detection_loop,
            args=(blackboard, logger, stop_event, args.mock_class),
            name="mock-detection", daemon=True))
    else:
        detection_agent = DetectionAgent(use_mock=False)
        threads.append(threading.Thread(
            target=detection_agent.run, args=(blackboard, logger),
            name="detection", daemon=True))

    threads.append(threading.Thread(
        target=intervention_agent.run, name="intervention", daemon=True))
    threads.append(threading.Thread(
        target=delivery_agent.run, args=(blackboard, logger),
        name="delivery", daemon=True))
    threads.append(threading.Thread(
        target=orchestrator_agent.run, name="orchestrator", daemon=True))

    for t in threads:
        t.start()
    _log.info("Started %d agent threads", len(threads))

    # ── SIGINT installs the same shutdown trigger as the duration timeout ─────
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())

    deadline = time.time() + args.duration
    try:
        while not stop_event.is_set() and time.time() < deadline:
            time.sleep(0.5)
    except KeyboardInterrupt:  # belt-and-braces if signal handler is bypassed
        pass

    # ── Shutdown (shared path) ────────────────────────────────────────────────
    _log.info("Shutting down …")
    stop_event.set()
    if detection_agent is not None:
        detection_agent.stop()        # releases the camera in its run() finally
    delivery_agent.stop()

    for t in threads:
        t.join(timeout=5.0)
        if t.is_alive():
            _log.warning("Thread %s did not stop within 5s", t.name)

    logger.log("main", "session_end", user_id=args.user_id)

    # ── Session-end Tier-3 bookkeeping ────────────────────────────────────────
    # If any Tier-3 widget was shown but never clicked (no matching entry in
    # response_history) flag silent_mode and log. Post-hoc only — the session
    # is already ending, so this affects the log, not runtime behavior.
    snap = blackboard.snapshot()
    responded_msg_ids = {r.get("msg_id") for r in snap.get("response_history", [])}
    tier3_unclicked = [
        iv for iv in snap.get("interventions", [])
        if str(iv.get("tier")) == "3" and iv.get("msg_id") not in responded_msg_ids
    ]
    if tier3_unclicked:
        blackboard.set_silent_mode(True)
        logger.log(
            "main", "tier3_never_clicked", action="silent_mode_set",
            count=len(tier3_unclicked),
            msg_ids=[iv.get("msg_id") for iv in tier3_unclicked],
        )
        
    if hasattr(logger, 'log_path'):
        transform_log_to_readable(logger.log_path)
    elif hasattr(logger, '_out_path'):
        transform_log_to_readable(logger._out_path)
    elif hasattr(logger, 'current_log_path'):
        transform_log_to_readable(logger.current_log_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
