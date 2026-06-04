"""Session entry point: wires the four agents around a shared blackboard.

Startup loads the user's adaptive profile and seeds the belief state with it;
shutdown (SIGINT or duration timeout) folds the session log back into the
profile and persists it. `--dry-run` swaps real camera detection for a mock
that posts Dirichlet-sampled softmax vectors, so the pipeline runs in CI.
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
    HIDDEN_STATES,
    LOG_DIR,
    NUM_CLASSES,
    SESSION_DURATION_SEC,
    USER_CONFIG_OVERRIDES,
    USER_CONFIG_PATH,
    USER_CONFIG_WARNINGS,
    WARMUP_DURATION,
)
from mas_engagement.logger import JsonlLogger
from mas_engagement.reasoning.profile import (
    load_profile,
    save_profile,
    update_profile_from_session,
)

_log = logging.getLogger("main")

# LOG_DIR comes from config so it follows the PyInstaller-aware _BASE_DIR.
_LOG_DIR = LOG_DIR
_MOCK_DETECTION_INTERVAL = 10.0  # seconds between mock softmax posts


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Student-engagement multi-agent session")
    p.add_argument("--duration", type=float, default=SESSION_DURATION_SEC,
                   help="Session length in seconds (default: SESSION_DURATION_SEC)")
    p.add_argument("--user-id", default=DEFAULT_USER_ID,
                   help="Profile id to load at startup / save at shutdown")
    p.add_argument("--dry-run", action="store_true",
                   help="Replace camera detection with a mock softmax generator")
    return p.parse_args()


def _initial_belief(profile: dict) -> dict:
    """Build a belief dict over HIDDEN_STATES from the profile's engagement prior."""
    prior = profile.get("engagement_prior")
    if isinstance(prior, dict):
        belief = {s: float(prior.get(s, 0.0)) for s in HIDDEN_STATES}
    else:  # list / sequence of floats aligned to HIDDEN_STATES
        belief = dict(zip(HIDDEN_STATES, (float(x) for x in prior)))
    total = sum(belief.values())
    if total > 0:
        belief = {s: v / total for s, v in belief.items()}
    return belief


def _mock_detection_loop(blackboard, logger, stop_event: threading.Event) -> None:
    """Dry-run detector: post a Dirichlet-sampled softmax every 10 seconds."""
    rng = np.random.default_rng()
    while not stop_event.is_set():
        softmax = [round(float(p), 6) for p in rng.dirichlet(np.ones(NUM_CLASSES))]
        level = int(max(range(NUM_CLASSES), key=lambda i: softmax[i]))
        confidence = float(max(softmax))
        now = time.time()
        blackboard.append_engagement(level, confidence, now, softmax=softmax)
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

    profile = load_profile(args.user_id)
    _log.info("Loaded profile %r (n_sessions=%s)",
              args.user_id, profile.get("n_sessions"))

    blackboard = SharedState(
        warmup_duration=WARMUP_DURATION,
        initial_belief=_initial_belief(profile),
    )
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
            target=_mock_detection_loop, args=(blackboard, logger, stop_event),
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

    # profile = update_profile_from_session(profile, logger.path)
    # save_profile(args.user_id, profile)
    # _log.info("Saved profile %r (n_sessions=%s)", args.user_id, profile.get("n_sessions"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
