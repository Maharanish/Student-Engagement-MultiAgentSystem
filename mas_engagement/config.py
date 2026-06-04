import sys
from pathlib import Path

import numpy as np

_PKG_ROOT = Path(__file__).resolve().parent

# When frozen with PyInstaller --onefile, Path(__file__) lives inside the temp
# extraction dir (sys._MEIPASS), so paths derived from _PKG_ROOT would not see
# the data/profiles/logs folders the user ships next to EduAgent.exe. Resolve
# the base dir from sys.executable in that case; otherwise fall back to repo
# root (the normal `python main.py` development path).
if getattr(sys, "frozen", False):
    _BASE_DIR = Path(sys.executable).resolve().parent
else:
    _BASE_DIR = _PKG_ROOT.parent

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX: int = 0          # hint index for OBS Virtual Camera; agent scans 0-2 on failure
CAMERA_SCAN_RANGE: int = 3     # number of indices to try when hint fails

# ── Detection ─────────────────────────────────────────────────────────────────
CAPTURE_FPS: int = 15
INFERENCE_INTERVAL: float = 2.0
NUM_FRAMES: int = 8
FRAME_SIZE: int = 224
NUM_CLASSES: int = 4
MTCNN_MARGIN_PX: int = 20

# ── Intervention ──────────────────────────────────────────────────────────────
INTERVENTION_POLL_HZ: float = 5.0
NO_REPEAT_WINDOW: int = 3      # IDs in this window are excluded from the next pick

# ── Delivery ──────────────────────────────────────────────────────────────────
DELIVERY_POLL_HZ: float = 10.0
TOAST_TIMEOUT_SEC: int = 8       # Tier-1 auto-dismiss timeout (seconds)
TIER2_TIMEOUT_SEC: int = 12      # Tier-2 auto-dismiss timeout (seconds)
TIER3_OPTIONS = ["Got it", "Need help", "Dismiss"]

# ── Orchestrator ──────────────────────────────────────────────────────────────
ORCHESTRATOR_POLL_HZ: float = 1.0
WARMUP_DURATION: float = 30.0    # REVISED: must match WARMUP_SEC (teacher validation)
CONFIDENCE_MIN: float = 0.4       # records below this are excluded from the signal
ENGAGEMENT_WINDOW_SEC: float = 2
DISENGAGE_RATIO: float = 0.5      # fraction of window that must be disengaged to trigger
MIN_TIER_GAP_SEC: float = 60.0    # minimum seconds between successive tier decisions
MAX_INTERVENTIONS: int = 4        # session cap; reaching it triggers silent mode
TIER3_PERSISTENT      = True   # widget stays until clicked; silent mode at session end only
TIER3_BREAK_NUDGE_SEC = 5      # seconds to show wellness nudge after "break" click

# "Istirahat sebentar" break flow: after the wellness nudge closes the system
# pauses interventions for BREAK_DURATION_SEC, then plays a chime and shows a
# persistent return widget. The session resumes (with belief intact) when the
# student clicks "Siap". See docs/CHANGES_post_validation.md § 9.5.
BREAK_DURATION_SEC: float = 120.0   # 2-minute break window
BREAK_CHIME_FREQ_HZ: int  = 880     # A5; gentle but noticeable
BREAK_CHIME_MS: int       = 500     # half-second chime

# Per-tier notification chimes. Tier-1 / Tier-2 are deliberately softer
# (lower frequency, shorter duration) than Tier-3 so the audio cue itself
# carries the tier's seriousness.
TIER1_CHIME_FREQ_HZ: int = 660      # E5; soft "ping"
TIER1_CHIME_MS:      int = 120
TIER2_CHIME_FREQ_HZ: int = 760      # G5; a touch more attention-grabbing
TIER2_CHIME_MS:      int = 180
# Tier-3 reuses the same chime as the return-from-break (BREAK_CHIME_*).

# ── Session ───────────────────────────────────────────────────────────────────
DEFAULT_USER_ID: str = "default_user"
SESSION_DURATION_SEC: float = 600.0   # default full-session length

# ── Bayesian Orchestrator (Phase-2 utility-based decision-making) ─────────────
HIDDEN_STATES = ["actively_dis", "drifting", "engaged", "frustrated"]
ACTIONS = ["tier_1", "tier_2", "tier_3", "do_nothing"]
DEFAULT_PRIOR = {
    "actively_dis": 0.10,
    "drifting": 0.20,
    "engaged": 0.60,
    "frustrated": 0.10,
}

# P(obs_class c | hidden_state s); rows are HIDDEN_STATES, cols are NUM_CLASSES
# detection classes (DAiSEE-style: 0=very low … 3=very high engagement).
LIKELIHOOD_DETECTION = np.array(
    [
        [0.70, 0.20, 0.07, 0.03],  # actively_dis
        [0.20, 0.55, 0.20, 0.05],  # drifting
        [0.05, 0.15, 0.50, 0.30],  # engaged
        [0.40, 0.30, 0.20, 0.10],  # frustrated
    ],
    dtype=np.float64,
)

# P(clicked | hidden_state). P(ignored | s) is derived as 1 - this.
# Frustrated users almost never click — drives frustrated mass on ignores.
LIKELIHOOD_CLICK = {
    "actively_dis": 0.50,
    "drifting": 0.55,
    "engaged": 0.85,
    "frustrated": 0.03,
}

# UTILITY_MATRIX[action_idx, state_idx]; ACTIONS × HIDDEN_STATES.
# Tier_1 best at drifting (gentle nudge); tier_2 at moderate actively_dis;
# tier_3 reserved for *sustained severe* actively_dis. do_nothing wins at
# engaged (no action needed) AND at high frustrated (give space).
UTILITY_MATRIX = np.array(
    [
        [3.5, 4.0, -1.0, -3.0],  # tier_1 — gentle nudge; best at drifting
        [4.0, 1.5, -2.0, -0.5],  # tier_2 — best at moderate actively_dis
        [3.0, 0.5, -3.0,  2.0],  # tier_3 — best when actively_dis AND frustrated
        [-2.0, -0.5, 2.0,  4.0], # do_nothing — best at engaged OR pure frustrated
    ],
    dtype=np.float64,
)

BELIEF_DECAY_RATE: float = 0.001  # λ per second; gentle pull toward DEFAULT_PRIOR

# Fatigue penalty for intervention actions: sigmoid in #interventions so far.
FATIGUE_SCALE: float = 2.0
FATIGUE_MIDPOINT: float = 3.0
FATIGUE_STEEPNESS: float = 1.5

# Cooldown / pacing for intervention actions.
MIN_GAP_SEC: float = 60.0  # REVISED: longer recovery window (teacher validation)
WARMUP_SEC: float = 0.0   # REVISED: first 10 min naturally engaging (teacher validation)

# Profile storage (next to the .exe when frozen; repo-root ./profiles otherwise).
PROFILES_DIR = _BASE_DIR / "profiles"

# ── Paths ─────────────────────────────────────────────────────────────────────
# All data paths resolve relative to _BASE_DIR so that the frozen release layout
#   EduAgent_Release/
#     EduAgent.exe
#     mas_engagement/data/{messages.json, E3_WCE_lr1e5.pth}
#     profiles/
#     logs/
# works without any further wiring.
MODEL_CONFIG_DIR = _BASE_DIR / "mas_engagement" / "data" / "config"
# Best checkpoint: E3_WCE_lr1e5 (accuracy=0.5458, QWK=0.1253)
MODEL_WEIGHTS_PATH = _BASE_DIR / "mas_engagement" / "data" / "E3_WCE_lr1e5.pth"
MESSAGES_PATH = _BASE_DIR / "mas_engagement" / "data" / "messages.json"
LOG_DIR = _BASE_DIR / "logs"


# ── Optional user_config.json runtime overrides ───────────────────────────────
# A non-technical user can drop a `user_config.json` next to the .exe (in dev
# mode: at repo root, both controlled by _BASE_DIR) to adjust a small whitelist
# of tunables without touching source. The override runs at *module load* time
# so every downstream `from mas_engagement.config import X` sees the patched
# value — applying overrides later (from main.py, for instance) would not
# affect agents that already snapshotted the default into their own namespace.
#
# main.py reads USER_CONFIG_OVERRIDES / USER_CONFIG_WARNINGS at startup and
# emits a `config_override` log record so the audit trail is preserved.

import json as _json

USER_CONFIG_PATH = _BASE_DIR / "user_config.json"
USER_CONFIG_OVERRIDES: dict = {}
USER_CONFIG_WARNINGS: list = []

_OVERRIDABLE = {
    "CONFIDENCE_MIN":    float,
    "MIN_GAP_SEC":       float,
    "MAX_INTERVENTIONS": int,
    # WARMUP_SEC and WARMUP_DURATION are handled specially below — they must
    # stay equal because they are read by two different gates (blackboard's
    # warmup_until and the utility-side _cooldown_blocked).
}


def _coerce(key, value, want):
    try:
        coerced = want(value)
    except (TypeError, ValueError) as exc:
        USER_CONFIG_WARNINGS.append(
            f"{key!r}: cannot convert {value!r} to {want.__name__} ({exc})"
        )
        return None
    return coerced


def _apply_user_overrides() -> None:
    if not USER_CONFIG_PATH.is_file():
        return
    try:
        with open(USER_CONFIG_PATH, encoding="utf-8") as fh:
            data = _json.load(fh)
    except (OSError, _json.JSONDecodeError) as exc:
        USER_CONFIG_WARNINGS.append(
            f"could not read {USER_CONFIG_PATH.name}: {exc}"
        )
        return
    if not isinstance(data, dict):
        USER_CONFIG_WARNINGS.append(
            f"{USER_CONFIG_PATH.name}: top-level must be an object/dict"
        )
        return

    # ── Warmup: WARMUP_SEC and WARMUP_DURATION are kept in lock-step ─────────
    warmup_value = None
    if "WARMUP_SEC" in data and "WARMUP_DURATION" in data:
        if data["WARMUP_SEC"] != data["WARMUP_DURATION"]:
            USER_CONFIG_WARNINGS.append(
                "WARMUP_SEC and WARMUP_DURATION disagree; using WARMUP_SEC."
            )
        warmup_value = _coerce("WARMUP_SEC", data["WARMUP_SEC"], float)
    elif "WARMUP_SEC" in data:
        warmup_value = _coerce("WARMUP_SEC", data["WARMUP_SEC"], float)
    elif "WARMUP_DURATION" in data:
        warmup_value = _coerce("WARMUP_DURATION", data["WARMUP_DURATION"], float)
    if warmup_value is not None:
        globals()["WARMUP_SEC"] = warmup_value
        globals()["WARMUP_DURATION"] = warmup_value
        USER_CONFIG_OVERRIDES["WARMUP_SEC"] = warmup_value
        USER_CONFIG_OVERRIDES["WARMUP_DURATION"] = warmup_value

    # ── Scalar overrides ─────────────────────────────────────────────────────
    for key, want in _OVERRIDABLE.items():
        if key not in data:
            continue
        coerced = _coerce(key, data[key], want)
        if coerced is None:
            continue
        globals()[key] = coerced
        USER_CONFIG_OVERRIDES[key] = coerced

    # ── Surface unknown keys (ignore comment / metadata keys starting with _) ─
    known = set(_OVERRIDABLE) | {"WARMUP_SEC", "WARMUP_DURATION"}
    for key in data:
        if key.startswith("_") or key in known:
            continue
        USER_CONFIG_WARNINGS.append(
            f"unknown key {key!r} in {USER_CONFIG_PATH.name} ignored"
        )


_apply_user_overrides()
del _apply_user_overrides, _coerce, _json
