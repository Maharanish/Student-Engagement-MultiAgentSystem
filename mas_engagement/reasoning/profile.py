"""Per-user adaptive profile: persisted prior + tier response rates."""
from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Dict

from mas_engagement.config import DEFAULT_PRIOR, HIDDEN_STATES, PROFILES_DIR


_DEFAULT_RATES = {"tier_1": 0.5, "tier_2": 0.5, "tier_3": 0.5}
_BETA_ALPHA = 1.0
_BETA_BETA = 1.0


def _default_profile() -> dict:
    return {
        "engagement_prior": deepcopy(DEFAULT_PRIOR),
        "response_rates": dict(_DEFAULT_RATES),
        "n_sessions": 0,
    }


def _profile_path(user_id: str) -> Path:
    return Path(PROFILES_DIR) / f"{user_id}.json"


def load_profile(user_id: str) -> dict:
    """Read profiles/{user_id}.json or return defaults if absent/unreadable."""
    path = _profile_path(user_id)
    if not path.is_file():
        return _default_profile()
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return _default_profile()
    base = _default_profile()
    base.update(data)
    return base


def save_profile(user_id: str, profile: dict) -> None:
    """Atomic write to profiles/{user_id}.json (temp file + os.replace)."""
    path = _profile_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{user_id}.", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(profile, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise


def _parse_session(log_path: str):
    """Yield (labels_count: dict, tier_clicks: dict[str, (clicks, total)]) from a JSONL."""
    label_counts = {s: 0 for s in HIDDEN_STATES}
    tier_stats: Dict[str, list] = {t: [0, 0] for t in _DEFAULT_RATES}  # [clicks, total]

    p = Path(log_path)
    if not p.is_file():
        return label_counts, tier_stats

    with open(p, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # Engagement labels: prefer explicit hidden-state label; else map level int.
            level_label = rec.get("state") or rec.get("label")
            if level_label in label_counts:
                label_counts[level_label] += 1
            else:
                lvl = rec.get("level")
                if isinstance(lvl, int) and 0 <= lvl < len(HIDDEN_STATES):
                    label_counts[HIDDEN_STATES[lvl]] += 1

            # Responses: any record carrying a 'tier' and a 'response' is treated
            # as an outcome event. response == "ignored" or None counts as no-click.
            tier_str = rec.get("tier")
            if tier_str is not None and "response" in rec:
                tier_key = tier_str if str(tier_str).startswith("tier_") else f"tier_{tier_str}"
                if tier_key in tier_stats:
                    response = rec["response"]
                    tier_stats[tier_key][1] += 1
                    if response not in (None, "ignored"):
                        tier_stats[tier_key][0] += 1

    return label_counts, tier_stats


def update_profile_from_session(profile: dict, log_path: str) -> dict:
    """Roll session evidence into the profile (Dirichlet prior, Beta response rates)."""
    label_counts, tier_stats = _parse_session(log_path)
    n_sessions = int(profile.get("n_sessions", 0))

    old_prior = profile.get("engagement_prior") or deepcopy(DEFAULT_PRIOR)
    total_obs = sum(label_counts.values())
    if total_obs > 0:
        observed = {s: label_counts[s] / total_obs for s in HIDDEN_STATES}
        new_prior = {
            s: (old_prior.get(s, 0.0) * n_sessions + observed[s]) / (n_sessions + 1)
            for s in HIDDEN_STATES
        }
    else:
        new_prior = {s: old_prior.get(s, 0.0) for s in HIDDEN_STATES}

    # Renormalize defensively.
    s_tot = sum(new_prior.values())
    if s_tot > 0:
        new_prior = {s: v / s_tot for s, v in new_prior.items()}

    new_rates = dict(profile.get("response_rates") or _DEFAULT_RATES)
    for tier, (clicks, total) in tier_stats.items():
        if total == 0:
            continue
        new_rates[tier] = (_BETA_ALPHA + clicks) / (_BETA_ALPHA + _BETA_BETA + total)

    return {
        "engagement_prior": new_prior,
        "response_rates": new_rates,
        "n_sessions": n_sessions + 1,
    }
