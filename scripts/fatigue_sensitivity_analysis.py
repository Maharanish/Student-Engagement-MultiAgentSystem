"""Fatigue-cost sensitivity sweep for the thesis evaluation chapter.

Runs the full multi-agent pipeline in dry-run mode (mocked detection, no
camera) once per FATIGUE_SCALE value, then compares how many interventions
fire and how they are distributed across tiers. This shows how load-bearing
the fatigue penalty is: a higher FATIGUE_SCALE should suppress interventions
(especially the later, more aggressive ones).

Method:
  * For each fatigue value we write a temporary ``user_config.json`` next to
    the repo root (the same whitelist-based override main.py already reads at
    startup — see mas_engagement.config). FATIGUE_SCALE is set to the swept
    value; WARMUP_SEC is lowered (``--warmup``, default 0) so interventions can
    actually occur within the short --duration window (the production default
    is a 600 s warmup that would suppress everything in a 300 s run).
  * We launch ``python main.py --dry-run --duration <d>`` as a subprocess so it
    re-imports config and picks up the override cleanly (config applies
    overrides at module-load time only).
  * The new ``logs/session_*.jsonl`` produced by that run is parsed for
    ``delivery / intervention_delivered`` events; we count them per tier.

The original ``user_config.json`` (if any) is always restored in a finally
block, so the repo's config is returned to its original state on exit — even
on Ctrl-C or error.

Usage:
  python scripts/fatigue_sensitivity_analysis.py
  python scripts/fatigue_sensitivity_analysis.py --duration 300 --warmup 0
  python scripts/fatigue_sensitivity_analysis.py --fatigue-values 0.1 0.3 0.5
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mas_engagement.config import LOG_DIR, USER_CONFIG_PATH  # noqa: E402

_DEFAULT_FATIGUE_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]
_TIERS = ("1", "2", "3")
_RESULTS_CSV = Path(LOG_DIR) / "sensitivity_analysis_results.csv"
_MAIN_PY = _REPO_ROOT / "main.py"


# ── Log parsing ───────────────────────────────────────────────────────────────

def _existing_logs() -> set[Path]:
    """Snapshot the current session log files so we can spot the new one."""
    log_dir = Path(LOG_DIR)
    if not log_dir.is_dir():
        return set()
    return set(log_dir.glob("session_*.jsonl"))


def _newest_new_log(before: set[Path]) -> Optional[Path]:
    """Return the log file created since `before` (newest if several)."""
    new = _existing_logs() - before
    if not new:
        return None
    return max(new, key=lambda p: p.stat().st_mtime)


def _count_interventions_per_tier(log_path: Path) -> Dict[str, int]:
    """Count `delivery / intervention_delivered` events per tier in a JSONL log."""
    counts = {t: 0 for t in _TIERS}
    with open(log_path, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if (
                rec.get("agent") == "delivery"
                and rec.get("event") == "intervention_delivered"
            ):
                tier = str(rec.get("tier"))
                if tier in counts:
                    counts[tier] += 1
    return counts


# ── Config override (write / restore user_config.json) ────────────────────────

def _write_user_config(fatigue_scale: float, warmup: float,
                        min_gap: Optional[float]) -> None:
    payload: Dict[str, Any] = {
        "_comment": "Temporary override written by fatigue_sensitivity_analysis.py",
        "FATIGUE_SCALE": fatigue_scale,
        "WARMUP_SEC": warmup,
        "WARMUP_DURATION": warmup,
    }
    if min_gap is not None:
        payload["MIN_GAP_SEC"] = min_gap
    with open(USER_CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


# ── Single dry-run ────────────────────────────────────────────────────────────

def _run_one(fatigue_scale: float, duration: float, warmup: float,
             min_gap: Optional[float]) -> Dict[str, Any]:
    """Run one dry-run session at `fatigue_scale` and return its tier counts."""
    _write_user_config(fatigue_scale, warmup, min_gap)

    before = _existing_logs()
    print(f"  → running main.py --dry-run --duration {duration:.0f} "
          f"(FATIGUE_SCALE={fatigue_scale}) …", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(_MAIN_PY), "--dry-run",
         "--duration", str(duration), "--user-id", f"fatigue_{fatigue_scale}"],
        cwd=str(_REPO_ROOT),
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"    warning: main.py exited with code {proc.returncode}",
              file=sys.stderr)

    log_path = _newest_new_log(before)
    if log_path is None:
        print("    warning: no new session log found for this run",
              file=sys.stderr)
        counts = {t: 0 for t in _TIERS}
    else:
        counts = _count_interventions_per_tier(log_path)

    total = sum(counts.values())
    print(f"    done in {elapsed:.0f}s — interventions: total={total} "
          f"tier1={counts['1']} tier2={counts['2']} tier3={counts['3']}",
          flush=True)

    return {
        "fatigue_scale": fatigue_scale,
        "total_interventions": total,
        **{f"tier_{t}": counts[t] for t in _TIERS},
        "log_file": log_path.name if log_path else "",
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_table(rows: List[Dict[str, Any]]) -> None:
    width = 72
    print()
    print("=" * width)
    print("  Fatigue-cost sensitivity — interventions per tier")
    print("=" * width)
    header = (f"{'FATIGUE_SCALE':>13} {'total':>6} "
              f"{'tier_1':>7} {'tier_2':>7} {'tier_3':>7}   distribution")
    print(header)
    print("-" * width)
    for r in rows:
        total = r["total_interventions"]
        if total:
            dist = " / ".join(
                f"{100.0 * r[f'tier_{t}'] / total:.0f}%" for t in _TIERS)
        else:
            dist = "—"
        print(f"{r['fatigue_scale']:>13} {total:>6} "
              f"{r['tier_1']:>7} {r['tier_2']:>7} {r['tier_3']:>7}   {dist}")
    print("=" * width)


def _write_csv(rows: List[Dict[str, Any]]) -> None:
    _RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["fatigue_scale", "total_interventions",
              "tier_1", "tier_2", "tier_3", "log_file"]
    with open(_RESULTS_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {_RESULTS_CSV}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FATIGUE_SCALE sensitivity sweep")
    p.add_argument("--duration", type=float, default=300.0,
                   help="Dry-run length per fatigue value in seconds (default 300)")
    p.add_argument("--warmup", type=float, default=0.0,
                   help="WARMUP_SEC override so interventions can fire within "
                        "--duration (default 0)")
    p.add_argument("--min-gap", type=float, default=None,
                   help="Optional MIN_GAP_SEC override (default: leave at config "
                        "value)")
    p.add_argument("--fatigue-values", type=float, nargs="+",
                   default=_DEFAULT_FATIGUE_VALUES,
                   help="FATIGUE_SCALE values to sweep "
                        "(default 0.1 0.2 0.3 0.4 0.5)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # Preserve any pre-existing user_config.json so we can restore it verbatim.
    original_existed = USER_CONFIG_PATH.is_file()
    original_bytes = (
        USER_CONFIG_PATH.read_bytes() if original_existed else None
    )

    rows: List[Dict[str, Any]] = []
    try:
        for fatigue_scale in args.fatigue_values:
            print(f"[fatigue_scale={fatigue_scale}]")
            rows.append(_run_one(
                fatigue_scale, args.duration, args.warmup, args.min_gap))
    finally:
        # ── Always restore config to its original state ──────────────────────
        if original_existed:
            USER_CONFIG_PATH.write_bytes(original_bytes)  # type: ignore[arg-type]
            print(f"restored original {USER_CONFIG_PATH.name}")
        elif USER_CONFIG_PATH.is_file():
            USER_CONFIG_PATH.unlink()
            print(f"removed temporary {USER_CONFIG_PATH.name}")

    if rows:
        _print_table(rows)
        _write_csv(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
