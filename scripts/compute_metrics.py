"""Ekstrak metrik dari session log JSONL dan hasilkan CSV + belief trajectory plot.

Penggunaan:
    python scripts/compute_metrics.py                          # log terbaru di logs/
    python scripts/compute_metrics.py logs/session_xxx.jsonl  # log spesifik
    python scripts/compute_metrics.py logs/session_xxx.jsonl -o results/out.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

STATES = ['actively_dis', 'drifting', 'engaged', 'frustrated']
STATE_COLORS = {
    'actively_dis': '#d62728',
    'drifting':     '#ff7f0e',
    'engaged':      '#2ca02c',
    'frustrated':   '#9467bd',
}
STATE_LABELS = {
    'actively_dis': 'Actively Disengaged',
    'drifting':     'Drifting',
    'engaged':      'Engaged',
    'frustrated':   'Frustrated',
}
TIER_COLORS = {'1': '#1f77b4', '2': '#ff7f0e', '3': '#d62728'}


def _find_latest_log() -> Path:
    log_dir = Path(__file__).resolve().parent.parent / 'logs'
    candidates = sorted(log_dir.glob('session_*.jsonl'), reverse=True)
    if not candidates:
        raise FileNotFoundError(f'Tidak ada session log di {log_dir}')
    return candidates[0]


def load_events(path: Path) -> list:
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_metrics(events: list) -> tuple:
    """Kembalikan (rows, interventions, session_info)."""
    start_ev = next((e for e in events if e.get('event') == 'session_start'), None)
    t0 = start_ev['ts'] if start_ev else events[0]['ts']
    session_info = {
        'user_id':  start_ev.get('user_id', 'unknown') if start_ev else 'unknown',
        'duration': start_ev.get('duration', None) if start_ev else None,
        'dry_run':  start_ev.get('dry_run', True) if start_ev else True,
    }

    rows = []
    for e in events:
        if e.get('event') != 'decision':
            continue
        b = e.get('belief', {})
        u = e.get('utilities', {})
        rows.append({
            't':             round(e['ts'] - t0, 3),
            'actively_dis':  round(b.get('actively_dis', 0), 6),
            'drifting':      round(b.get('drifting', 0), 6),
            'engaged':       round(b.get('engaged', 0), 6),
            'frustrated':    round(b.get('frustrated', 0), 6),
            'action':        e.get('action', ''),
            'n_evidence':    e.get('n_engagement_evidence', 0),
            'n_response':    e.get('n_response_evidence', 0),
            'eu_tier1':      round(u.get('tier_1', float('-inf')), 4) if math.isfinite(u.get('tier_1', float('-inf'))) else 'inf',
            'eu_tier2':      round(u.get('tier_2', float('-inf')), 4) if math.isfinite(u.get('tier_2', float('-inf'))) else 'inf',
            'eu_tier3':      round(u.get('tier_3', float('-inf')), 4) if math.isfinite(u.get('tier_3', float('-inf'))) else 'inf',
            'eu_do_nothing': round(u.get('do_nothing', 0), 4),
        })

    interventions = [
        {
            't':    round(e['ts'] - t0, 3),
            'tier': e.get('tier', '?'),
            'response': e.get('response', ''),
        }
        for e in events if e.get('event') == 'intervention_delivered'
    ]

    silent_t = next(
        (round(e['ts'] - t0, 3) for e in events if e.get('event') == 'silent_mode_on'),
        None
    )

    return rows, interventions, session_info, silent_t


def save_csv(rows: list, out_path: Path) -> None:
    if not rows:
        print('Tidak ada decision events — CSV tidak dibuat.')
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'CSV disimpan: {out_path}')


def plot_belief_trajectory(rows: list, interventions: list, silent_t,
                           session_info: dict, out_path: Path) -> None:
    if not rows:
        print('Tidak ada data untuk diplot.')
        return

    times = [r['t'] for r in rows]
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor('white')

    for state in STATES:
        ax.plot(times, [r[state] for r in rows],
                color=STATE_COLORS[state], linewidth=1.8,
                label=STATE_LABELS[state])

    # Marker intervensi
    for iv in interventions:
        color = TIER_COLORS.get(str(iv['tier']), '#555555')
        ax.axvline(iv['t'], color=color, linestyle='--', linewidth=1.2, alpha=0.7)
        ax.text(iv['t'] + 1, 0.97, f"T{iv['tier']}", fontsize=7,
                color=color, va='top', ha='left')

    # Garis silent mode
    if silent_t is not None:
        ax.axvline(silent_t, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
        ax.text(silent_t + 1, 0.88, 'silent', fontsize=7,
                color='black', va='top', ha='left')

    ax.set_ylim(0, 1)
    ax.set_xlabel('Waktu Sesi (detik)', fontsize=11)
    ax.set_ylabel('Belief P(state)', fontsize=11)
    ax.grid(axis='y', linestyle=':', alpha=0.4)

    # Legend
    handles = [
        plt.Line2D([0], [0], color=STATE_COLORS[s], linewidth=2,
                   label=STATE_LABELS[s])
        for s in STATES
    ]
    for tier, color in TIER_COLORS.items():
        handles.append(mpatches.Patch(color=color, alpha=0.6,
                                      label=f'Intervensi Tier-{tier}'))
    ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)

    uid = session_info.get('user_id', '')
    dry = ' [dry_run]' if session_info.get('dry_run') else ''
    ax.set_title(f'Belief Trajectory — {uid}{dry}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Plot disimpan : {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Compute metrics & belief trajectory dari session log')
    parser.add_argument('log', nargs='?', help='Path ke JSONL log (default: log terbaru di logs/)')
    parser.add_argument('-o', '--output', help='Path CSV output (default: results/<logname>.csv)')
    args = parser.parse_args()

    log_path = Path(args.log) if args.log else _find_latest_log()
    if not log_path.exists():
        print(f'ERROR: File tidak ditemukan: {log_path}', file=sys.stderr)
        sys.exit(1)

    results_dir = Path(__file__).resolve().parent.parent / 'results'
    csv_path = Path(args.output) if args.output else results_dir / (log_path.stem + '.csv')
    png_path = csv_path.with_suffix('.csv.belief_trajectory.png')

    print(f'Membaca: {log_path}')
    events = load_events(log_path)
    rows, interventions, session_info, silent_t = extract_metrics(events)

    print(f'Decision ticks : {len(rows)}')
    print(f'Intervensi     : {len(interventions)}')
    print(f'Silent mode    : {"t=" + str(silent_t) + "s" if silent_t else "tidak aktif"}')

    save_csv(rows, csv_path)
    plot_belief_trajectory(rows, interventions, silent_t, session_info, png_path)


if __name__ == '__main__':
    main()
