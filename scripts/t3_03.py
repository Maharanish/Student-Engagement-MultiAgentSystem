import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from mas_engagement.config import MIN_GAP_SEC

# Pakai argumen CLI jika ada, kalau tidak scan logs/ cari sesi dengan intervensi terbanyak
def _load(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]

def _find_log():
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    best, best_n = None, 0
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith('.jsonl'):
            continue
        path = os.path.join(log_dir, fname)
        try:
            rows = _load(path)
            n = sum(1 for r in rows if r.get('event') == 'intervention_delivered')
            if n > best_n:
                best, best_n = path, n
        except Exception:
            continue
    return best

log_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'session_t303new.jsonl'
)
if log_path is None:
    print('Tidak ada log ditemukan.')
    sys.exit(1)

rows = _load(log_path)

session_start_row = next((r for r in rows if r.get('event') == 'session_start'), None)
t0 = session_start_row['ts'] if session_start_row else rows[0]['ts']

deliveries = [r for r in rows if r.get('event') == 'intervention_delivered']
silent_on  = any(r.get('event') == 'silent_mode_on' for r in rows)

print(f'Total intervensi: {len(deliveries)}')
for i, d in enumerate(deliveries):
    rel_t = d['ts'] - t0
    gap   = (d['ts'] - deliveries[i-1]['ts']) if i > 0 else 0.0
    tier  = d.get('tier', '?')
    print(f'  [{i+1}] t={rel_t:.1f}s tier={tier} gap={gap:.1f}s')

print(f'Silent mode: {silent_on}')
print(f'MIN_GAP_SEC aktif: {float(MIN_GAP_SEC)}')
