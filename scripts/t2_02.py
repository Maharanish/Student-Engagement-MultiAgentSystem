import json, os
from collections import Counter

# Label asli model — output langsung TimeSformer sebelum diproses Orchestrator
LEVEL_LABEL = {0: 'very_low', 1: 'low', 2: 'high', 3: 'very_high'}

def load_detections(path):
    with open(path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    return [r for r in lines if r.get('event') == 'engagement_posted']

v1_file = 'logs/session_t2_01.jsonl'
v3_file = 'logs/session_t2_02.jsonl'

def ringkasan(detections, label):
    total = len(detections)
    if total == 0:
        return
    counts = Counter(LEVEL_LABEL[d['level']] for d in detections)
    dominant = max(counts, key=counts.get)
    confs = [d['confidence'] for d in detections]
    avg_conf = sum(confs) / total
    std_conf = (sum((c - avg_conf) ** 2 for c in confs) / total) ** 0.5
    bins = {'<0.40': 0, '0.40-0.60': 0, '0.60-0.80': 0, '>0.80': 0}
    for c in confs:
        if c < 0.40:   bins['<0.40'] += 1
        elif c < 0.60: bins['0.40-0.60'] += 1
        elif c < 0.80: bins['0.60-0.80'] += 1
        else:          bins['>0.80'] += 1
    print(f'{label}:')
    print(f'  Label dominan    : {dominant} ({counts[dominant]}/{total} = {counts[dominant]/total*100:.1f}%)')
    print(f'  Confidence rata  : {avg_conf:.3f}  std={std_conf:.3f}')
    print(f'  Distribusi conf  : {dict(bins)}')
    print(f'  Distribusi label : {dict(counts)}')

ringkasan(load_detections(v1_file), 'V1 (Fokus)')
print()
ringkasan(load_detections(v3_file), 'V3 (Tidak Fokus)')