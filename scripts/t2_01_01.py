import json, os

# Label asli model — ini output langsung dari TimeSformer
# tanpa melewati likelihood matrix Orchestrator
LEVEL_LABEL = {0: 'very_low', 1: 'low', 2: 'high', 3: 'very_high'}

with open('logs/' + 'session_t2_01.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]

detections = [r for r in lines if r.get('event') == 'engagement_posted']
fallbacks  = [r for r in lines if r.get('event') == 'no_face_fallback']

total = len(detections)
avg_conf = sum(d['confidence'] for d in detections) / total if total else 0

print(f'=== T2-01: Pipeline V1 (Fokus) ===')
print(f'Total inferensi         : {total}')
print(f'MTCNN berhasil          : {total - len(fallbacks)}/{total}')
print(f'Fallback center-crop    : {len(fallbacks)}/{total}')
print(f'Confidence rata-rata    : {avg_conf:.3f}')
print()

from collections import Counter
level_counts = Counter(LEVEL_LABEL[d['level']] for d in detections)
print('Distribusi label (output langsung model):')
for label, count in sorted(level_counts.items(), key=lambda x: -x[1]):
    bar = '█' * count
    print(f'  {label:12s}: {count}x = {count/total*100:.1f}%  {bar}')

dominant = max(level_counts, key=level_counts.get)
print(f'\nLabel dominan: {dominant}')
print(f'Mock: {all(d.get("mock") == False for d in detections)}  (harus False semua)')