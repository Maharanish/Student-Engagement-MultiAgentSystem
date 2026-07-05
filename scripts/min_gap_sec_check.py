import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from mas_engagement.tests._sim import replay
from mas_engagement.config import MIN_GAP_SEC

with open('mas_engagement/tests/fixtures/engagement_traces.json') as f:
    traces = json.load(f)

log, snap = replay(traces['max_budget'])
interventions = [(ts, act) for ts, act in log if act != 'do_nothing']
print(f'Total intervensi: {len(interventions)}')

if len(interventions) > 1:
    all_ok = True
    for i in range(1, len(interventions)):
        gap = interventions[i][0] - interventions[i-1][0]
        status = 'OK' if gap >= MIN_GAP_SEC else 'GAGAL'
        if status == 'GAGAL':
            all_ok = False
        print(f'  Gap {i}->{i+1}: {gap:.1f}s (MIN_GAP_SEC={MIN_GAP_SEC}) [{status}]')
    print(f'Hasil: {"LULUS" if all_ok else "GAGAL"}')
elif len(interventions) == 1:
    print('Hanya 1 intervensi — cooldown tidak bisa diukur dari trace ini.')
else:
    print('Tidak ada intervensi.')