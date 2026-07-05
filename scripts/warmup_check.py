import json, os

with open('logs/' + 'session_t1.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]

session_start = next(r['ts'] for r in lines if r.get('event') == 'session_start')
early = [r for r in lines
         if r.get('event') == 'intervention_delivered'
         and r['ts'] - session_start < 600]

print(f'Intervensi sebelum t=600: {len(early)} (harus 0)')