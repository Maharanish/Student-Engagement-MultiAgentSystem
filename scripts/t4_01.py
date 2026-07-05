import json, os

with open('logs/' + 'session_t401.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]

session_end = [r for r in lines if r.get('event') == 'session_end']
detections  = [r for r in lines if r.get('event') == 'engagement_posted']

print(f'=== T4-01: Ketahanan Sumber Video ===')
print(f'Sistem tidak crash   : {len(session_end) > 0}  (harus True)')
print(f'Total inferensi      : {len(detections)}')
print(f'Session end: {len(session_end) > 0}')