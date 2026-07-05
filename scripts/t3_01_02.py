import json, os
with open('logs/' + 'session_t301new.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
deliveries = [r for r in lines if r.get('event') == 'intervention_delivered']
for d in deliveries:
    print(f'tier={d.get("tier")}  latency_ms={d.get("latency_ms")}  e2e_latency_ms={d.get("e2e_latency_ms")}')
