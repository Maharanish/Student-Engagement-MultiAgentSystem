import json, os

with open('logs/' + 'session_t301new.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]

session_start = next(r['ts'] for r in lines if r.get('event') == 'session_start')
session_end   = next((r['ts'] for r in lines if r.get('event') == 'session_end'), None)
duration      = (session_end - session_start) if session_end else 0

# Response ada di dalam event intervention_delivered, field response_action
deliveries = [r for r in lines if r.get('event') == 'intervention_delivered']

clicked  = sum(1 for r in deliveries if r.get('response_action') == 'clicked')
ignored  = sum(1 for r in deliveries if r.get('response_action') == 'ignored')

print(f'=== Perilaku Adaptif MAS (T3-01) ===')
print(f'Durasi sesi              : {duration/60:.1f} menit')
print(f'Jumlah intervensi        : {len(deliveries)}')
if duration > 0:
    print(f'Intervensi per menit     : {len(deliveries)/(duration/60):.2f}')
if deliveries:
    print(f'Diklik                   : {clicked} ({clicked/len(deliveries)*100:.1f}%)')
    print(f'Diabaikan                : {ignored} ({ignored/len(deliveries)*100:.1f}%)')
    avg_latency = sum(r.get('latency_ms', 0) for r in deliveries) / len(deliveries)
    print(f'Latency respons rata-rata: {avg_latency:.0f} ms')

# Lag deteksi pertama ke intervensi pertama
detections = [r for r in lines if r.get('event') == 'engagement_posted']
if detections and deliveries:
    first_det_ts    = detections[0]['ts']
    first_interv_ts = deliveries[0]['ts']
    lag = first_interv_ts - first_det_ts
    print(f'Lag deteksi → intervensi : {lag:.1f} detik')
    print(f'  (deteksi pertama t={first_det_ts - session_start:.0f}s,')
    print(f'   intervensi pertama t={first_interv_ts - session_start:.0f}s)')

# Gap antar intervensi
if len(deliveries) > 1:
    print(f'\nGap antar intervensi:')
    for i in range(1, len(deliveries)):
        gap = deliveries[i]['ts'] - deliveries[i-1]['ts']
        status = 'OK' if gap >= 30 else 'TERLALU CEPAT'
        print(f'  Intervensi {i}→{i+1}: {gap:.1f}s [{status}]')

# Verifikasi tidak ada intervensi sebelum warmup
early = [r for r in deliveries if r['ts'] - session_start < 60]
print(f'\nIntervensi sebelum t=60  : {len(early)} (harus 0)')