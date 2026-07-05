import json, os

with open('logs/session_t201.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]

detections = [r for r in lines if r.get('event') == 'engagement_posted']
if len(detections) < 2:
    print('Tidak cukup data — perlu minimal 2 inferensi')
else:
    gaps = [detections[i]['ts'] - detections[i-1]['ts']
            for i in range(1, len(detections))]
    print(f'Jumlah inferensi     : {len(detections)}')
    print(f'Latency rata-rata    : {sum(gaps)/len(gaps):.3f} detik')
    print(f'Latency minimum      : {min(gaps):.3f} detik')
    print(f'Latency maksimum     : {max(gaps):.3f} detik')
    over = sum(1 for g in gaps if g > 2.5)
    print(f'Inferensi > 2.5 detik: {over} ({over/len(gaps)*100:.1f}%)')
    print(f'Status real-time     : {"OK" if max(gaps) < 3.0 else "PERLU DICATAT"}')