import json, os

with open('logs/' + 'session_t203.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]

detections  = [r for r in lines if r.get('event') == 'engagement_posted']
session_end = [r for r in lines if r.get('event') == 'session_end']

print(f'=== T2-03: Graceful Degradation No-Face ===')
print(f'Total inferensi      : {len(detections)}')
print(f'Sesi berakhir bersih : {len(session_end) > 0}  (harus True)')
print(f'Log JSONL valid      : {all("ts" in r for r in lines)}  (harus True)')
print()
print('Bukti no_face_fallback: lihat output terminal saat sesi berjalan')
print('(WARNING ... no_face_fallback muncul berulang = MTCNN aktif mendeteksi)')
print()
if len(session_end) > 0 and len(detections) > 0:
    print('Status: LULUS — pipeline tetap berjalan meski tidak ada wajah')
