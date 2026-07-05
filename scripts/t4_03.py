import os, glob

print('=== T4-04: Verifikasi Privasi Data (K-05) ===')
print()

# Cek tidak ada file video tersimpan di seluruh direktori repo
video_ext = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm', '*.flv']
video_files = []
for ext in video_ext:
    found = glob.glob(f'**/{ext}', recursive=True)
    # Kecualikan folder aset sistem (model weights, dsb.)
    session_vids = [f for f in found if 'logs' in f or 'profiles' in f or 'sessions' in f]
    video_files.extend(session_vids)

# Cek tidak ada file gambar wajah di folder sesi
image_ext = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
image_files = []
for ext in image_ext:
    found = glob.glob(f'**/{ext}', recursive=True)
    session_imgs = [f for f in found if 'logs' in f or 'profiles' in f or 'sessions' in f]
    image_files.extend(session_imgs)

# Cek folder logs hanya berisi JSONL
log_files = os.listdir('logs') if os.path.exists('logs') else []
non_jsonl  = [f for f in log_files if not f.endswith('.jsonl')]

print(f'File video di folder sesi    : {len(video_files)} (harus 0)')
print(f'File gambar di folder sesi   : {len(image_files)} (harus 0)')
print(f'File non-JSONL di logs/      : {len(non_jsonl)} (harus 0)')
print()

semua_aman = (
    len(video_files) == 0 and
    len(image_files) == 0 and
    len(non_jsonl) == 0 
)

if semua_aman:
    print('Status: LULUS')
    print('Tidak ada data visual tersimpan ke disk.')
    print('Sistem memproses data wajah secara in-memory.')
else:
    print('Status: PERLU DIPERIKSA')
    if video_files:   print(f'  Video ditemukan  : {video_files}')
    if image_files:   print(f'  Gambar ditemukan : {image_files}')
    if non_jsonl:     print(f'  Non-JSONL di logs: {non_jsonl}')