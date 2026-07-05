import psutil, time, json, os

pid = None
target_name = 'python'
output = []

print('Monitoring dimulai. Ctrl+C untuk berhenti.')
print(f'{'Detik':>6}  {'CPU%':>6}  {'RAM_MB':>8}  {'RAM_GB':>8}')
print('-' * 35)

try:
    for i in range(120):  # 120 x 5 detik = 10 menit
        # Cari proses python yang menjalankan main.py
        cpu_total = 0
        ram_total = 0
        count = 0
        for proc in psutil.process_iter(['name', 'cmdline', 'cpu_percent', 'memory_info']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'main.py' in cmdline:
                        cpu_total += proc.cpu_percent(interval=1)
                        ram_total += proc.memory_info().rss
                        count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if count > 0:
            ram_mb = ram_total / 1e6
            ram_gb = ram_total / 1e9
            t = i * 5
            print(f'{t:>6}s  {cpu_total:>6.1f}%  {ram_mb:>8.1f}  {ram_gb:>8.3f}')
            output.append({'t': t, 'cpu_pct': cpu_total, 'ram_mb': ram_mb})

        time.sleep(4)

except KeyboardInterrupt:
    pass

if output:
    avg_cpu = sum(r['cpu_pct'] for r in output) / len(output)
    peak_ram = max(r['ram_mb'] for r in output)
    first_ram = output[0]['ram_mb']
    last_ram = output[-1]['ram_mb']
    leak_ratio = last_ram / first_ram if first_ram > 0 else 1

    print()
    print('=== RINGKASAN ===')
    print(f'CPU rata-rata   : {avg_cpu:.1f}%')
    print(f'RAM peak        : {peak_ram:.1f} MB ({peak_ram/1024:.2f} GB)')
    print(f'RAM awal        : {first_ram:.1f} MB')
    print(f'RAM akhir       : {last_ram:.1f} MB')
    print(f'Rasio memory leak: {leak_ratio:.3f} (aman jika < 1.15)')
    print(f'Status CPU      : {"OK" if avg_cpu < 80 else "PERLU DICATAT"}')
    print(f'Status RAM      : {"OK" if peak_ram < 2048 else "PERLU DICATAT"}')
    print(f'Status leak     : {"OK" if leak_ratio < 1.15 else "PERLU DICATAT"}')

    with open('resource_profile_t4.json', 'w') as f:
        json.dump({'summary': {
            'avg_cpu_pct': avg_cpu,
            'peak_ram_mb': peak_ram,
            'leak_ratio': leak_ratio
        }, 'timeseries': output}, f, indent=2)
    print()
    print('Data tersimpan di resource_profile_t4.json')