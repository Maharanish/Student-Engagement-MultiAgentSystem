import json

with open('results/resource_profile_t4.json') as f:
    data = json.load(f)

s = data['summary']
ts = data['timeseries']

print('=== RINGKASAN T4-02: CPU & RAM Profiling ===')
print(f'CPU rata-rata    : {s["avg_cpu_pct"]:.1f}%')
print(f'RAM peak         : {s["peak_ram_mb"]:.1f} MB ({s["peak_ram_mb"]/1024:.2f} GB)')
print(f'Rasio memory leak: {s["leak_ratio"]:.3f} (aman jika < 1.15)')
print(f'Status RAM       : {"OK" if s["peak_ram_mb"] < 2048 else "PERLU DICATAT"}')
print(f'Status leak      : {"OK" if s["leak_ratio"] < 1.15 else "PERLU DICATAT"}')
print()
print(f'RAM awal         : {ts[0]["ram_mb"]:.1f} MB')
print(f'RAM akhir        : {ts[-1]["ram_mb"]:.1f} MB')
print(f'Durasi monitoring: {ts[-1]["t"]} detik ({len(ts)} titik data)')
print()
print(f'{"Detik":>6}  {"CPU%":>6}  {"RAM_MB":>8}')
print('-' * 28)
for r in ts:
    print(f'{r["t"]:>6}s  {r["cpu_pct"]:>6.1f}%  {r["ram_mb"]:>8.1f}')