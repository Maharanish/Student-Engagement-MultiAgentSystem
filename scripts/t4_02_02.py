import psutil
print('Jumlah CPU logical (termasuk HT):', psutil.cpu_count(logical=True))
print('Jumlah CPU physical              :', psutil.cpu_count(logical=False))

import json
with open('results/resource_profile_t4.json') as f:
    data = json.load(f)

s = data['summary']
n_cores = psutil.cpu_count(logical=True)
avg_per_core = s['avg_cpu_pct'] / n_cores
peak_cpu = max(r['cpu_pct'] for r in data['timeseries'])
peak_per_core = peak_cpu / n_cores

print()
print(f'CPU rata-rata (aggregate) : {s["avg_cpu_pct"]:.1f}%')
print(f'CPU rata-rata (per core)  : {avg_per_core:.1f}% dari 1 core')
print(f'CPU peak (aggregate)      : {peak_cpu:.1f}%')
print(f'CPU peak (per core)       : {peak_per_core:.1f}% dari 1 core')
print()
print(f'Interpretasi: sistem menggunakan rata-rata {avg_per_core:.1f}% dari kapasitas total CPU')
status = 'OK' if avg_per_core < 80 else 'PERLU DICATAT'
print(f'Status (per core)         : {status}')