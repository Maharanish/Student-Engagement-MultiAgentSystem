import argparse
import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_json(path: str) -> tuple[list, dict]:
    """Baca timeseries dan summary dari file JSON.

    Format yang didukung:
      - {"summary": {...}, "timeseries": [...]}   (output generate_t4_chart lama)
      - {"timeseries": [...]}                      (summary dihitung ulang)
      - [...]                                      (list timeseries langsung)
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        timeseries = data
        summary = None
    elif 'timeseries' in data:
        timeseries = data['timeseries']
        summary = data.get('summary')
    else:
        print(f'ERROR: Format JSON tidak dikenali di {path}', file=sys.stderr)
        sys.exit(1)

    # Hapus titik dengan cpu=0 dan ram sangat kecil (proses sudah exit)
    timeseries = [r for r in timeseries if r['cpu_pct'] > 0 and r['ram_mb'] > 10]

    if not timeseries:
        print('ERROR: Tidak ada data valid di timeseries', file=sys.stderr)
        sys.exit(1)

    return timeseries, summary


def compute_summary(timeseries: list) -> dict:
    cpus    = [r['cpu_pct'] for r in timeseries]
    rams    = [r['ram_mb'] for r in timeseries]
    avg_cpu = sum(cpus) / len(cpus)
    peak_cpu = max(cpus)
    peak_ram = max(rams)

    # Deteksi n_cores dari JSON kalau ada, default 8
    n_cores = timeseries[0].get('n_cores', 8)

    return {
        'avg_cpu_pct_aggregate': round(avg_cpu, 1),
        'avg_cpu_pct_per_core': round(avg_cpu / n_cores, 1),
        'peak_cpu_pct_aggregate': round(peak_cpu, 1),
        'peak_cpu_pct_per_core': round(peak_cpu / n_cores, 1),
        'peak_ram_mb': round(peak_ram, 1),
        'peak_ram_gb': round(peak_ram / 1024, 2),
        'n_cores_logical': n_cores,
        'monitoring_duration_sec': timeseries[-1]['t'],
    }


def plot_resource_profile(timeseries: list, summary: dict, output_path: str) -> None:
    times = [r['t'] for r in timeseries]
    rams  = [r['ram_mb'] for r in timeseries]
    cpus  = [r['cpu_pct'] for r in timeseries]

    color_ram = '#1f77b4'
    color_cpu = '#d62728'

    fig, ax1 = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor('white')

    # Shading fase load model (0–30 detik)
    ax1.axvspan(0, 30, alpha=0.12, color='gray', zorder=0)

    # ── RAM (axis kiri) ──────────────────────────────────────────────────────
    ax1.set_xlabel('Waktu Sesi (detik)', fontsize=11)
    ax1.set_ylabel('Penggunaan RAM (MB)', color=color_ram, fontsize=11)
    ax1.plot(times, rams, color=color_ram, linewidth=2.5, zorder=3, label='RAM (MB)')
    ax1.tick_params(axis='y', labelcolor=color_ram)
    ax1.set_ylim(0, 2100)
    ax1.set_xlim(0, 310)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(400))
    ax1.set_xticks(range(0, 320, 50))
    ax1.grid(axis='y', linestyle=':', alpha=0.4, zorder=0)

    # Anotasi RAM peak
    peak_ram = summary['peak_ram_mb']
    ax1.annotate(
        f"RAM peak: {peak_ram:.0f} MB",
        xy=(265, peak_ram),
        xytext=(200, peak_ram + 130),
        fontsize=8.5,
        color=color_ram,
        arrowprops=dict(arrowstyle='->', color=color_ram, lw=1.2),
    )

    # ── CPU (axis kanan) ─────────────────────────────────────────────────────
    ax2 = ax1.twinx()
    ax2.set_ylabel('Penggunaan CPU Agregat (%)', color=color_cpu, fontsize=11)
    ax2.plot(times, cpus, color=color_cpu, linewidth=1.3,
             alpha=0.85, zorder=2, label='CPU (%)')
    ax2.tick_params(axis='y', labelcolor=color_cpu)
    ax2.set_ylim(0, 600)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(100))

    # Garis rata-rata CPU
    avg_cpu = summary['avg_cpu_pct_aggregate']
    avg_per_core = summary['avg_cpu_pct_per_core']
    ax2.axhline(avg_cpu, color=color_cpu, linestyle='--',
                linewidth=0.8, alpha=0.5, zorder=1)
    ax2.annotate(
        f"Rata-rata CPU: {avg_cpu:.1f}%\n({avg_per_core:.1f}% per core)",
        xy=(300, avg_cpu),
        xytext=(220, avg_cpu + 90),
        fontsize=7.5,
        color=color_cpu,
        alpha=0.85,
    )

    # ── Label fase ───────────────────────────────────────────────────────────
    ax1.text(15, 120, 'Fase\nLoad\nModel', fontsize=8, color='gray',
             ha='center', va='bottom', style='italic')
    ax1.text(165, 120, 'Fase Inferensi Aktif', fontsize=8.5, color='#555555',
             ha='center', va='bottom', style='italic')

    # ── Legend ───────────────────────────────────────────────────────────────
    ram_line   = plt.Line2D([0], [0], color=color_ram, linewidth=2.5,
                             label='RAM (MB)')
    cpu_line   = plt.Line2D([0], [0], color=color_cpu, linewidth=1.3,
                             label='CPU Agregat (%)')
    load_patch = mpatches.Patch(color='gray', alpha=0.3,
                                label='Fase Load Model (0–30s)')
    ax1.legend(handles=[ram_line, cpu_line, load_patch],
               loc='center right', fontsize=9, framealpha=0.9)

    # ── Kotak ringkasan statistik ─────────────────────────────────────────────
    stats_text = (
        f"Ringkasan ({summary['n_cores_logical']} logical core)\n"
        f"CPU rata-rata : {avg_per_core:.1f}% per core\n"
        f"CPU peak      : {summary['peak_cpu_pct_per_core']:.1f}% per core\n"
        f"RAM peak      : {peak_ram:.0f} MB "
        f"({summary['peak_ram_gb']:.2f} GB)"
    )
    ax1.text(
        0.01, 0.97, stats_text,
        transform=ax1.transAxes,
        fontsize=8,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                  edgecolor='#cccccc', alpha=0.9),
    )

    plt.title(
        'Profil Penggunaan CPU dan RAM Selama Sesi Berjalan (T4-02)',
        fontsize=12, fontweight='bold', pad=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Grafik disimpan: {output_path}')


_ROOT = Path(__file__).resolve().parent.parent
_RESULTS = _ROOT / 'results'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate grafik profil CPU & RAM dari resource_profile_t4.json'
    )
    parser.add_argument(
        '--input', '-i',
        default=str(_RESULTS / 'resource_profile_t4.json'),
        help='Path ke file JSON input',
    )
    parser.add_argument(
        '--output', '-o',
        default=str(_RESULTS / 't4_02_resource_profile.png'),
        help='Path ke file PNG output',
    )
    parser.add_argument(
        '--cores', '-c',
        type=int,
        default=8,
        help='Jumlah logical CPU core (default: 8)',
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f'ERROR: File tidak ditemukan: {args.input}', file=sys.stderr)
        print('Pastikan resource_profile_t4.json ada di direktori yang sama,')
        print('atau tentukan path dengan: --input /path/to/file.json')
        sys.exit(1)

    print(f'Membaca data dari: {args.input}')
    timeseries, summary_from_file = load_json(args.input)

    # Gunakan summary dari file kalau ada, hitung ulang kalau tidak
    if summary_from_file and 'avg_cpu_pct_per_core' in summary_from_file:
        summary = summary_from_file
        # Override n_cores dari argumen kalau berbeda
        if args.cores != summary.get('n_cores_logical', 8):
            summary = compute_summary(timeseries)
            summary['n_cores_logical'] = args.cores
            summary['avg_cpu_pct_per_core'] = round(
                summary['avg_cpu_pct_aggregate'] / args.cores, 1)
            summary['peak_cpu_pct_per_core'] = round(
                summary['peak_cpu_pct_aggregate'] / args.cores, 1)
    else:
        summary = compute_summary(timeseries)
        summary['n_cores_logical'] = args.cores

    print(f'Data valid: {len(timeseries)} titik, '
          f't=0–{timeseries[-1]["t"]}s')

    plot_resource_profile(timeseries, summary, args.output)

    print()
    print('=== Ringkasan ===')
    for k, v in summary.items():
        print(f'  {k}: {v}')