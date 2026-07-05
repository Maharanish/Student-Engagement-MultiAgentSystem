"""
scripts/stress_test_blackboard.py

Stress-tests SharedState (Blackboard) thread-safety and throughput.

Tujuan: membuktikan secara empiris bahwa
  1. append_engagement tidak kehilangan evidence di bawah concurrent load
     (engagement_history adalah list tak-terbatas — setiap write tersimpan).
  2. latest-wins pada action/message queue (maxsize=1) SENGAJA menjatuhkan
     item lama — bukan bug, tapi desain. Drop-rate yang terukur memvalidasi
     bahwa mekanisme ini berjalan benar dan tidak menyebabkan race condition.
  3. Tidak ada deadlock walau pun N thread menulis bersamaan.

Tiga skenario berdasarkan write-rate per producer:
  normal   (0.10 Hz  ≈ INFERENCE_INTERVAL 10 s)
  high     (1.00 Hz  per-producer)
  extreme  (10.0 Hz  per-producer)

Metrik per skenario:
  written         — total append_engagement calls (counter thread-safe via lock)
  history_final   — panjang engagement_history setelah run selesai
  evidence_loss%  — (written − history_final) / written × 100 (diharapkan 0%)
  enqueued        — total set_pending_action calls
  consumed        — total consume_pending_action yang berhasil
  drop%           — (enqueued − consumed) / enqueued × 100
  batch_avg       — rata-rata record baru per tick consumer (simulasi 1 Hz orchestrator)
  batch_max       — maksimum record baru dalam satu tick
  throughput_wps  — writes per second efektif
  errors          — exception yang tertangkap selama run
  deadlock        — True jika ada thread gagal join dalam 5 detik

Output: tabel terminal + stress_test_results.csv
Run   : python scripts/stress_test_blackboard.py
"""

import csv
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Resolve repo root so script works from any cwd
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from mas_engagement.blackboard import SharedState  # noqa: E402 — path setup above

# ── Constants ──────────────────────────────────────────────────────────────────

DURATION_SEC = 30
N_PRODUCERS = 4
CONSUMER_POLL_HZ = 1.0           # matches OrchestratorAgent poll rate
DUMMY_SOFTMAX = [0.85, 0.05, 0.05, 0.05]

SCENARIOS = [
    ("normal",  0.10),   # 1 write / 10 s — baseline (matches INFERENCE_INTERVAL)
    ("high",    1.00),   # 1 write / s
    ("extreme", 10.0),   # 10 writes / s
]

# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    label: str
    write_rate_hz: float
    n_producers: int
    duration_sec: float = 0.0

    # Evidence integrity
    total_written: int = 0
    history_final: int = 0          # len(engagement_history) at end

    # Action queue
    action_enqueued: int = 0
    action_consumed: int = 0

    # Per-tick batch stats (list collected during run)
    _tick_batches: List[int] = field(default_factory=list, repr=False)

    # Health
    errors: int = 0
    deadlock: bool = False

    # ── Derived ───────────────────────────────────────────────────────────────

    @property
    def evidence_loss_pct(self) -> float:
        if self.total_written == 0:
            return 0.0
        return 100.0 * max(0, self.total_written - self.history_final) / self.total_written

    @property
    def action_drop_pct(self) -> float:
        if self.action_enqueued == 0:
            return 0.0
        dropped = max(0, self.action_enqueued - self.action_consumed)
        return 100.0 * dropped / self.action_enqueued

    @property
    def throughput_wps(self) -> float:
        return self.total_written / self.duration_sec if self.duration_sec > 0 else 0.0

    @property
    def batch_avg(self) -> float:
        return statistics.mean(self._tick_batches) if self._tick_batches else 0.0

    @property
    def batch_max(self) -> int:
        return max(self._tick_batches) if self._tick_batches else 0


# ── Core scenario runner ──────────────────────────────────────────────────────

def run_scenario(label: str, write_rate_hz: float, n_producers: int) -> ScenarioResult:
    """
    Jalankan satu skenario stress: N_PRODUCERS writer threads + 1 consumer.

    Producer mensimulasikan DetectionAgent: memanggil append_engagement dan
    set_pending_action pada frekuensi write_rate_hz.

    Consumer mensimulasikan OrchestratorAgent: snapshot + filter timestamp >
    last_poll setiap 1 / CONSUMER_POLL_HZ detik, lalu drain action queue.
    """
    result = ScenarioResult(label=label, write_rate_hz=write_rate_hz, n_producers=n_producers)
    bb = SharedState(warmup_duration=0)
    bb.set_session_start_time(time.time())

    stop = threading.Event()
    counter_lock = threading.Lock()

    # Shared counters (written by producers, read after all threads stop)
    _written = 0
    _action_enqueued = 0
    _action_consumed = 0
    _errors = 0
    _tick_batches: List[int] = []

    write_interval = 1.0 / write_rate_hz

    # ── Producer ──────────────────────────────────────────────────────────────

    def producer(pid: int) -> None:
        nonlocal _written, _action_enqueued, _errors
        local_written = 0
        local_enqueued = 0
        local_errors = 0

        t_next = time.time()
        while not stop.is_set():
            now = time.time()
            if now >= t_next:
                try:
                    # Simulate evidence arriving with current timestamp
                    # (matches detection.py post-inference re-stamp)
                    bb.append_engagement(0, 0.85, now, softmax=DUMMY_SOFTMAX[:])
                    local_written += 1

                    # Also exercise action queue (simulates orchestrator's
                    # set_pending_action; latest-wins means concurrent producers
                    # will race to overwrite each other — by design)
                    bb.set_pending_action("tier_1", now)
                    local_enqueued += 1
                except Exception as exc:  # noqa: BLE001
                    local_errors += 1

                t_next += write_interval

            # Sleep for the remainder of the interval (or until stop), capped
            # at 10 ms to stay responsive without busy-spinning
            sleep_for = min(max(t_next - time.time(), 0.0), 0.01)
            stop.wait(sleep_for)

        with counter_lock:
            _written += local_written
            _action_enqueued += local_enqueued
            _errors += local_errors

    # ── Consumer (simulates Orchestrator at CONSUMER_POLL_HZ) ────────────────

    def consumer() -> None:
        nonlocal _action_consumed, _errors
        local_consumed = 0
        local_errors = 0
        local_batches: List[int] = []

        poll_interval = 1.0 / CONSUMER_POLL_HZ
        last_poll_ts = time.time()

        while not stop.is_set():
            stop.wait(poll_interval)
            now = time.time()

            try:
                snap = bb.snapshot()
                # Replicate orchestrator's timestamp filter:
                # only count records newer than last poll
                new_records = [
                    e for e in snap["engagement_history"]
                    if e["timestamp"] > last_poll_ts
                ]
                local_batches.append(len(new_records))
                last_poll_ts = now

                # Drain action queue (Intervention agent role)
                item = bb.consume_pending_action(timeout=None)
                if item is not None:
                    local_consumed += 1
            except Exception as exc:  # noqa: BLE001
                local_errors += 1

        # Final drain after stop: pick up anything written in the last tick
        while True:
            item = bb.consume_pending_action(timeout=None)
            if item is None:
                break
            local_consumed += 1

        with counter_lock:
            _action_consumed += local_consumed
            _errors += local_errors
            _tick_batches.extend(local_batches)

    # ── Launch ────────────────────────────────────────────────────────────────

    producer_threads = [
        threading.Thread(target=producer, args=(i,), daemon=True, name=f"prod-{i}")
        for i in range(n_producers)
    ]
    consumer_thread = threading.Thread(target=consumer, daemon=True, name="consumer")

    t0 = time.time()
    for t in producer_threads:
        t.start()
    consumer_thread.start()

    time.sleep(DURATION_SEC)
    stop.set()

    # Join producers first so all writes complete before final snapshot
    for t in producer_threads:
        t.join(timeout=5.0)
        if t.is_alive():
            result.deadlock = True

    # Take final snapshot AFTER all producers have stopped
    final_snap = bb.snapshot()
    result.history_final = len(final_snap["engagement_history"])

    consumer_thread.join(timeout=5.0)
    if consumer_thread.is_alive():
        result.deadlock = True

    result.duration_sec = time.time() - t0
    result.total_written = _written
    result.action_enqueued = _action_enqueued
    result.action_consumed = _action_consumed
    result.errors = _errors
    result._tick_batches = _tick_batches

    return result


# ── Output helpers ────────────────────────────────────────────────────────────

_HEADER = (
    f"{'Scenario':<10} {'Rate/Hz':<9} {'Written':<9} {'HistFin':<9} "
    f"{'EvidLoss%':<11} {'Enqueued':<10} {'Consumed':<10} "
    f"{'Drop%':<8} {'Batch avg':<11} {'Batch max':<11} "
    f"{'Wps':<8} {'Err':<5} Deadlock"
)

def _row(r: ScenarioResult) -> str:
    return (
        f"{r.label:<10} {r.write_rate_hz:<9.2f} {r.total_written:<9} {r.history_final:<9} "
        f"{r.evidence_loss_pct:<11.1f} {r.action_enqueued:<10} {r.action_consumed:<10} "
        f"{r.action_drop_pct:<8.1f} {r.batch_avg:<11.2f} {r.batch_max:<11} "
        f"{r.throughput_wps:<8.2f} {r.errors:<5} {r.deadlock}"
    )


def _save_csv(results: List[ScenarioResult], path: Path) -> None:
    fieldnames = [
        "scenario", "write_rate_hz", "n_producers", "duration_sec",
        "total_written", "history_final", "evidence_loss_pct",
        "action_enqueued", "action_consumed", "action_drop_pct",
        "batch_avg", "batch_max", "throughput_wps", "errors", "deadlock",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({
                "scenario": r.label,
                "write_rate_hz": r.write_rate_hz,
                "n_producers": r.n_producers,
                "duration_sec": round(r.duration_sec, 2),
                "total_written": r.total_written,
                "history_final": r.history_final,
                "evidence_loss_pct": round(r.evidence_loss_pct, 2),
                "action_enqueued": r.action_enqueued,
                "action_consumed": r.action_consumed,
                "action_drop_pct": round(r.action_drop_pct, 2),
                "batch_avg": round(r.batch_avg, 2),
                "batch_max": r.batch_max,
                "throughput_wps": round(r.throughput_wps, 2),
                "errors": r.errors,
                "deadlock": r.deadlock,
            })


# ── Assertions ────────────────────────────────────────────────────────────────

def _assert_results(results: List[ScenarioResult]) -> None:
    """
    Validasi properti yang harus selalu terpenuhi, terlepas dari write rate.
    Cetak PASS / FAIL per assertion.
    """
    print("\n-- Assertions ------------------------------------------------------")
    failures = 0

    for r in results:
        # 1. No evidence loss: engagement_history harus menyimpan semua write
        ok = r.history_final == r.total_written
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"  [{status}] [{r.label}] evidence integrity: "
              f"written={r.total_written} history={r.history_final} "
              f"(loss={r.evidence_loss_pct:.1f}%)")

    for r in results:
        # 2. No deadlock
        ok = not r.deadlock
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"  [{status}] [{r.label}] no deadlock")

    for r in results:
        # 3. No unexpected exceptions
        ok = r.errors == 0
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"  [{status}] [{r.label}] zero errors (got {r.errors})")

    # 4. Drop rate should increase monotonically with write rate
    if len(results) >= 2:
        drop_rates = [r.action_drop_pct for r in results]
        ok = all(drop_rates[i] <= drop_rates[i + 1] for i in range(len(drop_rates) - 1))
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"  [{status}] action drop rate monotonically increases with write rate "
              f"({[f'{d:.1f}%' for d in drop_rates]})")

    if failures == 0:
        print("\n  Semua assertion LULUS - blackboard thread-safe, no evidence loss.")
    else:
        print(f"\n  {failures} assertion GAGAL - periksa log di atas.")

    print("--------------------------------------------------------------------")


# ── Interpretasi hasil ────────────────────────────────────────────────────────

def _print_interpretation(results: List[ScenarioResult]) -> None:
    print("\n-- Interpretasi ----------------------------------------------------")
    print(
        "  engagement_history adalah list tak-terbatas dengan RLock - setiap\n"
        "  append_engagement() tersimpan permanen. EvidLoss% diharapkan 0%\n"
        "  pada semua skenario.\n"
    )
    print(
        "  Action queue (maxsize=1, latest-wins) dirancang untuk DROP item lama.\n"
        "  Drop% yang tinggi pada skenario extreme adalah EXPECTED BEHAVIOR -\n"
        "  orchestrator hanya butuh action TERBARU, bukan setiap intermediate\n"
        "  decision. Tidak ada data yang hilang secara semantik.\n"
    )
    for r in results:
        total_rate = r.write_rate_hz * r.n_producers
        print(
            f"  [{r.label}] {total_rate:.1f} writes/s total -> "
            f"batch_avg={r.batch_avg:.1f} record/tick, "
            f"drop={r.action_drop_pct:.1f}% pada queue"
        )
    print("--------------------------------------------------------------------")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(
        f"\nBlackboard Stress Test\n"
        f"  producers  : {N_PRODUCERS} threads\n"
        f"  duration   : {DURATION_SEC}s per scenario\n"
        f"  consumer   : 1 thread @ {CONSUMER_POLL_HZ} Hz\n"
        f"  scenarios  : {[s[0] for s in SCENARIOS]}\n"
    )

    results: List[ScenarioResult] = []
    for label, rate_hz in SCENARIOS:
        total_hz = rate_hz * N_PRODUCERS
        print(
            f"[{label.upper():^8}] {rate_hz} Hz/producer x {N_PRODUCERS} = "
            f"{total_hz:.1f} Hz total -- running {DURATION_SEC}s ...",
            flush=True,
        )
        r = run_scenario(label, rate_hz, N_PRODUCERS)
        results.append(r)
        print(
            f"           done: written={r.total_written}, "
            f"loss={r.evidence_loss_pct:.1f}%, "
            f"drop={r.action_drop_pct:.1f}%, "
            f"wps={r.throughput_wps:.1f}"
        )

    # Summary table
    print("\n" + "=" * len(_HEADER))
    print(_HEADER)
    print("-" * len(_HEADER))
    for r in results:
        print(_row(r))
    print("=" * len(_HEADER))

    _assert_results(results)
    _print_interpretation(results)

    # Save CSV next to the script's repo root
    csv_path = _REPO_ROOT / "stress_test_results.csv"
    _save_csv(results, csv_path)
    print(f"\nCSV saved -> {csv_path}")


if __name__ == "__main__":
    main()
