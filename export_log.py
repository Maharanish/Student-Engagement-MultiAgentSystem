"""Utility to export raw JSONL logs into a human-readable JSON format."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

def transform_log_to_readable(log_path: Path | str) -> Path | None:
    """Reads a raw .jsonl log and exports a formatted .json file."""
    log_file = Path(log_path)
    if not log_file.is_file():
        print(f"File log tidak ditemukan: {log_file}")
        return None

    readable_records: List[Dict[str, Any]] = []
    start_time: float | None = None

    try:
        # PENDekatan Brutal: Baca file sebagai BYTES mentah, lalu paksa decode
        with open(log_file, 'rb') as f:
            raw_data = f.read()
        
        # Bersihkan karakter \x00 (NUL byte) yang sering membuat JSON error di Windows
        raw_data = raw_data.replace(b'\x00', b'')
        
        # Decode menjadi string (abaikan karakter aneh/BOM yang tidak bisa di-decode)
        text_data = raw_data.decode('utf-8-sig', errors='ignore')
        
        # Pecah berdasarkan baris enter
        lines = text_data.split('\n')
        
        for baris_ke, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            # Perlindungan jika JSON per baris rusak
            try:
                record = json.loads(line)
            except Exception as json_err:
                print(f"[Abaikan] Baris {baris_ke} rusak dan dilewati: {line[:30]}...")
                continue
                
            ts_raw = record.get("ts")
            if ts_raw is None:
                continue
                
            # Simpan waktu pertama sebagai titik 0
            if start_time is None:
                start_time = float(ts_raw)
                
            # Kalkulasi waktu relatif
            relative_sec = float(ts_raw) - start_time
            hours = int(relative_sec // 3600)
            minutes = int((relative_sec % 3600) // 60)
            seconds = int(relative_sec % 60)
            millis = int((relative_sec * 1000) % 1000)
            
            waktu_format = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
            
            # Ekstrak elemen utama
            agen = record.pop("agent", "unknown")
            event = record.pop("event", "unknown")
            
            record.pop("ts", None)
            
            new_record = {
                "time": waktu_format,
                "agent": agen,
                "event": event,
                "data": record 
            }
            readable_records.append(new_record)
            
        out_file = log_file.with_name(f"{log_file.stem}_readable.json")
        with open(out_file, "w", encoding="utf-8") as out_f:
            json.dump(readable_records, out_f, indent=2)
            
        print(f"Berhasil mengekspor log rapi ke: {out_file}")
        return out_file
        
    except Exception as e:
        print(f"Gagal mentransformasi log: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        transform_log_to_readable(sys.argv[1])
    else:
        print("Penggunaan: python export_log.py logs/session_xxx.jsonl")