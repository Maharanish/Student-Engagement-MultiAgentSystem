import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    from transformers import TimesformerConfig, TimesformerForVideoClassification
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

_PKG_ROOT = Path(__file__).resolve().parent.parent

from mas_engagement.config import MODEL_CONFIG_DIR, NUM_CLASSES  # noqa: E402

_N_RUNS: int = 20
_N_WARMUP: int = 3
_INPUT_SHAPE: tuple = (1, 8, 3, 224, 224)


def main() -> None:
    if not _TRANSFORMERS_AVAILABLE:
        print(
            "ERROR: transformers not installed.\n"
            "  pip install transformers",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = (
        TimesformerConfig.from_pretrained(str(MODEL_CONFIG_DIR))
        if MODEL_CONFIG_DIR.is_dir()
        else TimesformerConfig(num_labels=NUM_CLASSES)
    )
    model = TimesformerForVideoClassification(cfg)
    model.eval()

    dummy = torch.randn(*_INPUT_SHAPE)

    print(f"Warming up ({_N_WARMUP} runs)…")
    for _ in range(_N_WARMUP):
        with torch.no_grad():
            _ = model(pixel_values=dummy)

    print(f"Benchmarking ({_N_RUNS} runs)…")
    latencies: list = []
    for _ in range(_N_RUNS):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(pixel_values=dummy)
        latencies.append((time.perf_counter() - t0) * 1_000.0)

    arr = np.array(latencies)
    print()
    print(f"Runs : {_N_RUNS}")
    print(f"P50  : {np.percentile(arr, 50):.1f} ms")
    print(f"P95  : {np.percentile(arr, 95):.1f} ms")
    print(f"P99  : {np.percentile(arr, 99):.1f} ms")

    feasible = np.percentile(arr, 95) < 2000.0
    print()
    print(f"Real-time feasible (P95 < 2 s): {'YES' if feasible else 'NO'}")


if __name__ == "__main__":
    main()
