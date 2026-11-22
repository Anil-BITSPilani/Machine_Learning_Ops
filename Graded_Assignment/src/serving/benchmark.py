import argparse
import os
import platform
import sys
import time
import numpy as np
import requests
from src.data_pipeline import fetch_and_cache_data

API_URL = "http://127.0.0.1:8000"


def print_environment_metadata(n_warmup: int, n_samples: int):
    print("\n=== BENCHMARK HARDWARE & ENVIRONMENT METADATA ===")
    print(f"OS / Platform   : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Processor       : {platform.processor() or 'Apple Silicon / x86_64'}")
    print(f"Logical CPUs    : {os.cpu_count()}")
    print(f"Python Runtime  : {sys.version.split()[0]}")
    print(f"Target Endpoint : {API_URL}/predict")
    print(f"Warm-up Requests: {n_warmup} (excluded from latency stats)")
    print(f"Measured Samples: {n_samples} sequential HTTP POST requests")


def run_latency_benchmark(n_samples: int = 1000, n_warmup: int = 50):
    df = fetch_and_cache_data()
    total_needed = n_warmup + n_samples
    sample_rows = df.tail(total_needed).to_dict(orient="records")

    print_environment_metadata(n_warmup=n_warmup, n_samples=n_samples)

    for i in range(n_warmup):
        payload = {k: v for k, v in sample_rows[i].items() if k != "Class"}
        payload["transaction_id"] = f"warmup-{i}"
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        resp.raise_for_status()

    latencies_ms = []
    start_total = time.perf_counter()
    for i, row in enumerate(sample_rows[n_warmup:]):
        payload = {k: v for k, v in row.items() if k != "Class"}
        payload["transaction_id"] = f"bench-{i}"
        t0 = time.perf_counter()
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        resp.raise_for_status()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    total_sec = time.perf_counter() - start_total
    arr = np.array(latencies_ms)
    print(f"\n=== EMPIRICAL API LATENCY RESULTS (N={n_samples}) ===")
    print(f"Mean Latency : {np.mean(arr):.2f} ms")
    print(f"p50 Latency  : {np.percentile(arr, 50):.2f} ms")
    print(f"p95 Latency  : {np.percentile(arr, 95):.2f} ms")
    print(f"p99 Latency  : {np.percentile(arr, 99):.2f} ms")
    print(f"Max Latency  : {np.max(arr):.2f} ms")
    print(f"Throughput   : {n_samples / total_sec:.1f} requests/sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latency profiling benchmark")
    parser.add_argument("--samples", type=int, default=1000, help="Number of measured requests")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warm-up requests")
    args = parser.parse_args()

    run_latency_benchmark(n_samples=args.samples, n_warmup=args.warmup)