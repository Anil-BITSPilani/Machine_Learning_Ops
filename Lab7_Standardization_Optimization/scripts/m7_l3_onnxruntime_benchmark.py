import onnxruntime as ort
import numpy as np
import time
import json
import os
from PIL import Image
from torchvision import transforms

def main():
    # --- Segment 1: Create ONNX Runtime Session ---
    print("--- Creating ONNX Runtime Session ---")
    onnx_path = "models/resnet18.onnx"
    
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"], # Explicitly use CPU for a fair comparison
    )
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Input name: {input_name}")
    print(f"Output name: {output_name}")

    # --- Segment 2: Single Inference + Sanity Check ---
    print("\n--- Running Single Inference Sanity Check ---")
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    outputs = session.run([output_name], {input_name: dummy_input})
    logits = outputs[0]
    print(f"Logits shape: {logits.shape}")
    print(f"First 5 logits: {logits[0, :5]}")

    # --- Segment 3: Measure Latency ---
    print("\n--- Measuring ONNX Runtime Inference Latency ---")
    # Warm-up
    for _ in range(10):
        _ = session.run([output_name], {input_name: dummy_input})
        
    # Timed runs
    n_runs = 100
    latencies_ms = []
    for _ in range(n_runs):
        t0 = time.time()
        _ = session.run([output_name], {input_name: dummy_input})
        t1 = time.time()
        latencies_ms.append((t1 - t0) * 1000)
        
    latencies_ms = np.array(latencies_ms)
    avg_latency = latencies_ms.mean()
    p95_latency = np.percentile(latencies_ms, 95)
    print(f"Avg latency (ONNX Runtime): {avg_latency:.2f} ms")
    print(f"p95 latency (ONNX Runtime): {p95_latency:.2f} ms")

    # --- Segment 4: Compare with Baseline ---
    print("\n--- Comparison Summary ---")
    with open("models/baseline_metrics.json", "r") as f:
        baseline_metrics = json.load(f)
        
    onnx_metrics = {
        "model": "ResNet-18",
        "framework": "ONNX Runtime",
        "size_mb": round(os.path.getsize(onnx_path) / (1024 * 1024), 2),
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2)
    }
    
    # Extract values for cleaner f-string formatting
    bl_avg = baseline_metrics["avg_latency_ms"]
    bl_p95 = baseline_metrics["p95_latency_ms"]
    bl_size = baseline_metrics["size_mb"]
    onnx_avg = onnx_metrics["avg_latency_ms"]
    onnx_p95 = onnx_metrics["p95_latency_ms"]
    onnx_size = onnx_metrics["size_mb"]
    
    print("| Metric         | PyTorch (Baseline) | ONNX Runtime | Delta      |")
    print("|----------------|--------------------|--------------|------------|")
    print(f"| Avg Latency (ms) | {bl_avg:18} | {onnx_avg:12} | {onnx_avg - bl_avg:+10.2f} |")
    print(f"| p95 Latency (ms) | {bl_p95:18} | {onnx_p95:12} | {onnx_p95 - bl_p95:+10.2f} |")
    print(f"| Model Size (MB)  | {bl_size:18} | {onnx_size:12} | {onnx_size - bl_size:+10.2f} |")

if __name__ == "__main__":
    main()
