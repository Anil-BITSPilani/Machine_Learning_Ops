import torch
from torchvision import models
import os
import time
import numpy as np
from PIL import Image
from torchvision import transforms
import json

def main():
    # --- Segment 1: Load a Pre-trained CNN ---
    print("--- Loading Pre-trained Model ---")
    device = torch.device("cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval().to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ResNet-18")
    print(f"Total parameters: {total_params:,}")

    # --- Segment 2: Save & Measure Model Size ---
    print("\n--- Measuring Model Size ---")
    os.makedirs("models", exist_ok=True)
    ckpt_path = "models/resnet18_baseline.pt"
    torch.save(model.state_dict(), ckpt_path)
    
    size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"PyTorch checkpoint size: {size_mb:.2f} MB")

    # --- Segment 3: Measure Latency on Dummy Input ---
    print("\n--- Measuring Inference Latency ---")
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            
    # Timed runs
    n_runs = 100
    latencies_ms = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.time()
            _ = model(dummy_input)
            t1 = time.time()
            latencies_ms.append((t1 - t0) * 1000)
            
    latencies_ms = np.array(latencies_ms)
    avg_latency = latencies_ms.mean()
    p95_latency = np.percentile(latencies_ms, 95)
    print(f"Avg latency (PyTorch): {avg_latency:.2f} ms")
    print(f"p95 latency (PyTorch): {p95_latency:.2f} ms")

    # --- Segment 4: Tiny Accuracy Sanity Check ---
    print("\n--- Running Sanity Check ---")
    # Create a dummy image as a placeholder
    dummy_image = Image.new("RGB", (300, 300), color = "red")
    dummy_image.save("data/sample_images/cat.jpg")

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    img = Image.open("data/sample_images/cat.jpg").convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        top_prob, top_idx = probs.max(dim=1)
        
    print(f"Sanity check prediction: Class {top_idx.item()} with probability {top_prob.item():.4f}")

    # --- Segment 5: Baseline Summary ---
    baseline_metrics = {
        "model": "ResNet-18",
        "framework": "PyTorch",
        "size_mb": round(size_mb, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2)
    }
    
    with open("models/baseline_metrics.json", "w") as f:
        json.dump(baseline_metrics, f, indent=4)
        
    print("\n--- Baseline Summary ---")
    print(json.dumps(baseline_metrics, indent=4))
    print("\nBaseline metrics saved to models/baseline_metrics.json")

if __name__ == "__main__":
    main()
