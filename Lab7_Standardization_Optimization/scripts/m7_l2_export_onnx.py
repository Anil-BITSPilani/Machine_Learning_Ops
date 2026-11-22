import torch
from torchvision import models
import os
import onnx

def main():
    # --- Segment 1: Load Same Model & Weights ---
    print("--- Loading Pre-trained Model for Export ---")
    device = torch.device("cpu")
    model = models.resnet18()
    # Load the state dict we saved in the baseline script for consistency
    state_dict = torch.load("models/resnet18_baseline.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    print("Model loaded with baseline weights.")

    # --- Segment 2: Export to ONNX ---
    print("\n--- Exporting to ONNX ---")
    os.makedirs("models", exist_ok=True)
    onnx_path = "models/resnet18.onnx"
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Exported model to {onnx_path}")

    # --- Segment 3: Validate the ONNX File ---
    print("\n--- Validating ONNX Model ---")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid ✅")
    print(f"Number of nodes: {len(onnx_model.graph.node)}")

    # --- Segment 4: Compare File Sizes ---
    print("\n--- Comparing File Sizes ---")
    ckpt_path = "models/resnet18_baseline.pt"
    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

    print(f"PyTorch checkpoint: {ckpt_size_mb:.2f} MB")
    print(f"ONNX model:         {onnx_size_mb:.2f} MB")
    print(f"Delta:              {onnx_size_mb - ckpt_size_mb:+.2f} MB")

if __name__ == "__main__":
    main()
