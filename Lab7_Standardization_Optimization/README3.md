# M7 Optimization Lab: CNN with ONNX

This repository contains the lab materials for Module 7, covering model optimization by converting a pre-trained CNN to the ONNX format and benchmarking it with ONNX Runtime.

## Repository Structure

```
lab_7/
├── data/
│   └── sample_images/
│       └── cat.jpg             # Placeholder for sample image
├── models/                     # Output directory for saved models and metrics
├── scripts/
│   ├── m7_l1_baseline.py       # Lab 1: Baseline measurement script
│   ├── m7_l2_export_onnx.py    # Lab 2: ONNX export script
│   └── m7_l3_onnxruntime_benchmark.py # Lab 3: ONNX Runtime benchmark script
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Lab 1: Establish the Baseline

```bash
python scripts/m7_l1_baseline.py
```
This script loads ResNet-18, saves the checkpoint, and measures its size and CPU latency.

### 3. Lab 2: Export to ONNX

```bash
python scripts/m7_l2_export_onnx.py
```
This script converts the PyTorch model to the ONNX format and validates the output.

### 4. Lab 3: Benchmark with ONNX Runtime

```bash
python scripts/m7_l3_onnxruntime_benchmark.py
```
This script runs inference using ONNX Runtime and compares the performance against the PyTorch baseline.

