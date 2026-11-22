# M6 Retraining Lab: Pipelines & Model Promotion

This repository contains the lab materials for Module 6, covering the fundamentals of model retraining, champion-vs-challenger evaluation, and model promotion workflows.

## Repository Structure

```
Lab_6/
├── configs/
│   ├── train_config_v1.yaml    # Config for initial training run
│   ├── train_config_v2.yaml    # Config for retraining run
│   └── eval_config.yaml        # Config for champion-challenger evaluation
├── data/
│   ├── training_data_v1.csv    # Initial training data
│   ├── training_data_v2.csv    # New data snapshot for retraining
│   └── eval_data.csv           # Evaluation data for comparison
├── scripts/
│   ├── train.py                # Training pipeline script
│   └── evaluate_and_promote.py # Champion-challenger evaluation & promotion script
├── service/
│   └── app.py                  # FastAPI service with registry-based model loading
├── m6_lab_details.md           # Trainer's script and lab guide
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Lab 1: Retrain the Model

```bash
# Run initial training (creates Version 1)
python scripts/train.py --config configs/train_config_v1.yaml

# Run retraining on new data (creates Version 2)
python scripts/train.py --config configs/train_config_v2.yaml
```

### 3. Lab 2: Evaluate and Promote

First, use the MLflow UI (`mlflow ui`) to manually set Version 1 to the "Production" stage.

```bash
# Compare champion (Production) vs challenger (Version 2)
python scripts/evaluate_and_promote.py --config configs/eval_config.yaml --challenger_version 2
```

### 4. Lab 3: Run the Serving Application

```bash
export MLFLOW_TRACKING_URI=file://$(pwd)/mlruns
uvicorn service.app:app --host 0.0.0.0 --port 8000
```
