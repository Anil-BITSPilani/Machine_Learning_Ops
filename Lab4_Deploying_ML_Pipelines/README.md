# M4 MLOps Lab: Deployment Pipelines & MLOps Basics

This repository contains the lab materials for Module 4, covering the fundamentals of MLOps, including repository structure, experiment tracking with MLflow, and CI/CD pipelines.

## Repository Structure

```
Lab_4/
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI workflow
├── configs/
│   └── train_config.yaml   # Training configuration file
├── data/
│   └── training_data.csv   # Sample training data
├── models/                 # Output directory for trained models
├── scripts/
│   └── train.py            # Main training entrypoint script
├── src/
│   └── ml/
│       ├── data.py         # Data loading utilities
│       └── train.py        # Model training logic
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Training with MLflow

```bash
python scripts/train.py --config configs/train_config.yaml
```

### 3. View Results in MLflow UI

```bash
mlflow ui
```
Then open `http://localhost:5000` in your browser.

