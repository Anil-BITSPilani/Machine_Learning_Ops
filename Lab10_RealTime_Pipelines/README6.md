# M10 Data Pipelines Lab: Incremental Ingestion & Retraining Triggers

This repository contains the lab materials for Module 10, covering the fundamentals of data pipelines, incremental data ingestion, and connecting data freshness to model retraining.

## Repository Structure

```
Lab_10/
├── artifacts_m10/              # Output directory for training data and state
├── data_m10/                   # Landing zone for daily data files
├── scripts/
│   ├── m10_l1_simulate_daily_data.py   # Lab 1: Generates synthetic daily data
│   ├── m10_l2_incremental_ingestion.py # Lab 2: Incremental ingestion script
│   └── m10_l3_data_to_train_flow.py    # Lab 3: Orchestrates ingestion and retraining
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Lab 1: Simulate Daily Data Arrival

```bash
cd scripts
python m10_l1_simulate_daily_data.py
```
This creates `day_1.csv` through `day_5.csv` in the `data_m10/` directory.

### 3. Lab 2: Run Incremental Ingestion

```bash
python m10_l2_incremental_ingestion.py
```
This ingests the daily files into a master `training_data.csv` and demonstrates idempotency.

### 4. Lab 3: Run the Full Data-to-Train Flow

```bash
python m10_l3_data_to_train_flow.py
```
This orchestrates the ingestion and triggers a dummy retraining pipeline based on a simple policy.

