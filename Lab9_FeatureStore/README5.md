# M9 Feature Store Lab: Offline & Online Features

This repository contains the lab materials for Module 9, covering the fundamentals of feature engineering, simulating an online feature store, and understanding how to avoid training-serving skew.

## Repository Structure

```
Lab_9/
├── scripts/
│   ├── m9_l1_offline_features.py   # Lab 1: Builds an offline feature table
│   ├── m9_l2_online_store.py       # Lab 2: Simulates an online feature store
│   └── m9_l3_skew.py               # Lab 3: Demonstrates and fixes skew
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Lab 1: Build the Offline Feature Table

```bash
cd scripts
python m9_l1_offline_features.py
```
This creates `transactions.csv` and `offline_features.csv` in the `scripts/` directory.

### 3. Lab 2: Simulate the Online Feature Store

```bash
python m9_l2_online_store.py
```
This demonstrates how to build an in-memory feature cache and simulate online lookups.

### 4. Lab 3: Demonstrate and Fix Training-Serving Skew

```bash
python m9_l3_skew.py
```
This shows how inconsistent feature logic leads to skew and how a shared definition fixes it.

