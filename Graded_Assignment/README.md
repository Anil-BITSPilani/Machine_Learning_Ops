# Real-Time & Batch Financial Transaction Fraud Detection System (Option B)

End-to-end production ML engineering system built for **BITS Pilani WILP — ML Model Engineering**.

---

## 1. System Overview & Repository Layout

This system addresses extreme class imbalance (`0.172%` fraud across `284,807` transactions) in payment fraud detection through a leakage-free training pipeline, cost-sensitive gradient boosting, dual inference paths (low-latency FastAPI REST service + vectorized batch CLI), automated fallback resilience, empirical latency/throughput profiling, and PSI/KS data drift monitoring.

```text
Graded_Assignment/
├── Dockerfile                        # Multi-layer lightweight production API container
├── .gitignore                        # Excludes raw CSVs, caches, and temporary batch outputs
├── requirements.txt                  # Pinned Python 3.12 production & test dependencies
├── README.md                         # Architecture overview, setup, and runbook
├── data/
│   └── creditcard.csv                # Auto-downloaded from OpenML (id=1597) on first run
├── artifacts/
│   ├── fraud_pipeline.joblib         # Serialized SimpleImputer + RobustScaler + LightGBM pipeline
│   ├── metadata.json                 # Tuned threshold, calibration status, cost metrics & fallback cutoff
│   ├── train_reference.parquet       # 10,000-row training baseline sample for drift monitoring
│   └── drift_report.json             # Structured feature-level PSI & KS drift audit output
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py              # OpenML ingestion, cyclical time features & strict 70/15/15 split
│   ├── evaluate.py                   # PR-AUC, F1, explicit threshold calibration & cost-weighted loss
│   ├── train.py                      # Baseline vs. LightGBM training, calibration & artifact export
│   ├── drift_monitor.py              # Production vs. reference PSI & KS drift detection with alert rules
│   └── serving/
│       ├── __init__.py
│       ├── schemas.py                # Strict Pydantic contracts (extra="forbid", NaN/Inf rejection, None imputation)
│       ├── app.py                    # FastAPI REST service (/health, /predict) with logged fallback rule
│       ├── predict_batch.py          # Vectorized batch CLI scorer with record IDs & throughput logging
│       └── benchmark.py              # Repeated-request p50/p95/p99 latency & environment profiler
└── tests/
    ├── test_api.py                   # API contract, probability indexing, imputation, NaN/Inf & fallback tests
    ├── test_batch.py                 # Batch CLI scoring, ID generation, missing-column & fallback tests
    └── test_evaluate.py              # Threshold calibration constraint/fallback & financial cost tests
```

---

## 2. Setup & Model Training

### Prerequisites
* Python 3.12+ (with `.venv` active)
* Docker Desktop (optional, for containerized API execution)

### Installation & Training
```bash
# 1. Activate the project virtual environment
source .venv/bin/activate

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Fetch dataset, train baseline & LightGBM models, calibrate thresholds, and serialize artifacts
python -m src.train

# 4. Run full automated unit & edge-case test suite
pytest -v
```

### Modeling, Threshold Calibration & Business Cost Policy
* **Leak-Free Partitioning (`src/data_pipeline.py`):** Transactions are sorted strictly chronologically by `Time` and partitioned **70% Train / 15% Validation / 15% Test** without shuffling. `SimpleImputer(strategy="median")` and `RobustScaler()` are fitted exclusively on `X_train` inside `sklearn.pipeline.Pipeline`.
* **Imbalance Handling (`src/train.py`):** Compares class-weighted `LogisticRegression` against cost-sensitive `LGBMClassifier(scale_pos_weight = sqrt(N_neg / N_pos))`. The actual trained LightGBM weight is dampened from the raw `518.18:1` imbalance ratio to `22.76` to reduce probability saturation.
* **Explicit Threshold Calibration (`src/evaluate.py`):** Selects the validation threshold maximizing F1 subject to `Precision >= 0.80` (`status: "constraint_satisfied"`). If a model cannot satisfy `Precision >= 0.80` at any non-zero recall operating point, it logs an explicit warning and falls back deterministically to the unconstrained validation max-F1 threshold (`status: "fallback_max_f1"`), recording the outcome in `artifacts/metadata.json`.
* **Cost-Weighted Evaluation (`src/evaluate.py`):** Quantifies asymmetric financial trade-offs assuming `$10.00` operational review/friction cost per flagged alert (`FP` or `TP`) and `Amount + $25.00` chargeback loss per missed fraud (`FN`), reporting **Total Expected Cost**, **Cost per 1,000 Transactions**, and **Net Fraud Savings**.

### Verified End-to-End Results
* **Test suite:** `23 passed` under Python 3.12.
* **Held-out LightGBM:** PR-AUC `0.7263`, fraud precision `0.8372`, fraud recall `0.6923`, fraud F1 `0.7579`, and 36 true positives / 7 false positives / 16 false negatives.
* **Cost analysis:** The test configuration reported `$3,287.18` total expected cost, `$76.94` cost per 1,000 transactions, and `$4,181.70` net fraud savings.
* **Live local API smoke benchmark:** 100 measured sequential requests (10 warm-up requests) produced mean `1.55 ms`, p50 `1.53 ms`, p95 `1.96 ms`, p99 `2.06 ms`, maximum `2.19 ms`, and `643.6 requests/sec` on an Apple-arm host.
* **Normal temporal drift report:** V12, hour_sin, and hour_cos were critical; V17 and V4 were warnings. This report is separate from the synthetic shift demonstration.

---

## 3. Running Inference Services (Online API & Batch CLI)

### Path A: Real-Time Online REST API (Local Uvicorn)
```bash
uvicorn src.serving.app:app --host 127.0.0.1 --port 8000
```
* **Interactive Swagger Docs:** `http://127.0.0.1:8000/docs`
* **Health Check (`GET http://127.0.0.1:8000/health`):**
  ```bash
  curl -s http://127.0.0.1:8000/health
  ```
* **Real-Time Scoring (`POST http://127.0.0.1:8000/predict`):**
  * Rejects unknown keys (`ConfigDict(extra="forbid")`), negative amounts, and `NaN`/`Infinity` tokens with HTTP `422`.
  * Accepts `null` or omitted `Amount`/`V1`–`V28` features and imputes them via the pipeline's `SimpleImputer`.
  * Logs full stack traces via `logger.exception()` and degrades gracefully to a deterministic high-amount rule (`Amount >= 99.5th percentile`, `scoring_mode: "fallback_rule"`) if model inference fails.
  ```bash
  curl -s -X POST http://127.0.0.1:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"transaction_id":"tx-demo-001","Time":406.0,"Amount":149.62}'
  ```

### Path B: Containerized REST API (Docker)
```bash
docker build -t fraud-detection-api:1.0 .
docker run -d -p 8000:8000 --name fraud-api fraud-detection-api:1.0
curl -s http://127.0.0.1:8000/health
```

### Path C: Vectorized Batch CLI Scoring (`src/serving/predict_batch.py`)
```bash
# Generate a 10,000-record unseen batch file from the tail of creditcard.csv
python -c "import pandas as pd; pd.read_csv('data/creditcard.csv').tail(10000).drop(columns=['Class']).to_csv('data/sample_batch_input.csv', index=False)"

# Run batch inference CLI (preserves/assigns transaction_id, logs status & records/sec throughput)
python -m src.serving.predict_batch data/sample_batch_input.csv artifacts/batch_predictions.csv
```

---

## 4. Empirical Latency Profiling (`src/serving/benchmark.py`)

With the FastAPI server running on `http://127.0.0.1:8000`, run the benchmark script in a second terminal to profile `POST http://127.0.0.1:8000/predict`:

```bash
source .venv/bin/activate
python -m src.serving.benchmark --samples 1000 --warmup 50
```

* **Environment & Hardware Logging:** Records OS/kernel version, processor architecture, logical CPU count, and Python runtime version.
* **Warm-up Protocol:** Executes `--warmup` unmeasured requests (default: `50`) to prime model memory, OpenMP threads, and local socket pools before measuring `--samples` sequential `POST http://127.0.0.1:8000/predict` requests (default: `1,000`).
* **Reported Metrics:** Outputs **Mean**, **p50**, **p95**, **p99**, **Max** end-to-end latency (`ms`) and sustained throughput (`requests/sec`).

---

## 5. Data Drift Monitoring (`src/drift_monitor.py`)

Compare `artifacts/train_reference.parquet` against a production evaluation batch and export an audit report to `artifacts/drift_report.json`:

```bash
# Standard drift evaluation against the latest temporal test window
python -m src.drift_monitor

# Evaluate a custom production batch file (CSV or Parquet)
python -m src.drift_monitor --prod path/to/production_batch.parquet

# Verify critical alert triggers using synthetic covariate shift
python -m src.drift_monitor --simulate-shift
```

### Alerting Thresholds & Operational Actions

| Severity Level | Statistical Trigger Rule | Operational Action |
| :--- | :--- | :--- |
| **OK (`NO_ACTION_REQUIRED`)** | `PSI < 0.10` and (`KS < 0.10` or `p >= 0.01`) | Continue normal production serving |
| **Warning (`SEND_OPS_WARNING`)** | `0.10 <= PSI < 0.25` or (`KS >= 0.10` and `p < 0.01`) | Notify ML Ops on-call; inspect upstream feature pipeline |
| **Critical (`TRIGGER_RETRAINING`)** | `PSI >= 0.25` on one or more monitored features | Trigger automated model retraining & shadow validation |