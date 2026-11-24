# M11 Responsible AI Lab: Fairness Evaluation & Auditing

This repository contains the lab materials for **Module 11**, covering the fundamentals of **Responsible AI**, including segmented model evaluation, automated fairness checks, and structured logging for audit trails.

---

## 📋 Overview

Machine learning models can perform well on aggregate metrics while exhibiting significant disparities across demographic subgroups. This lab walks through a practical auditing pipeline on a synthetic binary classification problem (loan approval):

1. **Data & Model Setup:** Generate synthetic tabular data containing a protected group attribute (`Group_A` vs. `Group_B`) and train a baseline `RandomForestClassifier`.
2. **Lab 1 — Segmented Evaluation:** Disaggregate model accuracy, precision, recall (TPR), false positive rate (FPR), and selection rate across groups.
3. **Lab 2 — Automated Fairness Checks:** Validate model predictions against standard quantitative fairness thresholds (Four-Fifths / 80% Rule, Demographic Parity Difference, and Equal Opportunity Difference).
4. **Lab 3 — Audit Trail Logging:** Record SHA-256 artifact checksums, metrics, and compliance outcomes into an append-only JSONL audit log.

---

## 📁 Repository Structure

```text
Lab_11/
├── data/                                    # Holds the validation dataset
├── models/                                  # Holds the trained model
├── logs/                                    # Output directory for audit logs and visualizations
├── scripts/
│   ├── setup_data_and_model.py              # Generates synthetic data and trains a dummy model
│   ├── m11_l1_segmented_evaluation.py       # Lab 1: Segmented evaluation by group
│   ├── m11_l2_fairness_check.py             # Lab 2: Automated fairness check
│   └── m11_l3_logging_fairness.py           # Lab 3: Logging fairness metrics
├── README.md                                # This file
└── requirements.txt                         # Python dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites
* Python **3.9+**
* `pip` package manager

### 2. Installation
Clone or navigate into the `Lab_11/` directory, create a virtual environment (optional), and install dependencies:

```bash
cd Lab_11
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Initialize Dataset & Model
Generate the synthetic validation dataset (`data/validation_data.csv`) and trained model artifact (`models/dummy_model.joblib`):

```bash
python scripts/setup_data_and_model.py
```

---

## 🧪 Lab Exercises

### Lab 1: Segmented Evaluation by Group
Computes overall and subgroup performance metrics and generates a bar chart comparing group rates.

```bash
python scripts/m11_l1_segmented_evaluation.py
```
* **Outputs:**
  * `logs/l1_segmented_metrics.csv` — Tabular breakdown of count, accuracy, precision, TPR, FPR, and selection rate per segment.
  * `logs/l1_segmented_comparison.png` — Visual comparison across subgroups.

---

### Lab 2: Automated Fairness Check
Evaluates subgroup disparities against configurable governance thresholds and outputs a `PASS` / `FAIL` compliance report.

```bash
python scripts/m11_l2_fairness_check.py
```

| Fairness Metric | Definition | Default Threshold |
| :--- | :--- | :--- |
| **Disparate Impact Ratio** | $\frac{\min(\text{Selection Rate})}{\max(\text{Selection Rate})}$ | `>= 0.80` (Four-Fifths Rule) |
| **Demographic Parity Diff** | $\max(\text{Selection Rate}) - \min(\text{Selection Rate})$ | `<= 0.10` |
| **Equal Opportunity Diff** | $\max(\text{TPR}) - \min(\text{TPR})$ | `<= 0.10` |

* **Outputs:**
  * `logs/l2_fairness_check_report.json` — Machine-readable JSON report suitable for CI/CD quality gates.

---

### Lab 3: Logging Fairness Metrics for Audit Trails
Combines segmented metrics and fairness checks with cryptographic artifact hashes (`SHA-256`) and UTC timestamps for governance traceability.

```bash
python scripts/m11_l3_logging_fairness.py
```
* **Outputs:**
  * `logs/fairness_audit.log` — Human-readable execution log.
  * `logs/fairness_audit_trail.jsonl` — Append-only JSON Lines audit record containing model/dataset checksums, sample size, segmented results, and pass/fail status.

---

## 🛠️ Customization & Extension Ideas
* **Adjust Thresholds:** Edit `THRESHOLDS` in `scripts/m11_l2_fairness_check.py` to test stricter or more relaxed compliance policies.
* **Bias Mitigation:** Experiment with pre-processing (reweighing) or post-processing (group-specific decision thresholds) in `setup_data_and_model.py` and re-run Labs 1–3 to verify improvement in the audit log.
* **CI/CD Integration:** Exit with a non-zero status code in `m11_l2_fairness_check.py` when `overall_status == "FAIL"` to block biased models from deployment.