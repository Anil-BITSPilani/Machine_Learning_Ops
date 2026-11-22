import json
import logging
import os
import time
import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.data_pipeline import FEATURE_COLS, fetch_and_cache_data, temporal_train_val_test_split
from src.evaluate import evaluate_predictions, find_optimal_threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("model_training")


def main():
    os.makedirs("artifacts", exist_ok=True)

    # [Step 1/4] Load & Perform Strict Chronological 70/15/15 Split
    logger.info("[Step 1/4] Loading dataset and performing chronological 70/15/15 split...")
    t0 = time.perf_counter()
    df = fetch_and_cache_data()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = temporal_train_val_test_split(df)
    test_amounts = X_test["Amount"].to_numpy()
    logger.info(
        "[Step 1/4] Split complete in %.2fs | train=%d (fraud=%d) | val=%d (fraud=%d) | test=%d (fraud=%d)",
        time.perf_counter() - t0,
        len(y_train), int(y_train.sum()),
        len(y_val), int(y_val.sum()),
        len(y_test), int(y_test.sum()),
    )

    # [Step 2/4] Interpretable Baseline: Class-Weighted Logistic Regression
    logger.info("[Step 2/4] Fitting baseline class-weighted LogisticRegression...")
    t0 = time.perf_counter()
    baseline_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    baseline_pipe.fit(X_train, y_train)

    base_val_probs = baseline_pipe.predict_proba(X_val)[:, 1]
    base_thresh, base_cal = find_optimal_threshold(y_val, base_val_probs, min_precision=0.80)

    base_test_probs = baseline_pipe.predict_proba(X_test)[:, 1]
    base_metrics = evaluate_predictions(
        y_test,
        base_test_probs,
        threshold=base_thresh,
        amounts=test_amounts,
    )
    logger.info(
        "[Step 2/4] Baseline complete in %.2fs | PR-AUC=%.4f | Net Savings=$%.2f",
        time.perf_counter() - t0,
        base_metrics["pr_auc"],
        base_metrics["cost_metrics"]["net_fraud_savings"],
    )

    # [Step 3/4] Primary Model: Regularized Cost-Sensitive LightGBM
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    raw_imbalance_ratio = neg_count / max(pos_count, 1)
    # Dampened square-root weighting avoids probability saturation under extreme 518:1 skew
    dampened_pos_weight = float(np.sqrt(raw_imbalance_ratio))

    logger.info(
        "[Step 3/4] Fitting cost-sensitive LightGBM (raw_ratio=%.2f, dampened_scale_pos_weight=%.2f)...",
        raw_imbalance_ratio,
        dampened_pos_weight,
    )
    t0 = time.perf_counter()
    lgbm_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("clf", LGBMClassifier(
            n_estimators=250,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            scale_pos_weight=dampened_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )),
    ])
    lgbm_pipe.fit(X_train, y_train)

    lgbm_val_probs = lgbm_pipe.predict_proba(X_val)[:, 1]
    tuned_thresh, lgbm_cal = find_optimal_threshold(y_val, lgbm_val_probs, min_precision=0.80)

    lgbm_test_probs = lgbm_pipe.predict_proba(X_test)[:, 1]
    lgbm_metrics = evaluate_predictions(
        y_test,
        lgbm_test_probs,
        threshold=tuned_thresh,
        amounts=test_amounts,
    )
    logger.info(
        "[Step 3/4] LightGBM complete in %.2fs | PR-AUC=%.4f | Net Savings=$%.2f",
        time.perf_counter() - t0,
        lgbm_metrics["pr_auc"],
        lgbm_metrics["cost_metrics"]["net_fraud_savings"],
    )

    print("\n=== VALIDATION THRESHOLD CALIBRATION SUMMARY ===")
    print("Baseline (LogisticRegression) Calibration:", json.dumps(base_cal, indent=2))
    print("Primary  (LightGBM)            Calibration:", json.dumps(lgbm_cal, indent=2))

    print("\n=== TEST SET COMPARISON (STRICT TEMPORAL SPLIT + COST ANALYSIS) ===")
    print("Baseline (LogisticRegression):", json.dumps(base_metrics, indent=2))
    print("Primary  (LightGBM)          :", json.dumps(lgbm_metrics, indent=2))

    # [Step 4/4] Serialize Production Pipeline, Audit Metadata & Drift Reference Baseline
    logger.info("[Step 4/4] Serializing pipeline, metadata, and drift reference baseline...")
    t0 = time.perf_counter()
    artifact_path = "artifacts/fraud_pipeline.joblib"
    joblib.dump(lgbm_pipe, artifact_path)

    ref_stats = {
        "model_version": "lgbm-fraud-v1.0.0",
        "raw_imbalance_ratio": round(raw_imbalance_ratio, 2),
        "scale_pos_weight": round(dampened_pos_weight, 2),
        "decision_threshold": tuned_thresh,
        "threshold_calibration": {
            "primary_lgbm": lgbm_cal,
            "baseline_logreg": base_cal,
        },
        "fallback_amount_p995": float(np.percentile(X_train["Amount"], 99.5)),
        "feature_cols": FEATURE_COLS,
        "test_metrics": lgbm_metrics,
        "baseline_metrics": base_metrics,
    }
    meta_path = "artifacts/metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(ref_stats, f, indent=2)

    ref_sample_path = "artifacts/train_reference.parquet"
    X_train.sample(n=10000, random_state=42).to_parquet(ref_sample_path, index=False)

    logger.info(
        "[Step 4/4] Serialization complete in %.2fs | pipeline=%s | metadata=%s | reference=%s",
        time.perf_counter() - t0,
        artifact_path,
        meta_path,
        ref_sample_path,
    )


if __name__ == "__main__":
    main()