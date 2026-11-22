import argparse
import json
import logging
import os
import time
import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("batch_inference")

DEFAULT_FEATURE_COLS = ["Amount", "hour_sin", "hour_cos"] + [f"V{i}" for i in range(1, 29)]
RAW_REQUIRED_COLS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


def prepare_batch_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Validate required schema, reject non-finite (+/-inf) or invalid boundary values,
    preserve genuine NaN feature nulls for SimpleImputer, and compute cyclical time features.
    """
    missing = [c for c in RAW_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input batch is missing required columns: {missing}")

    out = df.copy()
    for col in RAW_REQUIRED_COLS:
        try:
            out[col] = pd.to_numeric(out[col], errors="raise").astype(float)
        except Exception as exc:
            raise ValueError(f"Column '{col}' contains non-numeric values: {exc}") from exc

        inf_count = int(np.isinf(out[col].to_numpy()).sum())
        if inf_count > 0:
            raise ValueError(
                f"Column '{col}' contains {inf_count} infinite (+/-inf) value(s); "
                "non-finite numbers are not permitted in batch inputs."
            )

    if out["Time"].isna().any() or (out["Time"] < 0).any():
        raise ValueError("Column 'Time' must be non-null and >= 0 across all batch records.")

    valid_amounts = out["Amount"].dropna()
    if (valid_amounts < 0).any():
        raise ValueError("Column 'Amount' contains negative values (< 0), which are invalid.")

    nan_cells = int(out[["Amount"] + [f"V{i}" for i in range(1, 29)]].isna().sum().sum())
    if nan_cells > 0:
        logger.info("Detected %d missing (NaN) feature cell(s); delegating to pipeline SimpleImputer.", nan_cells)

    hour_of_day = (out["Time"] / 3600.0) % 24.0
    out["hour_sin"] = np.sin(2.0 * np.pi * hour_of_day / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour_of_day / 24.0)
    return out[feature_cols].astype(float)


def run_batch_inference(
    input_path: str,
    output_path: str,
    model_path: str = "artifacts/fraud_pipeline.joblib",
    meta_path: str = "artifacts/metadata.json",
) -> pd.DataFrame:
    start_total = time.perf_counter()
    logger.info("Starting batch fraud scoring job | input=%s | output=%s", input_path, output_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as exc:
            logger.warning("Failed to parse metadata at %s (%s); using defaults.", meta_path, exc)

    threshold = float(meta.get("decision_threshold", 0.50))
    fallback_cutoff = float(meta.get("fallback_amount_p995", 2500.0))
    feature_cols = meta.get("feature_cols", DEFAULT_FEATURE_COLS)
    model_version = meta.get("model_version", "unknown")

    df_in = pd.read_csv(input_path)
    n_records = len(df_in)
    logger.info("Loaded %d records from %s", n_records, input_path)

    if n_records == 0:
        raise ValueError("Input CSV contains 0 records; aborting batch job.")

    # Fail fast on schema, non-finite (+/-inf), or domain boundary violations
    X_batch = prepare_batch_features(df_in, feature_cols)

    if "transaction_id" in df_in.columns:
        record_ids = df_in["transaction_id"].astype(str)
    else:
        record_ids = pd.Series([f"batch-rec-{i:06d}" for i in range(n_records)], index=df_in.index)

    # Vectorized ML inference (only genuine model loading/scoring errors trigger fallback)
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact missing at {model_path}")

        model = joblib.load(model_path)

        t_infer = time.perf_counter()
        probs = np.asarray(model.predict_proba(X_batch)[:, 1], dtype=float)
        if not np.all(np.isfinite(probs)):
            raise RuntimeError("Model returned non-finite prediction probabilities.")
        infer_sec = time.perf_counter() - t_infer

        preds = (probs >= threshold).astype(bool)
        scoring_mode = "ml_model"
        logger.info(
            "Vectorized ML inference succeeded | model=%s | records=%d | infer_time=%.4fs (%.0f records/sec)",
            model_version,
            n_records,
            infer_sec,
            n_records / max(infer_sec, 1e-6),
        )
    except Exception as exc:
        logger.warning("ML pipeline failed (%s: %s); engaging vectorized rule fallback.", type(exc).__name__, exc)
        safe_amounts = df_in["Amount"].fillna(0.0).astype(float).to_numpy()
        preds = safe_amounts >= fallback_cutoff
        probs = np.where(preds, 0.95, 0.01)
        scoring_mode = "fallback_rule"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_out = pd.DataFrame(
        {
            "transaction_id": record_ids,
            "fraud_probability": np.round(probs, 6),
            "is_fraud": preds,
            "threshold_used": threshold,
            "scoring_mode": scoring_mode,
        }
    )
    df_out.to_csv(output_path, index=False)

    total_sec = time.perf_counter() - start_total
    flagged_count = int(df_out["is_fraud"].sum())
    logger.info(
        "Batch job completed | output=%s | total_records=%d | flagged_fraud=%d (%.2f%%) | end_to_end=%.4fs (%.0f records/sec)",
        output_path,
        n_records,
        flagged_count,
        (flagged_count / n_records) * 100.0,
        total_sec,
        n_records / max(total_sec, 1e-6),
    )
    return df_out


def main():
    parser = argparse.ArgumentParser(description="Batch CLI fraud scoring script")
    parser.add_argument("input_csv", type=str, help="Path to unseen input transactions CSV")
    parser.add_argument("output_csv", type=str, help="Path to write batch predictions CSV")
    parser.add_argument("--model", type=str, default="artifacts/fraud_pipeline.joblib", help="Serialized model path")
    parser.add_argument("--meta", type=str, default="artifacts/metadata.json", help="Model metadata JSON path")
    args = parser.parse_args()

    run_batch_inference(
        input_path=args.input_csv,
        output_path=args.output_csv,
        model_path=args.model,
        meta_path=args.meta,
    )


if __name__ == "__main__":
    main()