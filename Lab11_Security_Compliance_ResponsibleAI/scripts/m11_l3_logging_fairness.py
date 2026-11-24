import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import joblib
import pandas as pd

from m11_l1_segmented_evaluation import compute_segment_metrics
from m11_l2_fairness_check import evaluate_fairness

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "validation_data.csv"
MODEL_PATH = BASE_DIR / "models" / "dummy_model.joblib"
LOGS_DIR = BASE_DIR / "logs"

FEATURES = ["income", "credit_score", "debt_to_income"]


def sha256_file(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("ResponsibleAIAudit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(LOGS_DIR / "fairness_audit.log")

    logger.info("Starting Responsible AI fairness audit run...")

    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    df["y_pred"] = model.predict(df[FEATURES])

    segment_df = compute_segment_metrics(df)
    fairness_report = evaluate_fairness(df)

    audit_record = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_artifact": MODEL_PATH.name,
        "model_sha256": sha256_file(MODEL_PATH),
        "dataset_artifact": DATA_PATH.name,
        "dataset_sha256": sha256_file(DATA_PATH),
        "sample_size": len(df),
        "segmented_metrics": segment_df.to_dict(orient="records"),
        "fairness_evaluation": fairness_report,
    }

    jsonl_path = LOGS_DIR / "fairness_audit_trail.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(audit_record) + "\n")

    logger.info(
        "Audit complete | Status=%s | DI_Ratio=%.4f | DP_Diff=%.4f | EO_Diff=%.4f",
        fairness_report["overall_status"],
        fairness_report["checks"]["disparate_impact_ratio"]["value"],
        fairness_report["checks"]["demographic_parity_diff"]["value"],
        fairness_report["checks"]["equal_opportunity_diff"]["value"],
    )
    logger.info("Appended structured audit entry -> %s", jsonl_path)


if __name__ == "__main__":
    main()