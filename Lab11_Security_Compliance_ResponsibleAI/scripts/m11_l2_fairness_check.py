import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "validation_data.csv"
MODEL_PATH = BASE_DIR / "models" / "dummy_model.joblib"
LOGS_DIR = BASE_DIR / "logs"

FEATURES = ["income", "credit_score", "debt_to_income"]
TARGET = "approved"
SENSITIVE_COL = "group"

THRESHOLDS = {
    "min_disparate_impact_ratio": 0.80,   # Four-Fifths (80%) Rule
    "max_demographic_parity_diff": 0.10,  # Max selection rate gap
    "max_equal_opportunity_diff": 0.10,   # Max TPR gap
}


def evaluate_fairness(df: pd.DataFrame) -> dict:
    group_stats = {}
    for group_name, sub_df in df.groupby(SENSITIVE_COL):
        y_true = sub_df[TARGET]
        y_pred = sub_df["y_pred"]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        selection_rate = float(y_pred.mean())
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        group_stats[group_name] = {
            "selection_rate": round(selection_rate, 4),
            "tpr": round(tpr, 4),
        }

    sel_rates = [v["selection_rate"] for v in group_stats.values()]
    tprs = [v["tpr"] for v in group_stats.values()]

    di_ratio = round(min(sel_rates) / max(sel_rates), 4) if max(sel_rates) > 0 else 0.0
    dp_diff = round(max(sel_rates) - min(sel_rates), 4)
    eo_diff = round(max(tprs) - min(tprs), 4)

    checks = {
        "disparate_impact_ratio": {
            "value": di_ratio,
            "threshold": f">= {THRESHOLDS['min_disparate_impact_ratio']}",
            "passed": bool(di_ratio >= THRESHOLDS["min_disparate_impact_ratio"]),
        },
        "demographic_parity_diff": {
            "value": dp_diff,
            "threshold": f"<= {THRESHOLDS['max_demographic_parity_diff']}",
            "passed": bool(dp_diff <= THRESHOLDS["max_demographic_parity_diff"]),
        },
        "equal_opportunity_diff": {
            "value": eo_diff,
            "threshold": f"<= {THRESHOLDS['max_equal_opportunity_diff']}",
            "passed": bool(eo_diff <= THRESHOLDS["max_equal_opportunity_diff"]),
        },
    }

    overall_passed = all(c["passed"] for c in checks.values())

    return {
        "overall_status": "PASS" if overall_passed else "FAIL",
        "group_metrics": group_stats,
        "checks": checks,
    }


def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    df["y_pred"] = model.predict(df[FEATURES])

    report = evaluate_fairness(df)

    print("\n=== Automated Fairness Check Report ===")
    print(f"Overall Status: {report['overall_status']}")
    for check_name, details in report["checks"].items():
        status = "PASS" if details["passed"] else "FAIL"
        print(
            f"- {check_name}: {details['value']} "
            f"(Rule: {details['threshold']}) -> [{status}]"
        )

    out_path = LOGS_DIR / "l2_fairness_check_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved fairness report -> {out_path}")


if __name__ == "__main__":
    main()