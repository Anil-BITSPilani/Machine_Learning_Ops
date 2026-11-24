from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "validation_data.csv"
MODEL_PATH = BASE_DIR / "models" / "dummy_model.joblib"
LOGS_DIR = BASE_DIR / "logs"

FEATURES = ["income", "credit_score", "debt_to_income"]
TARGET = "approved"
SENSITIVE_COL = "group"


def compute_segment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    segments = [("Overall", df)] + list(df.groupby(SENSITIVE_COL))

    for segment_name, sub_df in segments:
        y_true = sub_df[TARGET]
        y_pred = sub_df["y_pred"]

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        selection_rate = float(y_pred.mean())

        rows.append(
            {
                "segment": segment_name,
                "count": len(sub_df),
                "accuracy": round(accuracy_score(y_true, y_pred), 4),
                "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
                "tpr_recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
                "fpr": round(fpr, 4),
                "selection_rate": round(selection_rate, 4),
            }
        )

    return pd.DataFrame(rows)


def save_chart(metrics_df: pd.DataFrame, output_path: Path) -> None:
    group_df = metrics_df[metrics_df["segment"] != "Overall"].set_index("segment")
    plot_cols = ["accuracy", "tpr_recall", "fpr", "selection_rate"]

    ax = group_df[plot_cols].plot(kind="bar", figsize=(8, 5), rot=0)
    ax.set_title("Segmented Model Performance & Selection Rates")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    df["y_pred"] = model.predict(df[FEATURES])

    metrics_df = compute_segment_metrics(df)
    print("\n=== Segmented Evaluation Results ===")
    print(metrics_df.to_string(index=False))

    csv_out = LOGS_DIR / "l1_segmented_metrics.csv"
    chart_out = LOGS_DIR / "l1_segmented_comparison.png"

    metrics_df.to_csv(csv_out, index=False)
    save_chart(metrics_df, chart_out)

    print(f"\nSaved metrics table -> {csv_out}")
    print(f"Saved visualization -> {chart_out}")


if __name__ == "__main__":
    main()