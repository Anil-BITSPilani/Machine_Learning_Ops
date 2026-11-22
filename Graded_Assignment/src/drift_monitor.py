import argparse
import json
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from src.data_pipeline import add_temporal_features, fetch_and_cache_data, temporal_train_val_test_split

# Explicit Drift Alert Thresholds (Industry Standard)
PSI_WARNING_THRESHOLD = 0.10    # Moderate distribution shift -> Ops warning alert
PSI_CRITICAL_THRESHOLD = 0.25   # Significant distribution shift -> Automated retraining trigger
KS_ALPHA = 0.01                 # Statistical significance level for KS two-sample test
KS_STAT_THRESHOLD = 0.10        # Minimum practical effect size (KS D-statistic)

MONITORED_FEATURES = ["Amount", "V14", "V17", "V12", "V10", "V4", "hour_sin", "hour_cos"]


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index (PSI) using decile breakpoints from reference data."""
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    exp_counts, _ = np.histogram(expected, bins=breakpoints)
    act_counts, _ = np.histogram(actual, bins=breakpoints)

    exp_percents = np.clip(exp_counts / len(expected), 1e-6, None)
    act_percents = np.clip(act_counts / len(actual), 1e-6, None)

    return float(np.sum((act_percents - exp_percents) * np.log(act_percents / exp_percents)))

def classify_drift_status(psi: float, ks_stat: float, ks_pvalue: float) -> str:
    """Evaluate feature drift against explicit PSI and KS alert thresholds."""
    if psi >= PSI_CRITICAL_THRESHOLD:
        return "CRITICAL_DRIFT (Trigger Retraining)"
    if psi >= PSI_WARNING_THRESHOLD or (ks_stat >= KS_STAT_THRESHOLD and ks_pvalue < KS_ALPHA):
        return "WARNING (Investigate Shift)"
    return "OK (Within Bounds)"


def load_production_sample(prod_path: str | None, simulate_shift: bool = False) -> pd.DataFrame:
    """Load production batch file if provided; otherwise extract held-out temporal test window."""
    if prod_path is not None:
        if not os.path.exists(prod_path):
            raise FileNotFoundError(f"Production batch file not found: {prod_path}")
        print(f"Loading production sample from {prod_path}...")
        prod_df = pd.read_parquet(prod_path) if prod_path.endswith(".parquet") else pd.read_csv(prod_path)
        if "Time" in prod_df.columns and not {"hour_sin", "hour_cos"}.issubset(prod_df.columns):
            prod_df = add_temporal_features(prod_df)
    else:
        print("No external production log provided; sampling latest temporal test window as production batch...")
        df = fetch_and_cache_data()
        _, _, (X_test, _) = temporal_train_val_test_split(df)
        prod_df = X_test.tail(10000).copy().reset_index(drop=True)

    if simulate_shift:
        print("[SIMULATION] Injecting synthetic covariate shift into 'Amount' and 'V14'...")
        if "Amount" in prod_df.columns:
            prod_df["Amount"] = prod_df["Amount"] * 1.85 + 25.0
        if "V14" in prod_df.columns:
            prod_df["V14"] = prod_df["V14"] - 1.50

    return prod_df


def run_drift_monitoring(
    ref_path: str = "artifacts/train_reference.parquet",
    prod_path: str | None = None,
    output_report: str = "artifacts/drift_report.json",
    simulate_shift: bool = False,
) -> dict:
    if not os.path.exists(ref_path):
        raise FileNotFoundError(
            f"Reference baseline not found at {ref_path}. Run `python -m src.train` first."
        )

    ref_df = pd.read_parquet(ref_path)
    prod_df = load_production_sample(prod_path=prod_path, simulate_shift=simulate_shift)

    features = [c for c in MONITORED_FEATURES if c in ref_df.columns and c in prod_df.columns]
    if not features:
        raise ValueError("No overlapping monitored features found between reference and production datasets.")

    results = []
    critical_features = []
    warning_features = []

    print("\n=== PRODUCTION DATA DRIFT REPORT (train_reference.parquet vs. Production Sample) ===")
    print(f"Reference Rows : {len(ref_df):,d} | Production Rows : {len(prod_df):,d}")
    print(
        f"Alert Rules    : PSI >= {PSI_CRITICAL_THRESHOLD:.2f} -> CRITICAL (Retrain) | "
        f"PSI >= {PSI_WARNING_THRESHOLD:.2f} or (KS >= {KS_STAT_THRESHOLD:.2f} & p < {KS_ALPHA}) -> WARNING"
    )
    print("-" * 92)
    print(f"{'Feature':<10} | {'PSI':>8} | {'KS Stat':>8} | {'KS p-value':>12} | {'Status'}")
    print("-" * 92)

    for col in features:
        ref_vals = ref_df[col].dropna().to_numpy(dtype=float)
        prod_vals = prod_df[col].dropna().to_numpy(dtype=float)

        ref_vals = ref_vals[np.isfinite(ref_vals)]
        prod_vals = prod_vals[np.isfinite(prod_vals)]

        if len(prod_vals) == 0:
            raise ValueError(f"Production column '{col}' contains zero valid finite observations.")

        psi = calculate_psi(ref_vals, prod_vals, bins=10)
        ks_stat, ks_p = ks_2samp(ref_vals, prod_vals)
        status = classify_drift_status(psi, float(ks_stat), float(ks_p))

        if status.startswith("CRITICAL"):
            critical_features.append(col)
        elif status.startswith("WARNING"):
            warning_features.append(col)

        print(f"{col:<10} | {psi:8.4f} | {ks_stat:8.4f} | {ks_p:12.2e} | {status}")
        results.append(
            {
                "feature": col,
                "psi": round(psi, 4),
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": float(f"{ks_p:.4e}"),
                "status": status,
            }
        )

    print("-" * 92)
    overall_action = (
        "TRIGGER_RETRAINING"
        if critical_features
        else ("SEND_OPS_WARNING" if warning_features else "NO_ACTION_REQUIRED")
    )
    print(f"Overall Decision: {overall_action}")
    if critical_features:
        print(f"Critical Drift Features : {', '.join(critical_features)}")
    if warning_features:
        print(f"Warning Drift Features  : {', '.join(warning_features)}")

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reference_artifact": ref_path,
        "reference_sample_size": len(ref_df),
        "production_sample_size": len(prod_df),
        "thresholds": {
            "psi_warning": PSI_WARNING_THRESHOLD,
            "psi_critical_retrain": PSI_CRITICAL_THRESHOLD,
            "ks_stat_threshold": KS_STAT_THRESHOLD,
            "ks_alpha": KS_ALPHA,
        },
        "overall_action": overall_action,
        "critical_features": critical_features,
        "warning_features": warning_features,
        "feature_metrics": results,
    }

    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved JSON drift audit report to {output_report}\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PSI and KS drift monitoring against train_reference.parquet")
    parser.add_argument("--ref", type=str, default="artifacts/train_reference.parquet", help="Path to reference parquet")
    parser.add_argument("--prod", type=str, default=None, help="Optional path to production batch CSV/Parquet")
    parser.add_argument("--output", type=str, default="artifacts/drift_report.json", help="Output JSON report path")
    parser.add_argument(
        "--simulate-shift",
        action="store_true",
        help="Inject synthetic drift into production sample to verify alerting thresholds",
    )
    args = parser.parse_args()

    run_drift_monitoring(
        ref_path=args.ref,
        prod_path=args.prod,
        output_report=args.output,
        simulate_shift=args.simulate_shift,
    )