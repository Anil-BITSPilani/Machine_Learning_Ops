import numpy as np
import pandas as pd
from src.drift_monitor import run_drift_monitoring


def test_drift_monitor_derives_temporal_features_for_custom_csv(tmp_path):
    rng = np.random.default_rng(42)
    ref_df = pd.DataFrame({
        "Amount": rng.uniform(1, 500, 200),
        "V14": rng.normal(0, 1, 200),
        "hour_sin": rng.uniform(-1, 1, 200),
        "hour_cos": rng.uniform(-1, 1, 200),
    })
    ref_path = tmp_path / "ref.parquet"
    ref_df.to_parquet(ref_path, index=False)

    # Custom raw production CSV has 'Time' and raw features, but lacks hour_sin / hour_cos
    raw_prod = pd.DataFrame({
        "Time": np.linspace(0, 86400, 200),
        "Amount": rng.uniform(1, 500, 200),
        "V14": rng.normal(0, 1, 200),
    })
    prod_csv = tmp_path / "custom_prod.csv"
    raw_prod.to_csv(prod_csv, index=False)

    report_path = tmp_path / "drift_out.json"
    report = run_drift_monitoring(
        ref_path=str(ref_path),
        prod_path=str(prod_csv),
        output_report=str(report_path),
    )
    monitored = {item["feature"] for item in report["feature_metrics"]}
    assert "hour_sin" in monitored
    assert "hour_cos" in monitored