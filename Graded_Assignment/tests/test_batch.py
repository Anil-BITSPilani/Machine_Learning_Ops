import pandas as pd
import pytest
from src.serving.predict_batch import run_batch_inference


def _make_sample_df(n: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(n):
        row = {"Time": 400.0 + i * 10, "Amount": 120.0 + i * 50}
        for v in range(1, 29):
            row[f"V{v}"] = 0.05 * i
        rows.append(row)
    return pd.DataFrame(rows)


def test_batch_inference_generates_ids_and_predictions(tmp_path):
    in_csv = tmp_path / "input.csv"
    out_csv = tmp_path / "output.csv"
    _make_sample_df(5).to_csv(in_csv, index=False)

    res = run_batch_inference(str(in_csv), str(out_csv))
    assert out_csv.exists()
    assert len(res) == 5
    assert list(res.columns) == [
        "transaction_id",
        "fraud_probability",
        "is_fraud",
        "threshold_used",
        "scoring_mode",
    ]
    assert res["transaction_id"].iloc[0] == "batch-rec-000000"
    assert (res["scoring_mode"] == "ml_model").all()


def test_batch_inference_fallback_when_model_missing(tmp_path):
    in_csv = tmp_path / "input_fallback.csv"
    out_csv = tmp_path / "output_fallback.csv"
    df = _make_sample_df(2)
    df.loc[1, "Amount"] = 50000.0
    df.to_csv(in_csv, index=False)

    res = run_batch_inference(
        str(in_csv),
        str(out_csv),
        model_path=str(tmp_path / "nonexistent.joblib"),
    )
    assert (res["scoring_mode"] == "fallback_rule").all()
    assert bool(res["is_fraud"].iloc[1]) is True


def test_batch_inference_rejects_missing_columns(tmp_path):
    in_csv = tmp_path / "bad.csv"
    out_csv = tmp_path / "bad_out.csv"
    pd.DataFrame({"Time": [10.0], "Amount": [25.0]}).to_csv(in_csv, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        run_batch_inference(str(in_csv), str(out_csv))

import numpy as np


def test_batch_inference_imputes_csv_nan_values(tmp_path):
    in_csv = tmp_path / "nan_input.csv"
    out_csv = tmp_path / "nan_output.csv"
    df = _make_sample_df(3)
    df.loc[0, "Amount"] = np.nan
    df.loc[1, "V14"] = np.nan
    df.to_csv(in_csv, index=False)

    res = run_batch_inference(str(in_csv), str(out_csv))
    assert len(res) == 3
    assert (res["scoring_mode"] == "ml_model").all()
    assert np.all(np.isfinite(res["fraud_probability"]))


def test_batch_inference_rejects_infinity_values(tmp_path):
    in_csv = tmp_path / "inf_input.csv"
    out_csv = tmp_path / "inf_output.csv"
    df = _make_sample_df(2)
    df.loc[0, "V4"] = np.inf
    df.to_csv(in_csv, index=False)

    with pytest.raises(ValueError, match="infinite"):
        run_batch_inference(str(in_csv), str(out_csv))
