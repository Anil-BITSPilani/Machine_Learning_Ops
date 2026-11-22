import logging
import numpy as np
import pytest
from fastapi.testclient import TestClient
import src.serving.app as app_module
from src.serving.app import app, state

SAMPLE_TX = {
    "transaction_id": "tx-test-001",
    "Time": 406.0,
    "Amount": 149.62,
    "V1": -1.3598, "V2": -0.0727, "V3": 2.5363, "V4": 1.3781,
    "V5": -0.3383, "V6": 0.4623, "V7": 0.2395, "V8": 0.0986,
    "V9": 0.3637, "V10": 0.0907, "V11": -0.5516, "V12": -0.6178,
    "V13": -0.9913, "V14": -0.3111, "V15": 1.4681, "V16": -0.4704,
    "V17": 0.2079, "V18": 0.0257, "V19": 0.4039, "V20": 0.2514,
    "V21": -0.0183, "V22": 0.2778, "V23": -0.1104, "V24": 0.0669,
    "V25": 0.1285, "V26": -0.1891, "V27": 0.1335, "V28": -0.0210,
}


# 1. Existing Happy-Path Health & Real-Model Inference
def test_health_and_valid_prediction():
    with TestClient(app) as client:
        h = client.get("/health")
        assert h.status_code == 200
        assert h.json()["model_loaded"] is True

        r = client.post("/predict", json=SAMPLE_TX)
        assert r.status_code == 200
        body = r.json()
        assert body["scoring_mode"] == "ml_model"
        assert 0.0 <= body["fraud_probability"] <= 1.0


# 2. Exact Probability Extraction Bug Regression Test
def test_exact_positive_class_probability_extraction():
    """Verify /predict extracts positive fraud probability [:, 1] rather than class 0."""
    class StubBinaryModel:
        def predict_proba(self, X):
            return np.array([[0.13, 0.87]])

    with TestClient(app) as client:
        saved_model = state["model"]
        saved_meta = dict(state["meta"])
        state["model"] = StubBinaryModel()
        state["meta"] = dict(saved_meta, decision_threshold=0.50)
        try:
            r = client.post("/predict", json=SAMPLE_TX)
            assert r.status_code == 200
            body = r.json()
            assert body["scoring_mode"] == "ml_model"
            assert body["fraud_probability"] == pytest.approx(0.87, abs=1e-6)
            assert body["is_fraud"] is True
        finally:
            state["model"] = saved_model
            state["meta"] = saved_meta


# 3. Missing & Null Values Handled by Pipeline's SimpleImputer
def test_missing_and_null_features_imputed_by_pipeline():
    with TestClient(app) as client:
        sparse_tx = {
            "transaction_id": "tx-sparse-001",
            "Time": 850.0,
            "Amount": None,
            "V1": None,
            "V2": -0.0727,
            "V3": 2.5363,
            "V4": 1.3781,
        }
        r = client.post("/predict", json=sparse_tx)
        assert r.status_code == 200
        body = r.json()
        assert body["scoring_mode"] == "ml_model"
        assert 0.0 <= body["fraud_probability"] <= 1.0


# 4. NaN and Infinity Payloads Rejected
@pytest.mark.parametrize("bad_token", ["NaN", "Infinity", "-Infinity"])
def test_nan_and_infinity_payloads_rejected(bad_token):
    with TestClient(app) as client:
        raw_json = (
            '{"transaction_id": "tx-nan-001", "Time": 400.0, "Amount": 50.0, '
            f'"V1": {bad_token}}}'
        )
        r = client.post(
            "/predict",
            content=raw_json,
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422


# 5. Negative Amount Rejected
def test_reject_malformed_and_negative_amount():
    with TestClient(app) as client:
        bad_tx = dict(SAMPLE_TX, Amount=-50.0)
        r = client.post("/predict", json=bad_tx)
        assert r.status_code == 422


# 6. Extreme Numeric Outliers Handled Cleanly
def test_extreme_outlier_handling():
    with TestClient(app) as client:
        outlier_tx = dict(SAMPLE_TX, Amount=999999.99, V14=-45.0)
        r = client.post("/predict", json=outlier_tx)
        assert r.status_code == 200
        assert r.json()["scoring_mode"] == "ml_model"


# 7. Model Loading Failure at Startup
def test_model_loading_failure_sets_degraded_health_and_triggers_fallback(monkeypatch, caplog):
    # Seed stale state prior to startup to verify lifespan resets both model and meta
    state["model"] = object()
    state["meta"] = {"model_version": "stale-v0.0.1", "decision_threshold": 0.99}

    monkeypatch.setattr(app_module, "MODEL_PATH", "artifacts/nonexistent_model.joblib")
    monkeypatch.setattr(app_module, "META_PATH", "artifacts/nonexistent_meta.json")

    with caplog.at_level(logging.ERROR, logger="fraud_api"):
        with TestClient(app) as client:
            h = client.get("/health")
            assert h.status_code == 200
            assert h.json() == {
                "status": "degraded",
                "model_loaded": False,
                "model_version": "unavailable",
            }
            assert state["meta"] == {}
            assert "Startup degraded" in caplog.text

            r = client.post("/predict", json=SAMPLE_TX)
            assert r.status_code == 200
            assert r.json()["scoring_mode"] == "fallback_rule"


# 8. Runtime Fallback Behavior & Exception Logging
def test_runtime_fallback_behavior_and_exception_logging(caplog):
    class BrokenModel:
        def predict_proba(self, X):
            raise ValueError("Corrupted booster tree leaf")

    with TestClient(app) as client:
        saved_model = state["model"]
        state["model"] = BrokenModel()
        try:
            with caplog.at_level(logging.ERROR, logger="fraud_api"):
                high_amt_tx = dict(SAMPLE_TX, transaction_id="tx-fb-high", Amount=50000.0)
                r_high = client.post("/predict", json=high_amt_tx)
                assert r_high.status_code == 200
                assert r_high.json()["scoring_mode"] == "fallback_rule"
                assert r_high.json()["is_fraud"] is True
                assert r_high.json()["fraud_probability"] == pytest.approx(0.95)
                assert "ML inference failed for transaction_id=tx-fb-high" in caplog.text
                assert "Corrupted booster tree leaf" in caplog.text

                low_amt_tx = dict(SAMPLE_TX, transaction_id="tx-fb-low", Amount=25.0)
                r_low = client.post("/predict", json=low_amt_tx)
                assert r_low.status_code == 200
                assert r_low.json()["scoring_mode"] == "fallback_rule"
                assert r_low.json()["is_fraud"] is False
                assert r_low.json()["fraud_probability"] == pytest.approx(0.01)
        finally:
            state["model"] = saved_model


# 9. Fallback Handles Missing Amount Safely
def test_fallback_handles_missing_amount():
    with TestClient(app) as client:
        saved_model = state["model"]
        state["model"] = None
        try:
            sparse_tx = {"transaction_id": "tx-fallback-null", "Time": 500.0}
            r = client.post("/predict", json=sparse_tx)
            assert r.status_code == 200
            assert r.json()["scoring_mode"] == "fallback_rule"
            assert r.json()["is_fraud"] is False
        finally:
            state["model"] = saved_model


# 10. Malformed Extra Fields Rejected
def test_reject_unknown_extra_fields():
    with TestClient(app) as client:
        extra_field_tx = dict(SAMPLE_TX, V29=0.42, unexpected_key="malformed")
        r = client.post("/predict", json=extra_field_tx)
        assert r.status_code == 422
        assert any(err["type"] == "extra_forbidden" for err in r.json()["detail"])

def test_corrupted_joblib_artifact_leaves_api_in_degraded_fallback(tmp_path, monkeypatch, caplog):
    bad_model = tmp_path / "corrupt.joblib"
    bad_model.write_bytes(b"not-a-valid-pickle-or-joblib-stream")
    valid_meta = tmp_path / "meta.json"
    valid_meta.write_text('{"model_version": "v1", "decision_threshold": 0.5}', encoding="utf-8")

    monkeypatch.setattr(app_module, "MODEL_PATH", str(bad_model))
    monkeypatch.setattr(app_module, "META_PATH", str(valid_meta))

    with caplog.at_level(logging.ERROR, logger="fraud_api"):
        with TestClient(app) as client:
            h = client.get("/health")
            assert h.status_code == 200
            assert h.json()["status"] == "degraded"
            assert h.json()["model_loaded"] is False
            assert "failed to deserialize model artifact" in caplog.text

            r = client.post("/predict", json=SAMPLE_TX)
            assert r.status_code == 200
            assert r.json()["scoring_mode"] == "fallback_rule"


def test_corrupted_metadata_json_leaves_api_in_degraded_fallback(tmp_path, monkeypatch, caplog):
    bad_meta = tmp_path / "corrupt_meta.json"
    bad_meta.write_text("{invalid-json:::", encoding="utf-8")

    monkeypatch.setattr(app_module, "META_PATH", str(bad_meta))

    with caplog.at_level(logging.ERROR, logger="fraud_api"):
        with TestClient(app) as client:
            h = client.get("/health")
            assert h.status_code == 200
            assert h.json()["status"] == "degraded"
            assert h.json()["model_loaded"] is False
            assert "failed to parse metadata JSON" in caplog.text

            r = client.post("/predict", json=SAMPLE_TX)
            assert r.status_code == 200
            assert r.json()["scoring_mode"] == "fallback_rule"