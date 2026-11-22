import logging
import numpy as np
import pytest
from src.evaluate import evaluate_predictions, find_optimal_threshold


def test_find_optimal_threshold_satisfies_precision_constraint():
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_probs = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.82, 0.70, 0.85, 0.90, 0.95])

    thresh, info = find_optimal_threshold(y_true, y_probs, min_precision=0.80)
    assert info["status"] == "constraint_satisfied"
    assert info["val_precision"] >= 0.80
    assert thresh == pytest.approx(0.70, abs=1e-6)


def test_find_optimal_threshold_explicit_fallback_when_constraint_unmet(caplog):
    # Max achievable precision at non-zero recall is 0.50 (< 0.80)
    y_true = np.array([0, 0, 1, 1])
    y_probs = np.array([0.60, 0.60, 0.60, 0.60])

    with caplog.at_level(logging.WARNING, logger="model_evaluation"):
        thresh, info = find_optimal_threshold(y_true, y_probs, min_precision=0.80)
        assert info["status"] == "fallback_max_f1"
        assert thresh == pytest.approx(0.60, abs=1e-6)
        assert "Threshold calibration fallback" in caplog.text


def test_evaluate_predictions_computes_cost_weighted_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_probs = np.array([0.10, 0.85, 0.90, 0.20])
    amounts = np.array([50.0, 40.0, 500.0, 200.0])

    res = evaluate_predictions(
        y_true,
        y_probs,
        threshold=0.50,
        amounts=amounts,
        alert_review_cost=10.0,
        chargeback_fee=25.0,
    )
    cost = res["cost_metrics"]
    assert cost["unmitigated_fraud_cost"] == pytest.approx(750.0)
    assert cost["fn_missed_fraud_cost"] == pytest.approx(225.0)
    assert cost["alert_review_cost"] == pytest.approx(20.0)
    assert cost["total_expected_cost"] == pytest.approx(245.0)
    assert cost["net_fraud_savings"] == pytest.approx(505.0)