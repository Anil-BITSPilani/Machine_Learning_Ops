import logging
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

logger = logging.getLogger("model_evaluation")

DEFAULT_ALERT_REVIEW_COST = 10.0     # Operational review + customer friction cost per flagged alert (FP or TP)
DEFAULT_CHARGEBACK_FEE = 25.0        # Fixed dispute/chargeback overhead per missed fraud (FN)
DEFAULT_FALLBACK_FRAUD_AMT = 125.0   # Used only if transaction Amount array is not provided


def find_optimal_threshold(y_true, y_probs, min_precision: float = 0.80) -> tuple[float, dict]:
    """
    Tune decision probability threshold on the validation set to maximize F1 subject to
    precision >= min_precision. Explicitly records whether the constraint was satisfied
    or which deterministic fallback policy was applied.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)

    best_constrained_thresh = None
    best_constrained_f1 = -1.0
    best_constrained_stats = {}

    best_unconstrained_thresh = None
    best_unconstrained_f1 = -1.0
    best_unconstrained_stats = {}

    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if (p + r) <= 0 or r <= 0:
            continue
        f1 = float(2 * p * r / (p + r))

        if f1 > best_unconstrained_f1:
            best_unconstrained_f1 = f1
            best_unconstrained_thresh = float(t)
            best_unconstrained_stats = {
                "val_precision": round(float(p), 4),
                "val_recall": round(float(r), 4),
                "val_f1": round(f1, 4),
            }

        if p >= min_precision and f1 > best_constrained_f1:
            best_constrained_f1 = f1
            best_constrained_thresh = float(t)
            best_constrained_stats = {
                "val_precision": round(float(p), 4),
                "val_recall": round(float(r), 4),
                "val_f1": round(f1, 4),
            }

    if best_constrained_thresh is not None:
        calibration_info = {
            "status": "constraint_satisfied",
            "target_min_precision": min_precision,
            "selected_threshold": round(best_constrained_thresh, 6),
            **best_constrained_stats,
            "note": f"Selected threshold maximizing F1 while satisfying Precision >= {min_precision:.2f}.",
        }
        return best_constrained_thresh, calibration_info

    if best_unconstrained_thresh is not None:
        logger.warning(
            "Threshold calibration fallback: no validation threshold satisfied Precision >= %.2f "
            "(max achieved precision=%.4f). Falling back to unconstrained max-F1 threshold=%.6f.",
            min_precision,
            float(precisions[:-1].max()) if len(precisions) > 1 else 0.0,
            best_unconstrained_thresh,
        )
        calibration_info = {
            "status": "fallback_max_f1",
            "target_min_precision": min_precision,
            "selected_threshold": round(best_unconstrained_thresh, 6),
            **best_unconstrained_stats,
            "note": (
                f"No validation threshold satisfied Precision >= {min_precision:.2f}; "
                "explicitly fell back to unconstrained validation F1-maximizing threshold."
            ),
        }
        return best_unconstrained_thresh, calibration_info

    logger.warning(
        "Threshold calibration fallback: all validation thresholds yielded F1=0. "
        "Falling back to default threshold=0.50."
    )
    calibration_info = {
        "status": "fallback_default_0.50",
        "target_min_precision": min_precision,
        "selected_threshold": 0.50,
        "val_precision": 0.0,
        "val_recall": 0.0,
        "val_f1": 0.0,
        "note": "Degenerate validation probability distribution; defaulted threshold to 0.50.",
    }
    return 0.50, calibration_info


def evaluate_predictions(
    y_true,
    y_probs,
    threshold: float = 0.50,
    amounts=None,
    alert_review_cost: float = DEFAULT_ALERT_REVIEW_COST,
    chargeback_fee: float = DEFAULT_CHARGEBACK_FEE,
) -> dict:
    """Compute statistical metrics plus asymmetric business cost-weighted loss and net savings."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_probs, dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    pr_auc = average_precision_score(y_true_arr, y_prob_arr)
    f1_mac = f1_score(y_true_arr, y_pred, average="macro")
    f1_fraud = f1_score(y_true_arr, y_pred, pos_label=1)
    prec = precision_score(y_true_arr, y_pred, zero_division=0)
    rec = recall_score(y_true_arr, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred).ravel()

    if amounts is not None:
        amt_arr = np.asarray(amounts, dtype=float)
    else:
        amt_arr = np.full(len(y_true_arr), DEFAULT_FALLBACK_FRAUD_AMT, dtype=float)

    fn_mask = (y_true_arr == 1) & (y_pred == 0)
    fraud_mask = y_true_arr == 1

    fn_loss = float(np.sum(amt_arr[fn_mask] + chargeback_fee))
    review_loss = float((fp + tp) * alert_review_cost)
    total_cost = fn_loss + review_loss

    unmitigated_cost = float(np.sum(amt_arr[fraud_mask] + chargeback_fee))
    net_savings = unmitigated_cost - total_cost
    savings_pct = (net_savings / unmitigated_cost * 100.0) if unmitigated_cost > 0 else 0.0
    cost_per_1k = (total_cost / max(len(y_true_arr), 1)) * 1000.0

    return {
        "threshold": round(threshold, 6),
        "pr_auc": round(float(pr_auc), 4),
        "f1_macro": round(float(f1_mac), 4),
        "f1_fraud": round(float(f1_fraud), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "cost_metrics": {
            "assumed_review_cost_per_alert": alert_review_cost,
            "assumed_chargeback_fee_per_fn": chargeback_fee,
            "unmitigated_fraud_cost": round(unmitigated_cost, 2),
            "fn_missed_fraud_cost": round(fn_loss, 2),
            "alert_review_cost": round(review_loss, 2),
            "total_expected_cost": round(total_cost, 2),
            "cost_per_1k_transactions": round(cost_per_1k, 2),
            "net_fraud_savings": round(net_savings, 2),
            "net_savings_pct": round(float(savings_pct), 2),
        },
    }