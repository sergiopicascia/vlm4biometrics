"""
Metrics calculation for different biometric tasks.
"""

import numpy as np
from sklearn.metrics import roc_curve, auc


def calculate_verification_metrics(labels, scores):
    """Calculates FAR, FRR, Accuracy, EER, and AUC."""
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    if not isinstance(scores, np.ndarray):
        scores = np.array(scores)

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # Calculate EER
    t1_cand = np.where(fnr <= fpr)[0]
    t2_cand = np.where(fnr >= fpr)[0]

    t1 = t1_cand.min()
    t2 = t2_cand.max()

    if (fnr[t1] + fpr[t1]) <= (fnr[t2] + fpr[t2]):
        eer_index = t1
        eer_threshold = thresholds[eer_index]
        eer_low = fnr[t1]
        eer_high = fpr[t1]
    else:
        eer_index = t2
        eer_threshold = thresholds[eer_index]
        eer_low = fpr[t2]
        eer_high = fnr[t2]

    eer_high, eer_low
    eer = np.mean([eer_low, eer_high])

    # Calculate Accuracy at EER threshold
    predictions_at_eer = (scores >= eer_threshold).astype(int)
    accuracy_at_eer = np.mean(predictions_at_eer == labels)

    auc_score = auc(fpr, tpr)

    print("\n--- Face Verification Metrics ---")
    print(f"Equal Error Rate (EER): {eer:.4%}")
    print(f"Threshold at EER: {eer_threshold:.4f}")
    print(f"Accuracy at EER Threshold: {accuracy_at_eer:.2%}")
    print(f"Area Under ROC Curve (AUC): {auc_score:.4f}")
    print(f"FAR @ EER Threshold: {fpr[eer_index]:.4%}")
    print(f"FRR @ EER Threshold: {fnr[eer_index]:.4%}")
    print("--------------------------\n")

    return {
        "scores": scores.tolist(),
        "labels": labels.tolist(),
        "eer": eer,
        "eer_low": eer_low,
        "eer_high": eer_high,
        "threshold_at_eer": eer_threshold,
        "accuracy_at_eer": accuracy_at_eer,
        "auc": auc_score,
        "far_at_eer": fpr[eer_index],
        "frr_at_eer": fnr[eer_index],
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }
