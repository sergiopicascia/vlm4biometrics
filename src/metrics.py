"""
Metrics calculation for different biometric tasks.
"""

from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
    mean_absolute_error,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
import properscoring as ps


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

    eer = np.mean([eer_low, eer_high])

    # Calculate Accuracy at EER threshold
    predictions_at_eer = (scores >= eer_threshold).astype(int)
    accuracy_at_eer = np.mean(predictions_at_eer == labels)

    auc_score = auc(fpr, tpr)

    print("\n--- Verification Metrics ---")
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


def average_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Averages a list of metric dictionaries (produced by calculate_verification_metrics).
    """
    if not metrics_list:
        return {}

    keys_to_average = [
        "eer",
        "accuracy_at_eer",
        "auc",
        "far_at_eer",
        "frr_at_eer",
        "threshold_at_eer",
    ]

    valid_keys = [key for key in keys_to_average if key in metrics_list[0]]

    avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in valid_keys}
    return avg_metrics


def calculate_age_estimation_metrics(
    true_ages: np.ndarray,
    predicted_ages: np.ndarray,
    age_probabilities: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Calculates comprehensive metrics for the age estimation task.

    Args:
        true_ages: Ground truth ages.
        predicted_ages: The most likely predicted age for each sample.
        age_probabilities: A (num_samples, num_age_classes) array of predicted probabilities.
                        Required for weighted-MAE and CRPS.
    """
    if not isinstance(true_ages, np.ndarray):
        true_ages = np.array(true_ages)
    if not isinstance(predicted_ages, np.ndarray):
        predicted_ages = np.array(predicted_ages)

    metrics = {}
    errors = predicted_ages - true_ages

    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(true_ages, predicted_ages)
    metrics["mae"] = mae

    # Cumulative Score (CS)
    cs_thresholds = [1, 3, 5, 7, 10]
    cs_scores = {f"cs@{t}": np.mean(np.abs(errors) <= t) for t in cs_thresholds}
    metrics["cs_scores"] = cs_scores

    if age_probabilities is not None:
        age_range = np.arange(1, age_probabilities.shape[1] + 1)

        # MAE of the Expected Value
        expected_age = np.sum(age_probabilities * age_range, axis=1)
        metrics["expected_value_mae"] = mean_absolute_error(true_ages, expected_age)

        # Standard Deviation of Expected Value Errors
        expected_value_errors = expected_age - true_ages
        metrics["expected_value_error_std"] = np.std(expected_value_errors)

        # Mean Predicted Uncertainty
        expected_sq_age = np.sum(age_probabilities * (age_range**2), axis=1)
        predicted_variances = expected_sq_age - (expected_age**2)
        metrics["mean_predicted_uncertainty"] = np.mean(np.sqrt(predicted_variances))

        # Weighted MAE
        absolute_errors_matrix = np.abs(
            true_ages[:, np.newaxis] - age_range[np.newaxis, :]
        )
        weighted_mae_per_sample = np.sum(
            absolute_errors_matrix * age_probabilities, axis=1
        )
        metrics["weighted_mae"] = np.mean(weighted_mae_per_sample)

        # Continuous Ranked Probability Score (CRPS)
        crps_scores = [
            ps.crps_ensemble(obs, age_range, weights=fcast)
            for obs, fcast in zip(true_ages, age_probabilities)
        ]
        metrics["crps"] = np.mean(crps_scores)

    metrics["error_std"] = np.std(errors)
    metrics["predicted_ages"] = predicted_ages.tolist()

    print(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}")
    if "expected_value_mae" in metrics:
        print(f"Expected Value MAE: {metrics['expected_value_mae']:.4f}")
    if "weighted_mae" in metrics:
        print(f"Weighted MAE (Expected Error): {metrics['weighted_mae']:.4f}")
    if "crps" in metrics:
        print(f"CRPS: {metrics['crps']:.4f}")
    print("Cumulative Scores (CS):")
    for t, score in metrics["cs_scores"].items():
        print(f"  {t.upper()}: {score:.2%}")
    print(f"Standard Deviation of Errors: {metrics['error_std']:.4f}")
    print("------------------------------\n")

    return metrics


def calculate_classification_metrics(
    true_labels: List[str], predicted_labels: List[str], labels: List[str] = None
) -> Dict[str, Any]:
    """Calculates standard classification metrics (accuracy, F1, etc.)."""
    if labels is None:
        labels = sorted(list(set(true_labels) | set(predicted_labels)))

    accuracy = accuracy_score(true_labels, predicted_labels)
    report_dict = classification_report(
        true_labels, predicted_labels, labels=labels, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)

    print("\n--- Classification Metrics ---")
    print(f"Accuracy: {accuracy:.2%}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(
        classification_report(
            true_labels, predicted_labels, labels=labels, zero_division=0
        )
    )
    print("------------------------------\n")

    return {
        "accuracy": accuracy,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }
