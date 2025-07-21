"""
Biometrics Evaluation Package.
"""

from . import config

from .data_loaders import (
    LFWLoader,
    CASIAIrisLoader,
    FVCLoader,
    AgeDBLoader,
    CelebAAttributeLoader,
)
from .metrics import calculate_verification_metrics
from .models import GemmaModel
from .tasks import (
    FaceVerificationTask,
    IrisVerificationTask,
    FingerprintVerificationTask,
    AgeEstimationTask,
    GenderPredictionTask,
    AttributePredictionTask,
    VerificationMCQTask,
    AgeEstimationMCQTask,
    GenderPredictionMCQTask,
    AttributePredictionMCQTask,
)
from .utils import extract_option_label

__all__ = [
    "config",
    "LFWLoader",
    "CASIAIrisLoader",
    "FVCLoader",
    "AgeDBLoader",
    "CelebAAttributeLoader",
    "calculate_verification_metrics",
    "GemmaModel",
    "FaceVerificationTask",
    "IrisVerificationTask",
    "FingerprintVerificationTask",
    "AgeEstimationTask",
    "GenderPredictionTask",
    "AttributePredictionTask",
    "VerificationMCQTask",
    "AgeEstimationMCQTask",
    "GenderPredictionMCQTask",
    "AttributePredictionMCQTask",
    "extract_option_label",
]
