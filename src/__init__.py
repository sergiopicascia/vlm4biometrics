"""
Biometrics Evaluation Package.
"""

# Expose the configuration object directly
from . import config

# Expose key classes and functions from each module
from .data_loaders import LFWLoader
from .metrics import calculate_verification_metrics
from .models import GemmaModel
from .tasks import FaceVerificationTask

# You can also define what `from src import *` would import
__all__ = [
    "config",
    "LFWLoader",
    "calculate_verification_metrics",
    "GemmaModel",
    "FaceVerificationTask",
]
