"""
Task definitions for different biometric tasks.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from PIL import Image
import numpy as np
from .metrics import calculate_verification_metrics


class BaseTask(ABC):
    """
    Abstract base class for all tasks. A task defines:
    1. How to create a prompt from a data sample.
    2. What the possible text labels are (e.g., 'yes', 'no').
    3. How to evaluate the model's predictions against ground truth.
    """

    @abstractmethod
    def get_prompt(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Constructs the model-specific prompt from a data sample.
        Opens images and formats them into the chat template structure.
        """
        pass

    @abstractmethod
    def get_target_labels(self) -> List[str]:
        """Returns the list of possible string labels for the task."""
        pass

    @abstractmethod
    def evaluate(self, labels: List[int], scores: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates the predictions and returns a dictionary of metrics.

        Args:
            labels: A list of ground truth labels.
            scores: A numpy array of prediction scores from the model.
        """
        pass


class FaceVerificationTask(BaseTask):
    """Task for face verification: are two faces the same person?"""

    def get_prompt(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Creates the prompt for the face verification task."""
        image1 = Image.open(str(sample["image_paths"][0])).convert("RGB")
        image2 = Image.open(str(sample["image_paths"][1])).convert("RGB")

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image1},
                    {"type": "image", "image": image2},
                    {
                        "type": "text",
                        "text": "Do these two images show the same person? Answer with only 'yes' or 'no'.",
                    },
                ],
            },
        ]
        return messages

    def get_target_labels(self) -> List[str]:
        """The possible answers are 'no' and 'yes'."""
        return ["no", "yes"]

    def evaluate(self, labels: List[int], scores: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates using verification metrics.
        The positive class is 'yes', which has index 1 in get_target_labels.
        """
        # Scores for the positive class 'yes'
        positive_scores = scores[:, 1]
        return calculate_verification_metrics(labels, positive_scores)


class IrisVerificationTask(BaseTask):
    """Task for iris verification: are two iris images from the same person?"""

    def get_prompt(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Creates the prompt for the iris verification task."""
        image1 = Image.open(str(sample["image_paths"][0])).convert("RGB")
        image2 = Image.open(str(sample["image_paths"][1])).convert("RGB")

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image1},
                    {"type": "image", "image": image2},
                    {
                        "type": "text",
                        "text": "Do these two iris images belong to the same person? Answer with only 'yes' or 'no'.",
                    },
                ],
            },
        ]
        return messages

    def get_target_labels(self) -> List[str]:
        """The possible answers are 'no' and 'yes'."""
        return ["no", "yes"]

    def evaluate(self, labels: List[int], scores: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates using verification metrics.
        The positive class is 'yes', which has index 1 in get_target_labels.
        """
        # Scores for the positive class 'yes'
        positive_scores = scores[:, 1]
        return calculate_verification_metrics(labels, positive_scores)
