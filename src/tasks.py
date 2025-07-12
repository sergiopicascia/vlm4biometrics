"""
Task definitions for different biometric tasks.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from PIL import Image
import numpy as np
from .metrics import (
    calculate_verification_metrics,
    calculate_age_estimation_metrics,
    calculate_classification_metrics,
)


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


class FingerprintVerificationTask(BaseTask):
    """Task for fingerprint verification: are two fingerprints from the same finger?"""

    def get_prompt(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Creates the prompt for the fingerprint verification task."""
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
                        "text": "Do these two fingerprint images belong to the same finger? Answer with only 'yes' or 'no'.",
                    },
                ],
            },
        ]
        return messages

    def get_target_labels(self) -> List[str]:
        """The possible answers are 'no' and 'yes'."""
        return ["no", "yes"]

    def evaluate(self, labels: List[int], scores: np.ndarray) -> Dict[str, Any]:
        """Evaluates using verification metrics."""
        positive_scores = scores[:, 1]
        return calculate_verification_metrics(labels, positive_scores)


class AgeEstimationTask(BaseTask):
    """Task for age estimation from a single face image."""

    def __init__(self, min_age: int = 1, max_age: int = 101):
        self.min_age = min_age
        self.max_age = max_age
        # The possible labels are the string representation of all ages
        self._target_labels = [str(age) for age in range(min_age, max_age + 1)]

    def get_prompt(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        image = Image.open(str(sample["image_paths"][0])).convert("RGB")
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": "How old is the person in this image? Answer with only a number representing their age.",
                    },
                ],
            }
        ]

    def get_target_labels(self) -> List[str]:
        return self._target_labels

    def evaluate(self, labels: List[int], scores: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates age estimation predictions.

        Args:
            labels: A list of ground truth ages.
            scores: A numpy array of shape (num_samples, num_age_classes) containing probabilities.
        """
        # Predicted age is the index of the max probability, offset by min_age
        predicted_age_indices = np.argmax(scores, axis=1)
        predicted_ages = predicted_age_indices + self.min_age

        return calculate_age_estimation_metrics(
            np.array(labels), predicted_ages, age_probabilities=scores
        )


class GenderPredictionTask(BaseTask):
    """Task for gender classification from a single face image."""

    def get_prompt(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        image = Image.open(str(sample["image_paths"][0])).convert("RGB")
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": "What is the gender of the person in this image? Answer with only 'male' or 'female'.",
                    },
                ],
            }
        ]

    def get_target_labels(self) -> List[str]:
        return ["female", "male"]

    def evaluate(self, labels: List[str], scores: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates gender classification predictions.

        Args:
            labels: A list of ground truth genders ('f', 'm').
            scores: A numpy array of shape (num_samples, 2) with probabilities for [female, male].
        """
        # Map ground truth 'f'/'m' to 'female'/'male' to match predictions
        true_labels_mapped = ["female" if g == "f" else "male" for g in labels]

        # Predicted gender is the one with the higher probability
        predicted_indices = np.argmax(scores, axis=1)
        predicted_labels = [self.get_target_labels()[i] for i in predicted_indices]

        return calculate_classification_metrics(
            true_labels_mapped, predicted_labels, self.get_target_labels()
        )


class AttributePredictionTask(BaseTask):
    """Task for predicting a binary attribute from a single image."""

    def get_prompt(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Creates a dynamic prompt based on the attribute in the sample."""
        image = Image.open(str(sample["image_paths"][0])).convert("RGB")

        # Convert attribute name to a more natural language query
        # e.g., "5_o_Clock_Shadow" -> "5 o clock shadow"
        prompt_attr_name = sample["attribute_name"].replace("_", " ").lower()

        prompt_text = (
            f"Does the person in the image have the attribute '{prompt_attr_name}'? "
            f"Answer with only 'yes' or 'no'."
        )

        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

    def get_target_labels(self) -> List[str]:
        """The possible answers are 'no' and 'yes'."""
        return ["no", "yes"]

    def evaluate(self, labels: List[int], scores: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates binary classification predictions using standard metrics.

        Args:
            labels: A list of ground truth labels (0 or 1).
            scores: A numpy array of shape (num_samples, 2) with probabilities for [no, yes].
        """
        # Convert ground truth 0/1 to 'no'/'yes' to match predictions
        true_labels_mapped = ["yes" if l == 1 else "no" for l in labels]

        # Predicted label is the one with the higher probability
        predicted_indices = np.argmax(scores, axis=1)
        predicted_labels = [self.get_target_labels()[i] for i in predicted_indices]

        # The positive class is 'yes'
        return calculate_classification_metrics(
            true_labels_mapped, predicted_labels, labels=["no", "yes"]
        )
