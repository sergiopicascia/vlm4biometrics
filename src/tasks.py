"""
Task definitions for different biometric tasks.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import random
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
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": "How old is the person in this image? Answer with only a number representing their age.",
                    },
                ],
            },
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
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": "What is the gender of the person in this image? Answer with only 'male' or 'female'.",
                    },
                ],
            },
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
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            },
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


class BaseMCQTask(ABC):
    """
    Abstract base class for all Multiple-Choice Question (MCQ) tasks.
    """

    def __init__(self, num_options: int):
        if num_options < 2:
            raise ValueError("MCQ tasks must have at least 2 options.")
        self.num_options = num_options

    def get_option_labels(self) -> List[str]:
        """Returns a list of option characters, e.g., ['A', 'B', 'C']."""
        return [chr(65 + i) for i in range(self.num_options)]

    @abstractmethod
    def _get_question(self, sample: Dict[str, Any]) -> str:
        """Returns the question text for a given sample."""
        pass

    @abstractmethod
    def _get_correct_option_text(self, sample: Dict[str, Any]) -> str:
        """Returns the text of the correct answer for a given sample."""
        pass

    @abstractmethod
    def _get_distractor_pool(self, sample: Dict[str, Any]) -> List[str]:
        """Returns a list of all possible distractor texts for a given sample."""
        pass

    def generate_prompt_and_options(
        self, sample: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        """
        Generates a formatted MCQ prompt and returns the ordered list of option texts.

        Returns:
            A tuple containing:
            - The full, formatted prompt string.
            - The list of option texts in the order they appeared (e.g., ['Yes', 'No']).
        """
        correct_option = self._get_correct_option_text(sample)
        distractor_pool = self._get_distractor_pool(sample)

        # Select distractors randomly from the pool
        num_distractors = self.num_options - 1
        distractors = random.sample(distractor_pool, num_distractors)

        # Create final options list and shuffle it
        final_options = [correct_option] + distractors
        random.shuffle(final_options)

        # Format the prompt
        option_labels = self.get_option_labels()
        options_str = "\n".join(
            [f"{label}. {text}" for label, text in zip(option_labels, final_options)]
        )
        question = self._get_question(sample)
        prompt_text = f"{question}\n{options_str}"

        return prompt_text, final_options

    def evaluate(
        self, correct_option_texts: List[str], predicted_option_texts: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluates MCQ predictions using classification metrics.
        This method is shared by all MCQ tasks.
        """
        # Filter out samples where the model's output could not be parsed
        valid_true = [
            true
            for true, pred in zip(correct_option_texts, predicted_option_texts)
            if pred is not None
        ]
        valid_pred = [pred for pred in predicted_option_texts if pred is not None]

        # Get all unique labels for the classification report
        all_labels = sorted(list(set(valid_true) | set(valid_pred)))

        print(
            f"Total Parsable Predictions: {len(valid_pred)}/{len(predicted_option_texts)}"
        )

        return calculate_classification_metrics(
            valid_true, valid_pred, labels=all_labels
        )


class VerificationMCQTask(BaseMCQTask):
    """MCQ Task for any binary verification (Face, Iris, Fingerprint)."""

    def __init__(self, domain: str = "person"):
        super().__init__(num_options=2)
        self.domain = domain  # e.g., 'person', 'iris', 'finger'
        self.options = ["Yes", "No"]

    def _get_question(self, sample: Dict[str, Any]) -> str:
        return f"Do these two images show the same {self.domain}?"

    def _get_correct_option_text(self, sample: Dict[str, Any]) -> str:
        return "Yes" if sample["label"] == 1 else "No"

    def _get_distractor_pool(self, sample: Dict[str, Any]) -> List[str]:
        correct = self._get_correct_option_text(sample)
        return [opt for opt in self.options if opt != correct]


class AgeEstimationMCQTask(BaseMCQTask):
    """MCQ Task for binned age estimation."""

    def __init__(self):
        super().__init__(num_options=4)
        self.age_bins = [
            "0-9",
            "10-19",
            "20-29",
            "30-39",
            "40-49",
            "50-59",
            "60-69",
            "70-79",
            "80-90",
            "91-101",
        ]

    def _get_question(self, sample: Dict[str, Any]) -> str:
        return "What is the age range of the person shown in the image?"

    def _get_age_bin(self, age: int) -> str:
        """Helper to find the correct bin for a given age."""
        for bin_str in self.age_bins:
            low, high = map(int, bin_str.split("-"))
            if low <= age <= high:
                return bin_str
        return None

    def _get_correct_option_text(self, sample: Dict[str, Any]) -> str:
        return self._get_age_bin(sample["age"])

    def _get_distractor_pool(self, sample: Dict[str, Any]) -> List[str]:
        correct_bin = self._get_correct_option_text(sample)
        return [b for b in self.age_bins if b != correct_bin]


class GenderPredictionMCQTask(BaseMCQTask):
    """MCQ Task for gender prediction."""

    def __init__(self):
        super().__init__(num_options=2)
        self.options = ["male", "female"]

    def _get_question(self, sample: Dict[str, Any]) -> str:
        return "What is the gender of the person in this image?"

    def _get_correct_option_text(self, sample: Dict[str, Any]) -> str:
        return "male" if sample["gender"] == "m" else "female"

    def _get_distractor_pool(self, sample: Dict[str, Any]) -> List[str]:
        correct = self._get_correct_option_text(sample)
        return [opt for opt in self.options if opt != correct]


class AttributePredictionMCQTask(BaseMCQTask):
    """MCQ Task for CelebA attribute prediction."""

    def __init__(self):
        super().__init__(num_options=2)
        self.options = ["Yes", "No"]

    def _get_question(self, sample: Dict[str, Any]) -> str:
        prompt_attr_name = sample["attribute_name"].replace("_", " ").lower()
        return f"Does the person in the image have the attribute '{prompt_attr_name}'?"

    def _get_correct_option_text(self, sample: Dict[str, Any]) -> str:
        return "Yes" if sample["label"] == 1 else "No"

    def _get_distractor_pool(self, sample: Dict[str, Any]) -> List[str]:
        correct = self._get_correct_option_text(sample)
        return [opt for opt in self.options if opt != correct]
