"""
Dataset loaders for different biometric datasets.
Each loader should implement the BaseDatasetLoader interface.
"""

from abc import ABC, abstractmethod
import random
from pathlib import Path
from typing import List, Dict, Any
from . import config


class BaseDatasetLoader(ABC):
    """
    Abstract base class for all dataset loaders.
    Each loader is responsible for parsing its specific dataset format
    and returning a standardized list of samples.
    """

    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """
        Loads the dataset and returns a list of samples.
        Each sample should be a dictionary containing at least:
        - 'image_paths': A list of paths to the input images.
        - 'label': The ground truth label for the sample.

        Example for verification: [{'image_paths': [path1, path2], 'label': 1}, ...]
        Example for age estimation: [{'image_paths': [path1], 'label': 42}, ...]
        """
        pass


class LFWLoader(BaseDatasetLoader):
    """Loads the Labeled Faces in the Wild (LFW) dataset for the verification task."""

    def load(self) -> List[Dict[str, Any]]:
        samples = []
        with open(config.LFW_PAIRS_FILE, "r") as f:
            lines = f.readlines()[1:]  # Skip header

        for i, line in enumerate(lines):
            parts = line.strip().split(",")
            if len(parts) == 4 and not parts[-1]:
                # Matched pair
                name, n1, n2, _ = parts
                img1_path = config.LFW_IMAGE_DIR / name / f"{name}_{int(n1):04d}.jpg"
                img2_path = config.LFW_IMAGE_DIR / name / f"{name}_{int(n2):04d}.jpg"
                samples.append({"image_paths": [img1_path, img2_path], "label": 1})
            elif len(parts) == 4:
                # Mismatched pair
                name1, n1, name2, n2 = parts
                img1_path = config.LFW_IMAGE_DIR / name1 / f"{name1}_{int(n1):04d}.jpg"
                img2_path = config.LFW_IMAGE_DIR / name2 / f"{name2}_{int(n2):04d}.jpg"
                samples.append({"image_paths": [img1_path, img2_path], "label": 0})
            else:
                print(f"Warning: Skipping malformed line {i+1}: {line.strip()}")

        print(f"Loaded {len(samples)} pairs from LFW.")
        return samples


class CASIAIrisLoader(BaseDatasetLoader):
    """
    Loads and generates pairs for the CASIA-Iris-Thousand dataset.
    This loader dynamically creates genuine and impostor pairs.
    """

    def __init__(self, num_total_pairs: int = 20000, random_seed: int = 42):
        self.num_total_pairs = num_total_pairs
        self.random_seed = random_seed
        self.dataset_root = config.CASIA_ROOT
        print(
            f"Initialized CASIAIrisLoader to generate {num_total_pairs} pairs with seed {random_seed}."
        )

    def _get_images_by_subject_eye(self) -> Dict[tuple, List[Path]]:
        """Scans the dataset and organizes images by (subject_id, eye_type)."""
        images_by_subject_eye = {}
        subject_ids = sorted(
            [
                d.name
                for d in self.dataset_root.iterdir()
                if d.is_dir() and d.name.isdigit()
            ]
        )

        for subject_id in subject_ids:
            subject_dir = self.dataset_root / subject_id
            for eye_type in ["L", "R"]:
                eye_dir = subject_dir / eye_type
                if eye_dir.exists() and eye_dir.is_dir():
                    image_paths = sorted(list(eye_dir.glob("*.jpg")))
                    if image_paths:
                        images_by_subject_eye[(subject_id, eye_type)] = image_paths
        return images_by_subject_eye

    def load(self) -> List[Dict[str, Any]]:
        """Generates genuine and impostor pairs and returns them as samples."""
        random.seed(self.random_seed)
        images_by_subject_eye = self._get_images_by_subject_eye()
        subject_eye_keys = list(images_by_subject_eye.keys())
        all_subject_ids = sorted(list(set(key[0] for key in subject_eye_keys)))

        samples = []
        num_genuine_pairs = self.num_total_pairs // 2
        num_impostor_pairs = self.num_total_pairs - num_genuine_pairs

        # --- Generate Genuine Pairs ---
        while len(samples) < num_genuine_pairs:
            subject_id, eye_type = random.choice(subject_eye_keys)
            if len(images_by_subject_eye[(subject_id, eye_type)]) < 2:
                continue
            img1_path, img2_path = random.sample(
                images_by_subject_eye[(subject_id, eye_type)], 2
            )
            samples.append({"image_paths": [img1_path, img2_path], "label": 1})

        # --- Generate Impostor Pairs ---
        while len(samples) - num_genuine_pairs < num_impostor_pairs:
            s_id1, s_id2 = random.sample(all_subject_ids, 2)
            eye_type = random.choice(["L", "R"])

            keys1 = [
                k
                for k in subject_eye_keys
                if k[0] == s_id1 and k[1] == eye_type and images_by_subject_eye[k]
            ]
            keys2 = [
                k
                for k in subject_eye_keys
                if k[0] == s_id2 and k[1] == eye_type and images_by_subject_eye[k]
            ]

            if not keys1 or not keys2:
                continue

            key1 = random.choice(keys1)
            img1_path = random.choice(images_by_subject_eye[key1])
            key2 = random.choice(keys2)
            img2_path = random.choice(images_by_subject_eye[key2])

            samples.append({"image_paths": [img1_path, img2_path], "label": 0})

        # Shuffle the final list of samples
        random.shuffle(samples)
        print(f"Loaded {len(samples)} pairs from CASIA-Iris-Thousand.")
        return samples
