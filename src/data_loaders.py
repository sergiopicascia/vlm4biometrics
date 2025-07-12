"""
Dataset loaders for different biometric datasets.
Each loader should implement the BaseDatasetLoader interface.
"""

from abc import ABC, abstractmethod
import random
import glob
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm.auto import tqdm
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


class FVCLoader(BaseDatasetLoader):
    """
    Loads pairs from the FVC datasets.
    This loader is an iterator that yields data for each sub-database
    (e.g., FVC2000-Db1, FVC2000-Db2, etc.) sequentially.
    """

    def __init__(self, fvc_datasets: List[str] = []):
        self.fvc_datasets_to_run = (
            fvc_datasets if fvc_datasets else config.FVC_DATASET_NAMES
        )
        self.base_fvc_path = config.FVC_BASE_DIR

    def _parse_fvc_index_file(
        self, index_file_path: Path, image_dir: Path, label: int
    ) -> List[Dict[str, Any]]:
        """Helper to parse a single index file and return samples."""
        samples = []
        if not index_file_path.exists():
            return []
        with open(index_file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                img1_name, img2_name = parts
                img1_path = image_dir / img1_name
                img2_path = image_dir / img2_name
                if img1_path.exists() and img2_path.exists():
                    samples.append(
                        {"image_paths": [img1_path, img2_path], "label": label}
                    )
        return samples

    def load(self) -> None:
        """This method is not used for the iterator pattern."""
        raise NotImplementedError(
            "FVCLoader should be used as an iterator. Use load_sub_dbs() instead."
        )

    def load_sub_dbs(self):
        """
        A generator that yields the name and samples for each FVC sub-database.
        Yields:
            tuple: (db_id_str, list_of_samples)
        """
        for fvc_year_name in self.fvc_datasets_to_run:
            fvc_year_root = self.base_fvc_path / fvc_year_name
            fvc_db_index_dir = fvc_year_root / "Dbs"

            for db_name_prefix in config.FVC_DBS:
                db_id = f"{fvc_year_name}-{db_name_prefix.strip('_')}"
                all_samples = []

                # Process Set A
                image_dir_a = fvc_db_index_dir / f"{db_name_prefix}a"
                all_samples.extend(
                    self._parse_fvc_index_file(
                        fvc_db_index_dir / "index_a.MFA", image_dir_a, 1
                    )
                )
                all_samples.extend(
                    self._parse_fvc_index_file(
                        fvc_db_index_dir / "index_a.MFR", image_dir_a, 0
                    )
                )

                # Process Set B
                image_dir_b = fvc_db_index_dir / f"{db_name_prefix}b"
                all_samples.extend(
                    self._parse_fvc_index_file(
                        fvc_db_index_dir / "index_B.MFA", image_dir_b, 1
                    )
                )
                all_samples.extend(
                    self._parse_fvc_index_file(
                        fvc_db_index_dir / "index_B.MFR", image_dir_b, 0
                    )
                )

                if all_samples:
                    print(f"Loaded {len(all_samples)} pairs for {db_id}.")
                    yield db_id, fvc_year_name, all_samples
                else:
                    print(f"Warning: No pairs found for {db_id}. Skipping.")


class AgeDBLoader(BaseDatasetLoader):
    """Loads the AgeDB dataset.
    Filename format: ID_NameSurname_Age_Gender.jpg
    """

    def __init__(self, min_age: int = 1, max_age: int = 101):
        self.dataset_root = config.AGEDB_ROOT
        self.min_age = min_age
        self.max_age = max_age

    def load(self) -> List[Dict[str, Any]]:
        samples = []
        filepaths = glob.glob(str(self.dataset_root / "*.jpg"))

        for filepath_str in filepaths:
            filepath = Path(filepath_str)
            filename = filepath.name
            parts = filename.rsplit(".", 1)[0].split("_")

            if len(parts) < 4:
                continue

            try:
                age = int(parts[2])
                gender = parts[3].lower()
            except (ValueError, IndexError):
                continue

            if not (self.min_age <= age <= self.max_age) or gender not in ["m", "f"]:
                continue

            samples.append({"image_paths": [filepath], "age": age, "gender": gender})

        print(f"Loaded {len(samples)} samples from AgeDB.")
        return samples


class CelebAAttributeLoader(BaseDatasetLoader):
    """
    Loads the CelebA dataset for attribute prediction.
    It "unpivots" the attribute table to create a separate task
    for each (image, attribute) pair.
    """

    def __init__(self, partition_num: int = 2):
        """
        Args:
            partition_num (int): The partition to use (0: train, 1: val, 2: test, -1: all).
        """
        self.attr_filepath = config.CELEBA_ATTR_FILE
        self.partition_filepath = config.CELEBA_PARTITION_FILE
        self.image_dir = config.CELEBA_IMAGE_DIR
        self.partition_num = partition_num

    def load(self) -> List[Dict[str, Any]]:
        attr_df = pd.read_csv(self.attr_filepath)
        partition_df = pd.read_csv(self.partition_filepath)

        df = pd.merge(attr_df, partition_df, on="image_id")

        if self.partition_num != -1:
            df = df[df["partition"] == self.partition_num]

        attribute_names = df.columns[1:41].tolist()

        samples = []
        print("Parsing CelebA attributes into individual tasks...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating CelebA tasks"):
            for attr_name in attribute_names:
                label = 1 if row[attr_name] == 1 else 0

                samples.append(
                    {
                        "image_paths": [self.image_dir / row["image_id"]],
                        "attribute_name": attr_name,
                        "label": label,
                    }
                )

        print(
            f"Loaded {len(samples)} (image, attribute) tasks from CelebA partition {self.partition_num}."
        )
        return samples
