import argparse
import pickle
from pathlib import Path
import sys
from collections import defaultdict
from tqdm.auto import tqdm
import numpy as np
import torch

script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src import config
from src.data_loaders import CelebAAttributeLoader
from src.tasks import AttributePredictionTask
from src.models import GemmaModel


def main(args):
    # Setup
    loader = CelebAAttributeLoader(partition_num=args.partition_num)
    task = AttributePredictionTask()
    model = GemmaModel(model_path=args.model, device=args.device)

    # Data Loading
    samples = loader.load()

    # Run Inference
    all_scores = []
    target_labels = task.get_target_labels()

    for i in tqdm(
        range(0, len(samples), args.batch_size), desc="Processing CelebA tasks"
    ):
        batch_samples = samples[i : min(i + args.batch_size, len(samples))]
        batch_prompts = [task.get_prompt(s) for s in batch_samples]
        batch_scores = model.get_label_scores(batch_prompts, target_labels)
        all_scores.append(batch_scores)

    all_scores = np.concatenate(all_scores, axis=0)

    # Evaluate Metrics
    # Group results by attribute name
    results_by_attribute = defaultdict(lambda: {"labels": [], "scores": []})
    for i, sample in enumerate(samples):
        attr_name = sample["attribute_name"]
        results_by_attribute[attr_name]["labels"].append(sample["label"])
        results_by_attribute[attr_name]["scores"].append(all_scores[i])

    final_metrics = {}
    all_true_labels_flat = []
    all_scores_flat = []

    print("\n===== Calculating Per-Attribute Metrics =====")
    # Calculate per-attribute metrics
    for attr_name, data in sorted(results_by_attribute.items()):
        print(f"\n--- Metrics for Attribute: {attr_name} ---")
        attr_labels = data["labels"]
        attr_scores = np.array(data["scores"])

        # We use the task's evaluate method, which now uses calculate_classification_metrics
        final_metrics[attr_name] = task.evaluate(attr_labels, attr_scores)

        all_true_labels_flat.extend(attr_labels)
        all_scores_flat.extend(attr_scores)

    # Calculate overall metrics
    print("\n===== Calculating Overall Metrics =====")
    final_metrics["overall"] = task.evaluate(
        all_true_labels_flat, np.array(all_scores_flat)
    )

    # Save Results
    model_name_safe = Path(args.model).name
    partition_str = args.partition_num if args.partition_num != -1 else "all"
    output_filename = (
        f"celeba_attributes_{model_name_safe}_partition{partition_str}.pkl"
    )
    output_path = config.OUTPUT_DIR / output_filename

    with open(output_path, "wb") as f:
        pickle.dump(final_metrics, f)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VLM on CelebA Attribute Prediction."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name or path of the multimodal LLM on HuggingFace Hub.",
    )
    parser.add_argument(
        "--partition_num",
        type=int,
        default=2,
        choices=[-1, 0, 1, 2],
        help="CelebA partition to use (-1: all, 0: train, 1: val, 2: test).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda:0", "cuda:1"],  # Add more if needed
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing (image, attribute) tasks.",
    )
    args = parser.parse_args()
    main(args)
