import argparse
import pickle
from pathlib import Path
import sys
from collections import defaultdict
from tqdm.auto import tqdm
from PIL import Image

script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src import config
from src.data_loaders import CelebAAttributeLoader
from src.tasks import AttributePredictionMCQTask
from src.utils import get_model


def main(args):
    # --- 1. Setup ---
    loader = CelebAAttributeLoader(partition_num=args.partition_num)
    task = AttributePredictionMCQTask()
    model = get_model(model_path=args.model, device=args.device)

    # --- 2. Data Loading ---
    # `samples` is a flat list of all (image, attribute) tasks
    samples = loader.load()

    # --- 3. Run MCQ Inference and Parsing ---
    all_predictions = (
        []
    )  # This will store tuples of (attr_name, correct_text, predicted_text)

    for i in tqdm(
        range(0, len(samples), args.batch_size), desc="Processing CelebA Tasks (MCQ)"
    ):
        batch_samples = samples[i : min(i + args.batch_size, len(samples))]

        batch_prompts_for_model = []
        batch_options_context = []

        for sample in batch_samples:
            prompt_text, final_options = task.generate_prompt_and_options(sample)
            batch_options_context.append(final_options)

            # Note: For CelebA, we only have one image per sample
            image = Image.open(str(sample["image_paths"][0])).convert("RGB")
            model_prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            batch_prompts_for_model.append(model_prompt)

        # Get parsed labels ('A', 'B', etc.) from the model
        parsed_labels = model.get_mcq_predictions(
            batch_prompts=batch_prompts_for_model,
            option_labels=task.get_option_labels(),
            options_text=batch_options_context,
        )

        # Process the batch results
        for idx, parsed_label in enumerate(parsed_labels):
            sample = batch_samples[idx]
            final_options = batch_options_context[idx]
            correct_text = task._get_correct_option_text(sample)

            predicted_text = None
            if parsed_label:
                option_index = ord(parsed_label) - ord("A")
                if 0 <= option_index < len(final_options):
                    predicted_text = final_options[option_index]

            all_predictions.append(
                (sample["attribute_name"], correct_text, predicted_text)
            )

    # --- 4. Evaluate Metrics ---
    # Group results by attribute name
    results_by_attribute = defaultdict(lambda: {"correct": [], "predicted": []})
    for attr_name, correct_text, predicted_text in all_predictions:
        results_by_attribute[attr_name]["correct"].append(correct_text)
        results_by_attribute[attr_name]["predicted"].append(predicted_text)

    final_metrics = {}
    all_correct_flat = []
    all_predicted_flat = []

    print("\n===== Calculating Per-Attribute Metrics =====")
    for attr_name, data in sorted(results_by_attribute.items()):
        print(f"\n--- Metrics for Attribute: {attr_name} ---")
        correct_labels = data["correct"]
        predicted_labels = data["predicted"]

        # The task's evaluate method handles the metrics calculation
        final_metrics[attr_name] = task.evaluate(correct_labels, predicted_labels)

        all_correct_flat.extend(correct_labels)
        all_predicted_flat.extend(predicted_labels)

    # Calculate overall metrics
    print("\n===== Calculating Overall Metrics =====")
    final_metrics["overall"] = task.evaluate(all_correct_flat, all_predicted_flat)

    # --- 5. Save Results ---
    model_name_safe = Path(args.model).name
    partition_str = args.partition_num if args.partition_num != -1 else "all"
    output_filename = (
        f"celeba_mcq_attributes_{model_name_safe}_partition{partition_str}.pkl"
    )
    output_path = config.OUTPUT_DIR / output_filename

    with open(output_path, "wb") as f:
        pickle.dump(final_metrics, f)
    print(f"\nMCQ-based results for CelebA saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VLM on CelebA Attribute Prediction using MCQ method."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name or path of the VLM on HuggingFace Hub.",
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
        default="cuda:0",
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for processing (image, attribute) tasks.",
    )
    args = parser.parse_args()
    main(args)
