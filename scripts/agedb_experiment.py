import argparse
import pickle
from pathlib import Path
import sys
from tqdm.auto import tqdm
import numpy as np
import torch

script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src import config
from src.data_loaders import AgeDBLoader
from src.tasks import AgeEstimationTask, GenderPredictionTask
from src.utils import get_model


def main(args):
    # Setup
    loader = AgeDBLoader()
    model = get_model(model_path=args.model, device=args.device)

    # Data Loading
    samples = loader.load()

    # Run Tasks Sequentially
    tasks_to_run = {
        "age": AgeEstimationTask(),
        "gender": GenderPredictionTask(),
    }

    for task_name, task_instance in tasks_to_run.items():
        print(f"\n===== Running Task: {task_name.title()} Prediction =====")

        all_scores = []
        target_labels = task_instance.get_target_labels()

        for i in tqdm(
            range(0, len(samples), args.batch_size), desc=f"Processing for {task_name}"
        ):
            batch_samples = samples[i : min(i + args.batch_size, len(samples))]
            batch_prompts = [
                task_instance.get_prompt(sample) for sample in batch_samples
            ]

            # Use the generic model interface to get scores for this task's labels
            batch_scores = model.get_log_probs(batch_prompts, target_labels)
            all_scores.append(batch_scores)

        all_scores = np.concatenate(all_scores, axis=0)

        # Get the appropriate ground truth labels for the current task
        if task_name == "age":
            true_labels = [s["age"] for s in samples]
        elif task_name == "gender":
            true_labels = [s["gender"] for s in samples]
        else:
            continue  # Should not happen

        # Evaluate and save results
        metrics = task_instance.evaluate(true_labels, all_scores)

        model_name_safe = Path(args.model).name
        output_filename = f"agedb_{task_name}_{model_name_safe}.pkl"
        output_path = config.OUTPUT_DIR / output_filename

        with open(output_path, "wb") as f:
            pickle.dump(metrics, f)
        print(f"Results for {task_name} saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLM on AgeDB dataset.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name or path of the multimodal LLM on HuggingFace Hub.",
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
        default=16,  # Adjusted default, as we do 2 passes (age+gender) per image effectively
        help="Batch size for processing images.",
    )
    args = parser.parse_args()
    main(args)
