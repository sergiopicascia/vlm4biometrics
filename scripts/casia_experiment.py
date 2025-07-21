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
from src.data_loaders import CASIAIrisLoader
from src.tasks import IrisVerificationTask
from src.models import GemmaModel


def main(args):
    # Instantiate the components for this specific experiment
    loader = CASIAIrisLoader(
        num_total_pairs=args.num_pairs, random_seed=args.random_seed
    )
    task = IrisVerificationTask()
    model = GemmaModel(model_path=args.model, device=args.device)

    # Data Loading
    samples = loader.load()
    true_labels = [sample["label"] for sample in samples]

    # Run Inference
    all_scores = []
    target_labels = task.get_target_labels()

    for i in tqdm(range(0, len(samples), args.batch_size), desc="Processing Batches"):
        batch_samples = samples[i : min(i + args.batch_size, len(samples))]
        batch_prompts = [task.get_prompt(sample) for sample in batch_samples]
        batch_scores = model.get_log_probs(batch_prompts, target_labels)
        all_scores.append(batch_scores)

    all_scores = np.concatenate(all_scores, axis=0)

    # Compute metrics
    metrics = task.evaluate(true_labels, all_scores)

    # Save Results
    model_name = Path(args.model).name
    output_filename = (
        f"casia-iris_{model_name}_s{args.random_seed}_n{args.num_pairs}.pkl"
    )
    output_path = config.OUTPUT_DIR / output_filename

    with open(output_path, "wb") as f:
        pickle.dump(metrics, f)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Multimodal LLM on Iris Verification (CASIA)."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name or path of the multimodal LLM on HuggingFace Hub.",
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=20000,
        help="Total image pairs to generate.",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Seed for pair generation."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda:0", "cuda:1"],
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Processing batch size."
    )
    args = parser.parse_args()
    main(args)
