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
from src.data_loaders import FVCLoader
from src.tasks import FingerprintVerificationTask
from src.models import GemmaModel
from src.metrics import average_metrics


def run_inference_on_samples(samples, task, model, batch_size):
    """Helper function to run inference on a list of samples."""
    all_scores = []
    target_labels = task.get_target_labels()

    for i in tqdm(
        range(0, len(samples), batch_size), desc="Processing Batches", leave=False
    ):
        batch_samples = samples[i : min(i + batch_size, len(samples))]
        batch_prompts = [task.get_prompt(sample) for sample in batch_samples]
        batch_scores = model.get_label_scores(batch_prompts, target_labels)
        all_scores.append(batch_scores)

    return np.concatenate(all_scores, axis=0) if all_scores else np.array([])


def main(args):
    # --- Setup components ---
    loader = FVCLoader(fvc_datasets=args.fvc_datasets)
    task = FingerprintVerificationTask()
    model = GemmaModel(model_path=args.model, device=args.device)

    overall_results = {}

    # --- Main Loop over FVC Years and Sub-Databases ---
    for db_id, fvc_year, samples in loader.load_sub_dbs():
        print(f"\n===== Processing FVC Sub-Database: {db_id} =====")

        # Initialize result structure for the year if not present
        if fvc_year not in overall_results:
            overall_results[fvc_year] = {"Dbs": {}, "db_metrics_list": []}

        # Run inference
        scores = run_inference_on_samples(samples, task, model, args.batch_size)
        if scores.size == 0:
            print(f"No scores generated for {db_id}, skipping.")
            continue

        true_labels = [s["label"] for s in samples]

        # Evaluate and store metrics
        db_metrics = task.evaluate(true_labels, scores)

        print(f"\n--- Metrics for {db_id} ---")
        for key, val in db_metrics.items():
            if isinstance(val, (int, float)):
                print(f"  {key.replace('_', ' ').title()}: {val:.4f}")
        print("--------------------------\n")

        overall_results[fvc_year]["Dbs"][db_id] = db_metrics
        overall_results[fvc_year]["db_metrics_list"].append(db_metrics)

    # --- Calculate and Print Averages ---
    for fvc_year, year_results in overall_results.items():
        avg_metrics = average_metrics(year_results["db_metrics_list"])
        year_results["Average"] = avg_metrics
        del year_results["db_metrics_list"]  # Clean up before saving

        print(f"\n--- Average Metrics for {fvc_year} ---")
        for key, val in avg_metrics.items():
            print(f"  Avg {key.replace('_', ' ').title()}: {val:.4f}")
        print("-----------------------------------\n")

    # --- Save Final Results ---
    model_name = Path(args.model).name
    output_filename = f"fvc_{model_name}.pkl"
    output_path = config.OUTPUT_DIR / output_filename

    with open(output_path, "wb") as f:
        pickle.dump(overall_results, f)
    print(f"Overall results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VLM on FVC Fingerprint Verification."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name or path of the multimodal LLM on HuggingFace Hub.",
    )
    parser.add_argument(
        "--fvc_datasets",
        nargs="+",
        default=None,
        help=f"List of FVC datasets. Default: all {config.FVC_DATASET_NAMES}.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (e.g., cuda:0, cpu).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for processing pairs."
    )
    args = parser.parse_args()
    main(args)
