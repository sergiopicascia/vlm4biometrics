import argparse
import pickle
from pathlib import Path
import sys
from tqdm.auto import tqdm
from PIL import Image

script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src import config
from src.data_loaders import FVCLoader
from src.tasks import VerificationMCQTask
from src.utils import get_model
from src.metrics import average_metrics


def main(args):
    # --- 1. Setup components ---
    loader = FVCLoader(fvc_datasets=args.fvc_datasets)
    # Use the generic verification task, specifying the 'fingerprint' domain
    task = VerificationMCQTask(domain="fingerprint")
    model = get_model(model_path=args.model, device=args.device)

    overall_results = {}

    # --- 2. Main Loop over FVC Years and Sub-Databases ---
    for db_id, fvc_year, samples in loader.load_sub_dbs():
        print(f"\n===== Processing FVC Sub-Database: {db_id} (MCQ) =====")

        # Initialize result structure for the year if not present
        if fvc_year not in overall_results:
            overall_results[fvc_year] = {"Dbs": {}, "db_metrics_list": []}

        # --- 3. Run MCQ Inference for the current sub-database ---
        all_correct_option_texts = []
        all_predicted_option_texts = []

        for i in tqdm(
            range(0, len(samples), args.batch_size),
            desc=f"Processing {db_id}",
            leave=False,
        ):
            batch_samples = samples[i : min(i + args.batch_size, len(samples))]

            batch_prompts_for_model = []
            batch_options_context = []

            for sample in batch_samples:
                prompt_text, final_options = task.generate_prompt_and_options(sample)
                all_correct_option_texts.append(task._get_correct_option_text(sample))
                batch_options_context.append(final_options)

                image1 = Image.open(str(sample["image_paths"][0])).convert("RGB")
                image2 = Image.open(str(sample["image_paths"][1])).convert("RGB")
                model_prompt = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image1},
                            {"type": "image", "image": image2},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                batch_prompts_for_model.append(model_prompt)

            parsed_labels = model.get_mcq_predictions(
                batch_prompts=batch_prompts_for_model,
                option_labels=task.get_option_labels(),
                options_text=batch_options_context,
            )

            for parsed_label, final_options in zip(
                parsed_labels, batch_options_context
            ):
                if parsed_label:
                    option_index = ord(parsed_label) - ord("A")
                    if 0 <= option_index < len(final_options):
                        predicted_text = final_options[option_index]
                        all_predicted_option_texts.append(predicted_text)
                    else:
                        all_predicted_option_texts.append(None)
                else:
                    all_predicted_option_texts.append(None)

        # --- 4. Evaluate and store metrics for the sub-database ---
        db_metrics = task.evaluate(all_correct_option_texts, all_predicted_option_texts)

        print(f"\n--- Metrics for {db_id} (MCQ) ---")
        for key, val in db_metrics.items():
            if isinstance(val, (int, float)):
                print(f"  {key.replace('_', ' ').title()}: {val:.4f}")
        print("--------------------------\n")

        overall_results[fvc_year]["Dbs"][db_id] = db_metrics
        overall_results[fvc_year]["db_metrics_list"].append(db_metrics)

    # --- 5. Calculate and Print Averages ---
    for fvc_year, year_results in overall_results.items():
        # Note: We can reuse average_metrics, but it will average different keys (accuracy, etc.)
        avg_metrics = average_metrics(year_results["db_metrics_list"])
        year_results["Average"] = avg_metrics
        del year_results["db_metrics_list"]  # Clean up

        print(f"\n--- Average MCQ Metrics for {fvc_year} ---")
        for key, val in avg_metrics.items():
            print(f"  Avg {key.replace('_', ' ').title()}: {val:.4f}")
        print("-----------------------------------\n")

    # --- 6. Save Final Results ---
    model_name_safe = Path(args.model).name
    output_filename = f"fvc_mcq_{model_name_safe}.pkl"
    output_path = config.OUTPUT_DIR / output_filename

    with open(output_path, "wb") as f:
        pickle.dump(overall_results, f)
    print(f"Overall MCQ results for FVC saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VLM on FVC Fingerprint Verification using MCQ method."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name or path of the VLM on HuggingFace Hub.",
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
        default="cuda:0",
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing pairs.",
    )
    args = parser.parse_args()
    main(args)
