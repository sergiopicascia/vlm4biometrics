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
from src.data_loaders import AgeDBLoader
from src.tasks import AgeEstimationMCQTask, GenderPredictionMCQTask
from src.utils import get_model


def main(args):
    # --- 1. Setup ---
    loader = AgeDBLoader()
    model = get_model(model_path=args.model, device=args.device)

    # --- 2. Data Loading ---
    samples = loader.load()

    # --- 3. Define and Run Tasks Sequentially ---
    tasks_to_run = {
        "age": AgeEstimationMCQTask(),
        "gender": GenderPredictionMCQTask(),
    }

    for task_name, task_instance in tasks_to_run.items():
        print(f"\n===== Running Task: {task_name.title()} Prediction (MCQ) =====")

        all_correct_option_texts = []
        all_predicted_option_texts = []

        for i in tqdm(
            range(0, len(samples), args.batch_size),
            desc=f"Processing for {task_name} (MCQ)",
        ):
            batch_samples = samples[i : min(i + args.batch_size, len(samples))]

            batch_prompts_for_model = []
            batch_options_context = []

            for sample in batch_samples:
                prompt_text, final_options = task_instance.generate_prompt_and_options(
                    sample
                )

                all_correct_option_texts.append(
                    task_instance._get_correct_option_text(sample)
                )
                batch_options_context.append(final_options)

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

            # Call the model's method to handle generation and parsing
            parsed_labels = model.get_mcq_predictions(
                batch_prompts=batch_prompts_for_model,
                option_labels=task_instance.get_option_labels(),
                options_text=batch_options_context,
            )

            # Convert the parsed label ('A', 'B', etc.) back to its meaningful text
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

        # --- 4. Evaluate ---
        # The task's evaluate method calls calculate_classification_metrics
        metrics = task_instance.evaluate(
            all_correct_option_texts, all_predicted_option_texts
        )

        # --- 5. Save Results ---
        model_name_safe = Path(args.model).name
        output_filename = f"agedb_mcq_{task_name}_{model_name_safe}.pkl"
        output_path = config.OUTPUT_DIR / output_filename

        with open(output_path, "wb") as f:
            pickle.dump(metrics, f)
        print(f"Results for {task_name} (MCQ) saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VLM on AgeDB using MCQ method."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name or path of the VLM on HuggingFace Hub.",
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
        help="Batch size for processing images.",
    )
    args = parser.parse_args()
    main(args)
