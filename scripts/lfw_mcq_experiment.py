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
from src.data_loaders import LFWLoader
from src.tasks import VerificationMCQTask
from src.utils import get_model


def main(args):
    loader = LFWLoader()
    task = VerificationMCQTask(domain="person")
    model = get_model(model_path=args.model, device=args.device)

    # Data Loading
    samples = loader.load()

    # Run MCQ Inference and Parsing
    all_correct_option_texts = []
    all_predicted_option_texts = []

    for i in tqdm(
        range(0, len(samples), args.batch_size), desc="Processing Batches (MCQ)"
    ):
        batch_samples = samples[i : min(i + args.batch_size, len(samples))]
        batch_prompts_for_model = []
        batch_options_context = (
            []
        )  # Store the ordered options for each sample in the batch

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

        # Convert the parsed label ('A', 'B') back to its meaningful text ('Yes', 'No')
        for parsed_label, final_options in zip(parsed_labels, batch_options_context):
            if parsed_label:
                option_index = ord(parsed_label) - ord("A")
                if 0 <= option_index < len(final_options):
                    predicted_text = final_options[option_index]
                    all_predicted_option_texts.append(predicted_text)
                else:
                    all_predicted_option_texts.append(None)
            else:
                all_predicted_option_texts.append(None)

    # Evaluate ---
    metrics = task.evaluate(all_correct_option_texts, all_predicted_option_texts)

    # Save Results ---
    model_name_safe = Path(args.model).name
    output_filename = f"lfw_mcq_{model_name_safe}.pkl"
    output_path = config.OUTPUT_DIR / output_filename

    with open(output_path, "wb") as f:
        pickle.dump(metrics, f)
    print(f"\nMCQ-based experiment results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VLM on LFW using MCQ method."
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
        help="Batch size for processing pairs.",
    )
    args = parser.parse_args()
    main(args)
