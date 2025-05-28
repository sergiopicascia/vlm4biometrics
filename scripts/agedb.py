import argparse
from pathlib import Path
import math
import pickle
import glob
from PIL import Image
from tqdm.auto import tqdm
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
)
from transformers import AutoProcessor, AutoModelForCausalLM

AGEDB_IMAGE_DIR = Path("./datasets/AgeDB/")

MIN_AGE = 1
MAX_AGE = 101
AGE_PROMPT_TEXT = "How old is the person in this image? Answer with only a number representing their age."
GENDER_PROMPT_TEXT = "What is the gender of the person in this image? Answer with only 'male' or 'female'."
TARGET_END_OF_TURN_TOKEN_STR = "<end_of_turn>"


def parse_agedb_dataset(image_dir_path: Path):
    """Parses the AgeDB dataset from the given directory.

    Filename format: "ID_NameSurname_Age_Gender.jpg"
    """
    image_paths = []
    true_ages = []
    true_genders = []  # Store as 'm' or 'f' initially

    # Using glob to find all .jpg files, can be adjusted for other extensions
    for filepath_str in glob.glob(str(image_dir_path / "*.jpg")):
        filepath = Path(filepath_str)
        filename = filepath.name
        parts = filename.rsplit(".", 1)[0].split("_")

        if len(parts) < 4:
            print(
                f"Warning: Skipping malformed filename (not enough parts): {filename}"
            )
            continue

        age_str = parts[2]
        gender_char = parts[3]

        age = int(age_str)
        if not (MIN_AGE <= age <= MAX_AGE):
            print(f"Warning: Skipping image with out-of-range age {age}: {filename}")
            continue
        if gender_char not in ["m", "f"]:
            print(
                f"Warning: Skipping image with invalid gender '{gender_char}': {filename}"
            )
            continue

        image_paths.append(filepath)
        true_ages.append(age)
        true_genders.append(gender_char)

    return image_paths, true_ages, true_genders


def calculate_age_metrics(true_ages, predicted_ages):
    """Calculates metrics for age estimation."""
    true_ages_np = np.array(true_ages)
    predicted_ages_np = np.array(predicted_ages)

    mae = mean_absolute_error(true_ages_np, predicted_ages_np)
    errors = (
        predicted_ages_np - true_ages_np
    )  # Positive means overestimation, negative means underestimation

    # Cumulative Score (CS)
    cs_thresholds = [1, 3, 5, 7, 10]  # years
    cs_scores = {}
    for t in cs_thresholds:
        cs_scores[f"CS@{t}yrs"] = np.mean(np.abs(errors) <= t) * 100

    print("\n--- Age Estimation Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print("Cumulative Scores (CS):")
    for t, score in cs_scores.items():
        print(f"  {t}: {score:.2f}%")
    print(f"Standard Deviation of Errors: {np.std(errors):.4f}")
    print("------------------------------\n")

    return {
        "mae": mae,
        "cs_scores": cs_scores,
        "error_std": np.std(errors),
        "errors": errors.tolist(),  # For potential further analysis
    }


def calculate_gender_metrics(
    true_genders, predicted_genders, labels=["female", "male"]
):
    """Calculates metrics for gender prediction."""
    # Ensure true_genders are in the same format as predicted_genders ('female', 'male')
    true_genders_mapped = ["female" if g == "f" else "male" for g in true_genders]

    accuracy = accuracy_score(true_genders_mapped, predicted_genders) * 100

    print("\n--- Gender Prediction Metrics ---")
    print(f"Accuracy: {accuracy:.2f}%")

    try:
        cm = confusion_matrix(true_genders_mapped, predicted_genders, labels=labels)
        print("Confusion Matrix:")
        print(cm)
        # Ensure there are predicted samples for both classes for a full report
        # Or handle cases where some classes might not be predicted (less likely for male/female)
        if len(set(true_genders_mapped)) > 1 and len(set(predicted_genders)) > 1:
            report = classification_report(
                true_genders_mapped, predicted_genders, labels=labels, zero_division=0
            )
            print("Classification Report:")
            print(report)
        else:
            print(
                "Classification report cannot be generated (likely due to only one class present in true or predictions)."
            )

    except Exception as e:
        print(f"Could not generate detailed classification report/CM: {e}")

    print("------------------------------\n")

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist() if "cm" in locals() else None,
        "classification_report": report if "report" in locals() else None,
    }


def get_normalized_male_prob(logprob_male_val, logprob_female_val):
    """
    Calculates the normalized probability of 'male' given log probabilities of 'male' and 'female'.
    """
    prob_male = math.exp(logprob_male_val) if logprob_male_val > -float("inf") else 0.0
    prob_female = (
        math.exp(logprob_female_val) if logprob_female_val > -float("inf") else 0.0
    )

    total_prob = prob_male + prob_female
    if total_prob == 0:
        return 0.5  # Uncertain score

    normalized_prob_male = prob_male / total_prob
    return normalized_prob_male


def main(args):
    image_paths, true_ages_all, true_genders_all = parse_agedb_dataset(AGEDB_IMAGE_DIR)

    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    tokenizer = processor.tokenizer

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    end_of_turn_token_id = tokenizer.encode(
        TARGET_END_OF_TURN_TOKEN_STR, add_special_tokens=False
    )[0]

    # --- Tokenization for Age Task ---
    max_tokens_for_age_number = 0
    for age_val in range(MIN_AGE, MAX_AGE + 1):
        max_tokens_for_age_number = max(
            max_tokens_for_age_number,
            len(tokenizer.encode(str(age_val), add_special_tokens=False)),
        )
    MAX_NEW_TOKENS_FOR_AGE_TASK = (
        max_tokens_for_age_number + 1
    )  # +1 for the end-of-turn token

    # --- Tokenization for Gender Task ---
    male_token_ids = tokenizer.encode("male", add_special_tokens=False)
    female_token_ids = tokenizer.encode("female", add_special_tokens=False)
    MAX_NEW_TOKENS_FOR_GENDER_TASK = max(len(male_token_ids), len(female_token_ids))

    # --- Store predictions ---
    predicted_ages_list = []
    predicted_genders_list = []

    for i in tqdm(range(0, len(image_paths), args.batch_size)):
        batch_indices = range(i, min(i + args.batch_size, len(image_paths)))
        batch_image_pil_list = [
            Image.open(image_paths[idx]).convert("RGB") for idx in batch_indices
        ]

        # === 1. Age Estimation ===
        batch_age_messages = []
        for img_pil in batch_image_pil_list:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant."}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_pil},
                        {"type": "text", "text": AGE_PROMPT_TEXT},
                    ],
                },
            ]
            batch_age_messages.append(messages)

        age_inputs = processor.apply_chat_template(
            batch_age_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.inference_mode():
            age_outputs = model.generate(
                **age_inputs,
                max_new_tokens=MAX_NEW_TOKENS_FOR_AGE_TASK,
                do_sample=False,
                top_k=None,
                top_p=None,
                temperature=None,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        for batch_sample_idx in range(len(batch_image_pil_list)):
            log_probs_for_each_age = []
            for age_candidate in range(MIN_AGE, MAX_AGE + 1):
                age_candidate_str = str(age_candidate)
                token_ids_for_age_candidate = tokenizer.encode(
                    age_candidate_str, add_special_tokens=False
                )
                # We need probability of "age_candidate_str" + EOS
                token_ids_for_age_candidate_eot = token_ids_for_age_candidate + [
                    end_of_turn_token_id
                ]

                if len(token_ids_for_age_candidate_eot) > len(age_outputs.scores):
                    # If the candidate age sequence is longer than the model's output, skip it
                    log_probs_for_each_age.append(-float("inf"))
                    continue

                current_age_log_prob = 0.0
                for token_step_idx, token_id_to_match in enumerate(
                    token_ids_for_age_candidate_eot
                ):
                    # scores[token_step_idx] contains logits for (token_step_idx+1)-th token
                    # Logits for current sample in batch: age_outputs.scores[token_step_idx][batch_sample_idx]
                    step_logits = age_outputs.scores[token_step_idx][
                        batch_sample_idx, :
                    ]
                    step_log_probs = torch.log_softmax(step_logits, dim=-1)
                    current_age_log_prob += step_log_probs[token_id_to_match].item()

                log_probs_for_each_age.append(current_age_log_prob)

            # Convert log_probs to probabilities and find the most likely age
            age_probs_tensor = torch.softmax(
                torch.tensor(log_probs_for_each_age, device=device), dim=0
            )
            predicted_age_index = torch.argmax(age_probs_tensor).item()
            predicted_age = (
                MIN_AGE + predicted_age_index
            )  # Index 0 corresponds to MIN_AGE
            predicted_ages_list.append(predicted_age)

        # === 2. Gender Prediction ===
        batch_gender_messages = []
        for img_pil in batch_image_pil_list:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant."}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_pil},
                        {"type": "text", "text": GENDER_PROMPT_TEXT},
                    ],
                },
            ]
            batch_gender_messages.append(messages)

        gender_inputs = processor.apply_chat_template(
            batch_gender_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,  # Important: padding=True
        ).to(device)

        with torch.inference_mode():
            gender_outputs = model.generate(
                **gender_inputs,
                max_new_tokens=MAX_NEW_TOKENS_FOR_GENDER_TASK,
                do_sample=False,
                top_k=None,
                top_p=None,
                temperature=None,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        for batch_sample_idx in range(len(batch_image_pil_list)):
            # Calculate log probability for the "male" sequence
            log_prob_male_seq = 0.0
            if len(male_token_ids) <= MAX_NEW_TOKENS_FOR_GENDER_TASK:
                for token_step_idx, token_id_to_match in enumerate(male_token_ids):
                    step_logits = gender_outputs.scores[token_step_idx][
                        batch_sample_idx, :
                    ]
                    step_log_probs = torch.log_softmax(step_logits, dim=-1)
                    log_prob_male_seq += step_log_probs[token_id_to_match].item()
            else:
                log_prob_male_seq = -float("inf")  # Sequence too long

            # Calculate log probability for the "female" sequence
            log_prob_female_seq = 0.0
            if len(female_token_ids) <= MAX_NEW_TOKENS_FOR_GENDER_TASK:
                for token_step_idx, token_id_to_match in enumerate(female_token_ids):
                    step_logits = gender_outputs.scores[token_step_idx][
                        batch_sample_idx, :
                    ]
                    step_log_probs = torch.log_softmax(step_logits, dim=-1)
                    log_prob_female_seq += step_log_probs[token_id_to_match].item()
            else:
                log_prob_female_seq = -float("inf")  # Sequence too long

            normalized_prob_male = get_normalized_male_prob(
                log_prob_male_seq, log_prob_female_seq
            )

            predicted_gender = "male" if normalized_prob_male >= 0.5 else "female"
            predicted_genders_list.append(predicted_gender)

        # Clear some memory
        del age_inputs, age_outputs, gender_inputs, gender_outputs, batch_image_pil_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Calculate and Save Metrics ---
    # Ensure we only use ground truth for the images processed (if there were skips)
    # This is handled implicitly as predicted_lists will match processed images.

    age_metrics = calculate_age_metrics(true_ages_all, predicted_ages_list)
    gender_metrics = calculate_gender_metrics(true_genders_all, predicted_genders_list)

    with open(
        f"./outputs/agedb-age_{args.model.split('/')[-1]}_results.pkl", "wb"
    ) as f:
        pickle.dump(age_metrics, f)

    with open(
        f"./outputs/agedb-gender_{args.model.split('/')[-1]}_results.pkl", "wb"
    ) as f:
        pickle.dump(gender_metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Multimodal LLM on AgeDB.")
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
