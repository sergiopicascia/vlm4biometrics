import argparse
from pathlib import Path
import math
import pickle
from PIL import Image
from tqdm.auto import tqdm
import torch
import numpy as np
from sklearn.metrics import roc_curve
from transformers import AutoProcessor, AutoModelForCausalLM

LFW_ROOT = Path("./datasets/lfw")
PAIRS_FILE = LFW_ROOT / "pairs.csv"
IMAGE_DIR = LFW_ROOT / "lfw-deepfunneled"
PAIRS_PER_FOLD = 300


def parse_lfw_pairs(pairs_filepath, fold_num=-1):
    """Parses the LFW pairs.csv file for a specific fold.

    Args:
        pairs_filepath (str): Path to the pairs.csv file.
        fold_num (int): The fold number to parse (1-10). Default is -1, to parse all folds.
    """
    pairs = []
    labels = []  # 1 for match, 0 for mismatch

    with open(pairs_filepath, "r") as f:
        lines = f.readlines()

    if fold_num == -1:
        fold_lines = lines[1:]  # Skip the header line
    elif 1 <= fold_num <= 10:
        start_line_index = 1 + (fold_num - 1) * (2 * PAIRS_PER_FOLD)
        end_line_index = start_line_index + 2 * PAIRS_PER_FOLD
        fold_lines = lines[start_line_index:end_line_index]
    else:
        raise ValueError(
            f"Fold number must be between 1 and 10, or -1, got {fold_num}."
        )

    for i, line in enumerate(fold_lines):
        parts = line.strip().split(",")
        if len(parts) == 4 and not parts[-1]:
            # Matched pair
            name, n1, n2, _ = parts
            img1_path = IMAGE_DIR / name / f"{name}_{int(n1):04d}.jpg"
            img2_path = IMAGE_DIR / name / f"{name}_{int(n2):04d}.jpg"
            pairs.append((img1_path, img2_path))
            labels.append(1)
        elif len(parts) == 4:
            # Mismatched pair
            name1, n1, name2, n2 = parts
            img1_path = IMAGE_DIR / name1 / f"{name1}_{int(n1):04d}.jpg"
            img2_path = IMAGE_DIR / name2 / f"{name2}_{int(n2):04d}.jpg"
            pairs.append((img1_path, img2_path))
            labels.append(0)
        else:
            print(f"Warning: Skipping malformed line {i+1}: {line.strip()}")

    return pairs, labels


def calculate_metrics(labels, scores):
    """Calculates FAR, FRR, Accuracy, EER, and AUC."""
    labels = np.array(labels)
    scores = np.array(scores)

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer_threshold = thresholds[eer_index]
    eer = (fpr[eer_index] + fnr[eer_index]) / 2.0

    predictions_at_eer = (scores >= eer_threshold).astype(int)
    accuracy_at_eer = np.mean(predictions_at_eer == labels) * 100

    auc_score = np.trapezoid(tpr, fpr)

    far_at_eer = fpr[eer_index] * 100
    frr_at_eer = fnr[eer_index] * 100

    print("\n--- Face Verification Metrics ---")
    print(f"Equal Error Rate (EER): {eer:.4f}")
    print(f"Threshold at EER: {eer_threshold:.4f}")
    print(f"Accuracy at EER Threshold: {accuracy_at_eer:.2f}%")
    print(f"Area Under ROC Curve (AUC): {auc_score:.4f}")
    print(f"FAR @ EER Threshold: {far_at_eer:.4f}")
    print(f"FRR @ EER Threshold: {frr_at_eer:.4f}")
    print("--------------------------\n")

    return {
        "eer": eer,
        "threshold": eer_threshold,
        "accuracy_at_eer": accuracy_at_eer,
        "auc": auc_score,
        "far_at_eer": far_at_eer,
        "frr_at_eer": frr_at_eer,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def get_normalized_yes_prob(logprob_yes_val, logprob_no_val):
    """
    Calculates the normalized probability of 'Yes' given log probabilities of 'Yes' and 'No'.
    """
    prob_yes = math.exp(logprob_yes_val) if logprob_yes_val > -float("inf") else 0.0
    prob_no = math.exp(logprob_no_val) if logprob_no_val > -float("inf") else 0.0

    total_prob = prob_yes + prob_no
    if total_prob == 0:
        return 0.5  # Uncertain score

    normalized_prob_yes = prob_yes / total_prob
    return normalized_prob_yes


def main(args):
    pairs, labels = parse_lfw_pairs(PAIRS_FILE, args.fold_num)

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

    yes_token_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("no", add_special_tokens=False)[0]

    scores = []
    for i in tqdm(range(0, len(pairs), args.batch_size)):
        batch_indices = range(i, min(i + args.batch_size, len(pairs)))
        batch_pairs = [pairs[j] for j in batch_indices]
        batch_messages = []
        for pair in batch_pairs:
            image1 = Image.open(str(pair[0])).convert("RGB")
            image2 = Image.open(str(pair[1])).convert("RGB")
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
                        {"type": "image", "image": image1},
                        {"type": "image", "image": image2},
                        {
                            "type": "text",
                            "text": "Do these two images show the same person? Answer with only 'yes' or 'no'.",
                        },
                    ],
                },
            ]
            batch_messages.append(messages)

        batch_inputs = processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **batch_inputs,
                max_new_tokens=1,
                do_sample=False,
                top_k=None,
                top_p=None,
                temperature=None,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        first_token_logits_batch = outputs.scores[0]
        first_token_logprobs_batch = torch.log_softmax(first_token_logits_batch, dim=-1)
        for j in range(first_token_logprobs_batch.shape[0]):
            logprobs_for_sample = first_token_logprobs_batch[j]
            logprob_yes = logprobs_for_sample[yes_token_id].item()
            logprob_no = logprobs_for_sample[no_token_id].item()
            normalized_yes_prob = get_normalized_yes_prob(logprob_yes, logprob_no)
            scores.append(normalized_yes_prob)

    metrics = calculate_metrics(labels[: len(scores)], scores)
    with open(
        f"./outputs/lfw_{args.model.split('/')[1]}_fold{args.fold_num if args.fold_num != -1 else '-all'}.pkl",
        "wb",
    ) as f:
        pickle.dump(metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Multimodal LLM on LFW.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name or path of the multimodal LLM on HuggingFace Hub.",
    )
    parser.add_argument(
        "--fold_num",
        type=int,
        default=-1,
        choices=range(1, 11),
        help="Fold number (1-10, or -1 for all) to evaluate on.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        choices=["cpu", "cuda:0", "cuda:1"],
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for processing pairs.",
    )
    args = parser.parse_args()
    main(args)
