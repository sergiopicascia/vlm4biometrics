import argparse
from pathlib import Path
import math
import pickle
from collections import defaultdict
from PIL import Image
from tqdm.auto import tqdm
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from transformers import AutoProcessor, AutoModelForCausalLM

CELEBA_ROOT = Path("./datasets/CelebA")
ATTR_FILE = CELEBA_ROOT / "list_attr_celeba.csv"
PARTITION_FILE = CELEBA_ROOT / "list_eval_partition.csv"
IMAGE_DIR_CELEBA = CELEBA_ROOT / "img_align_celeba" / "img_align_celeba"


def parse_celeba_attributes(
    attr_filepath, partition_filepath, image_dir, partition_num=-1
):
    """Parses the CelebA attributes and partition files.

    Args:
        attr_filepath (str): Path to the list_attr_celeba.csv file.
        partition_filepath (str): Path to the list_eval_partition.csv file.
        image_dir (Path): Path to the directory containing CelebA images.
        partition_num (int): The partition to use (0: train, 1: val, 2: test, -1: all).
    """
    attr_df = pd.read_csv(attr_filepath)
    partition_df = pd.read_csv(partition_filepath)

    # Merge dataframes to easily filter by partition
    df = pd.merge(attr_df, partition_df, on="image_id")

    if partition_num != -1:
        df = df[df["partition"] == partition_num]

    attribute_names = df.columns[
        1:41
    ].tolist()  # First col is image_id, next 40 are attributes

    # Create a list of tasks: (image_path, attribute_name_original, attribute_name_for_prompt, true_label)
    tasks = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = image_dir / row["image_id"]
        for attr_name in attribute_names:
            # Convert attribute name to a more natural language query
            # e.g., "5_o_Clock_Shadow" -> "5 o clock shadow"
            prompt_attr_name = attr_name.replace("_", " ").lower()
            true_label = 1 if row[attr_name] == 1 else 0  # Convert -1/1 to 0/1
            tasks.append((img_path, attr_name, prompt_attr_name, true_label))

    return tasks, attribute_names


def calculate_metrics(labels, scores, attribute_name="Overall"):
    """Calculates FAR, FRR, Accuracy, EER, and AUC."""
    labels = np.array(labels)
    scores = np.array(scores)

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # eer_index = np.nanargmin(np.abs(fpr - fnr))
    # eer_threshold = thresholds[eer_index]
    # eer = (fpr[eer_index] + fnr[eer_index]) / 2.0

    t1_cand = np.where(fnr <= fpr)[0]
    t2_cand = np.where(fnr >= fpr)[0]

    t1 = t1_cand.min()
    t2 = t2_cand.max()

    if (fnr[t1] + fpr[t1]) <= (fnr[t2] + fpr[t2]):
        eer_index = t1
        eer_threshold = thresholds[eer_index]
        eer_low = fnr[t1]
        eer_high = fpr[t1]
    else:
        eer_index = t2
        eer_threshold = thresholds[eer_index]
        eer_low = fpr[t2]
        eer_high = fnr[t2]

    eer = np.mean([eer_low, eer_high])

    predictions_at_eer = (scores >= eer_threshold).astype(int)
    accuracy_at_eer = np.mean(predictions_at_eer == labels) * 100

    auc_score = np.trapezoid(tpr, fpr)

    far_at_eer = fpr[eer_index] * 100
    frr_at_eer = fnr[eer_index] * 100

    print(f"\n--- Soft Biometric Attribute Metrics ({attribute_name}) ---")
    print(f"Equal Error Rate (EER): {eer:.4f}")
    print(f"Threshold at EER: {eer_threshold:.4f}")
    print(f"Accuracy at EER Threshold: {accuracy_at_eer:.2f}%")
    print(f"Area Under ROC Curve (AUC): {auc_score:.4f}")
    print(f"FAR @ EER Threshold: {far_at_eer:.4f}")
    print(f"FRR @ EER Threshold: {frr_at_eer:.4f}")
    print("--------------------------------------------\n")

    return {
        "scores": scores,
        "labels": labels,
        "eer": eer,
        "eer_low": eer_low,
        "eer_high": eer_high,
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
    tasks, attribute_names_list = parse_celeba_attributes(
        ATTR_FILE, PARTITION_FILE, IMAGE_DIR_CELEBA, args.partition_num
    )

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

    per_attribute_scores = defaultdict(list)
    per_attribute_labels = defaultdict(list)
    for i in tqdm(range(0, len(tasks), args.batch_size)):
        batch_tasks = tasks[i : min(i + args.batch_size, len(tasks))]
        batch_messages = []
        processed_images_for_batch = {}  # Cache PIL images within a batch
        for img_path, _, prompt_attr_name, _ in batch_tasks:
            if img_path not in processed_images_for_batch:
                processed_images_for_batch[img_path] = Image.open(
                    str(img_path)
                ).convert("RGB")
            image = processed_images_for_batch[img_path]
            prompt_text = (
                f"Does the person in the image have the attribute '{prompt_attr_name}'? "
                f"Answer with only 'yes' or 'no'."
            )
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
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text},
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
            padding=True,
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

            per_attribute_scores[batch_tasks[j][1]].append(normalized_yes_prob)
            per_attribute_labels[batch_tasks[j][1]].append(batch_tasks[j][3])

    output = {}
    # 1. Calculate Overall Metrics
    overall_flat_scores = []
    overall_flat_labels = []
    for attr_name in attribute_names_list:
        overall_flat_scores.extend(per_attribute_scores.get(attr_name, []))
        overall_flat_labels.extend(per_attribute_labels.get(attr_name, []))

    output["overall"] = calculate_metrics(
        overall_flat_labels,
        overall_flat_scores,
        attribute_name="Overall",
    )

    # 2. Calculate Per-Attribute Metrics
    for attr_name in attribute_names_list:
        scores_for_attr = per_attribute_scores.get(attr_name, [])
        labels_for_attr = per_attribute_labels.get(attr_name, [])

        attr_metrics_result = calculate_metrics(
            labels_for_attr, scores_for_attr, attribute_name=attr_name
        )
        output[attr_name] = attr_metrics_result

    with open(
        f"./outputs/celeba_{args.model.split('/')[1]}_partition{args.partition_num if args.partition_num != -1 else '-all'}.pkl",
        "wb",
    ) as f:
        pickle.dump(output, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Multimodal LLM on CelebA.")
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
        choices=[-1, 0, 1, 2],  # -1 for all, 0 for train, 1 for val, 2 for test
        help="CelebA partition to evaluate on (-1: all, 0: train, 1: val, 2: test).",
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
