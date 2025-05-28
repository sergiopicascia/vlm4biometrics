import argparse
from pathlib import Path
import math
import random
import pickle
from PIL import Image
from tqdm.auto import tqdm
import torch
import numpy as np
from sklearn.metrics import roc_curve
from transformers import AutoProcessor, AutoModelForCausalLM

CASIA_ROOT = Path("./datasets/casia-iris-thousand/CASIA-Iris-Thousand")


def get_all_iris_images_by_subject_eye(dataset_root: Path):
    """
    Scans the CASIA-Iris-Thousand dataset and organizes images by subject and eye.
    Returns a dictionary: {(subject_id, eye_type): [list_of_image_paths]}
    """
    images_by_subject_eye = {}
    subject_ids = sorted(
        [d.name for d in dataset_root.iterdir() if d.is_dir() and d.name.isdigit()]
    )

    for subject_id in tqdm(subject_ids, desc="Scanning dataset"):
        subject_dir = dataset_root / subject_id
        for eye_type in ["L", "R"]:
            eye_dir = subject_dir / eye_type
            if eye_dir.exists() and eye_dir.is_dir():
                image_paths = sorted([p for p in eye_dir.glob("*.jpg")])
                if image_paths:
                    images_by_subject_eye[(subject_id, eye_type)] = image_paths
    return images_by_subject_eye


def generate_iris_pairs(dataset_root: Path, num_total_pairs: int, random_seed: int):
    """
    Generates genuine and impostor pairs for iris verification.

    Args:
        dataset_root (Path): Path to the CASIA-Iris-Thousand dataset.
        num_total_pairs (int): Total number of pairs to generate.
        random_seed (int): Random seed for reproducibility.
    """
    random.seed(random_seed)

    images_by_subject_eye = get_all_iris_images_by_subject_eye(dataset_root)
    subject_eye_keys = list(images_by_subject_eye.keys())

    pairs = []
    labels = []  # 1 for genuine, 0 for impostor

    num_genuine_pairs = num_total_pairs // 2
    num_impostor_pairs = num_total_pairs - num_genuine_pairs

    # --- Generate Genuine Pairs ---
    while len(pairs) < num_genuine_pairs:
        subject_id, eye_type = random.choice(subject_eye_keys)
        image_list = images_by_subject_eye[(subject_id, eye_type)]
        img1_path, img2_path = random.sample(image_list, 2)
        pairs.append((img1_path, img2_path))
        labels.append(1)

    # --- Generate Impostor Pairs ---
    all_subject_ids = sorted(list(set(key[0] for key in subject_eye_keys)))
    while len(pairs) - num_genuine_pairs < num_impostor_pairs:
        # Select two different subject IDs
        s_id1, s_id2 = random.sample(all_subject_ids, 2)
        # Select a random eye type for each subject
        eye_type = random.choice(["L", "R"])

        # Get all (subject_id, eye_type) keys for these subjects
        keys_for_s1 = [
            key for key in subject_eye_keys if key[0] == s_id1 and key[1] == eye_type
        ]
        keys_for_s2 = [
            key for key in subject_eye_keys if key[0] == s_id2 and key[1] == eye_type
        ]

        # Pick a random (subject_id, eye_type) and then a random image for each subject
        key1 = random.choice(keys_for_s1)
        img1_path = random.choice(images_by_subject_eye[key1])

        key2 = random.choice(keys_for_s2)
        img2_path = random.choice(images_by_subject_eye[key2])

        pairs.append((img1_path, img2_path))
        labels.append(0)

    # Shuffle pairs and labels together
    combined = list(zip(pairs, labels))
    random.shuffle(combined)
    pairs, labels = zip(*combined)
    pairs = list(pairs)
    labels = list(labels)

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

    print("\n--- Iris Verification Metrics ---")
    print(f"Equal Error Rate (EER): {eer:.4f}")
    print(f"Threshold at EER: {eer_threshold:.4f}")
    print(f"Accuracy at EER Threshold: {accuracy_at_eer:.2f}%")
    print(f"Area Under ROC Curve (AUC): {auc_score:.4f}")
    print(f"FAR @ EER Threshold: {far_at_eer:.4f}%")  # False Acceptance Rate
    print(f"FRR @ EER Threshold: {frr_at_eer:.4f}%")  # False Rejection Rate
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
    return 0.5 if total_prob == 0 else prob_yes / total_prob


def main(args):
    pairs, labels = generate_iris_pairs(CASIA_ROOT, args.num_pairs, args.random_seed)

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
            image1 = Image.open(str(pair[0]))
            image2 = Image.open(str(pair[1]))
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
                            "text": "Do these two iris images belong to the same person? Answer with only 'yes' or 'no'.",
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

    metrics = calculate_metrics(labels, scores)

    with open(
        f"./outputs/casia-iris_{args.model.split('/')[1]}_s{args.random_seed}_n{args.num_pairs}.pkl",
        "wb",
    ) as f:
        pickle.dump(metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Multimodal LLM on Iris Verification (CASIA-Iris-Thousand)."
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
        default=20000,  # As requested
        help="Total number of image pairs to generate and test.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for pair generation reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
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
