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


FVC_BASE_DIR = Path("./datasets/fvc200X")
FVC_DATASET_NAMES = ["FVC2000", "FVC2002", "FVC2004"]
FVC_DBS = ["Db1_", "Db2_", "Db3_", "Db4_"]


def _parse_fvc_index_file(
    index_file_path: Path,
    image_dir: Path,
    label: int,
    pairs_list: list,
    labels_list: list,
):
    with open(index_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            img1_name, img2_name = parts
            img1_path = image_dir / img1_name
            img2_path = image_dir / img2_name

            pairs_list.append((img1_path, img2_path))
            labels_list.append(label)


def parse_fvc_db_pairs(fvc_dataset_name: str, db_name: str, base_fvc_path: Path):
    """
    Parses genuine and impostor pairs for a specific FVC sub-database (e.g., FVC2000 Db1).
    The index files (index_a.MFA, index_B.MFA, etc.) are assumed to contain filenames
    relative to the specific DbX_a or DbX_b image directory.
    """
    pairs = []
    labels = []
    fvc_year_root = base_fvc_path / fvc_dataset_name  # "./datasets/fvc200X/FVC2000"
    fvc_db_index_dir = fvc_year_root / "Dbs"  # "./datasets/fvc200X/FVC2000/Dbs"

    # Process Set A pairs
    image_dir_a = fvc_db_index_dir / f"{db_name}a"
    _parse_fvc_index_file(
        fvc_db_index_dir / "index_a.MFA",
        image_dir_a,
        1,
        pairs,
        labels,
    )
    _parse_fvc_index_file(
        fvc_db_index_dir / "index_a.MFR",
        image_dir_a,
        0,
        pairs,
        labels,
    )

    # Process Set B pairs
    image_dir_b = fvc_db_index_dir / f"{db_name}b"
    _parse_fvc_index_file(
        fvc_db_index_dir / "index_B.MFA",
        image_dir_b,
        1,
        pairs,
        labels,
    )
    _parse_fvc_index_file(
        fvc_db_index_dir / "index_B.MFR",
        image_dir_b,
        0,
        pairs,
        labels,
    )

    return pairs, labels


def calculate_metrics(labels, scores, db_id_for_print=""):
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

    print(f"\n--- Fingerprint Verification Metrics for {db_id_for_print} ---")
    print(f"Equal Error Rate (EER): {eer:.4f}")
    print(f"Threshold at EER: {eer_threshold:.4f}")
    print(f"Accuracy at EER Threshold: {accuracy_at_eer:.2f}%")
    print(f"Area Under ROC Curve (AUC): {auc_score:.4f}")
    print(f"FAR @ EER Threshold: {far_at_eer:.4f}%")
    print(f"FRR @ EER Threshold: {frr_at_eer:.4f}%")
    print("---------------------------------------\n")

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


def average_metrics_list(metrics_list: list):
    avg_metrics = {
        key: np.mean([m[key] for m in metrics_list])
        for key in [
            "eer",
            "accuracy_at_eer",
            "auc",
            "far_at_eer",
            "frr_at_eer",
            "threshold",
        ]
    }

    return avg_metrics


def get_normalized_yes_prob(logprob_yes_val, logprob_no_val):
    """
    Calculates the normalized probability of 'Yes' given log probabilities of 'Yes' and 'No'.
    """
    prob_yes = math.exp(logprob_yes_val) if logprob_yes_val > -float("inf") else 0.0
    prob_no = math.exp(logprob_no_val) if logprob_no_val > -float("inf") else 0.0

    total_prob = prob_yes + prob_no
    return 0.5 if total_prob == 0 else prob_yes / total_prob


def main(args):
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

    overall_results = {}
    fvc_datasets_to_run = args.fvc_datasets if args.fvc_datasets else FVC_DATASET_NAMES
    for fvc_year_name in fvc_datasets_to_run:
        print(f"\n===== Processing FVC Dataset: {fvc_year_name} =====")
        current_fvc_year_results = {"Dbs": {}}
        db_metrics_for_averaging = []
        for db_name in FVC_DBS:
            db_id = f"{fvc_year_name}-{db_name}"
            print(f"\n--- Processing Sub-database: {db_id} ---")

            pairs, labels = parse_fvc_db_pairs(
                fvc_year_name, db_name, Path(FVC_BASE_DIR)
            )
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
                                    "text": "Do these two fingerprint images belong to the same finger? Answer with only 'yes' or 'no'.",
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
                first_token_logprobs_batch = torch.log_softmax(
                    outputs.scores[0], dim=-1
                )
                for j in range(first_token_logprobs_batch.shape[0]):
                    logprobs_sample = first_token_logprobs_batch[j]
                    logprob_yes = logprobs_sample[yes_token_id].item()
                    logprob_no = logprobs_sample[no_token_id].item()
                    scores.append(get_normalized_yes_prob(logprob_yes, logprob_no))

            db_metrics = calculate_metrics(labels, scores, db_id_for_print=db_id)
            current_fvc_year_results["Dbs"][db_name] = db_metrics
            db_metrics_for_averaging.append(db_metrics)

        # Average metrics for the current FVC dataset year
        avg_metrics = average_metrics_list(db_metrics_for_averaging)
        current_fvc_year_results["Average"] = avg_metrics
        print(f"\n--- Average Metrics for {fvc_year_name} ---")
        for key, val in avg_metrics.items():
            print(f"  Avg {key.replace('_', ' ').title()}: {val:.4f}")
        print("-----------------------------------\n")

        overall_results[fvc_year_name] = current_fvc_year_results

    with open(f"./outputs/fvc_{args.model.split('/')[1]}.pkl", "wb") as f:
        pickle.dump(overall_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Multimodal LLM on FVC Fingerprint Verification."
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
        default=None,  # Default will use FVC_DATASET_NAMES
        help=f"List of FVC datasets to process. Default: all {FVC_DATASET_NAMES}.",
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
