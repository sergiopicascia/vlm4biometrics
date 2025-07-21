"""
Model handlers for different vision-language models.
Each handler encapsulates model-specific loading and inference logic.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForCausalLM


class BaseModel(ABC):
    """Abstract base class for Vision Language Models."""

    @abstractmethod
    def __init__(self, model_path: str, device: str, **kwargs):
        """Initializes the model, tokenizer, and processor."""
        self.model_path = model_path
        self.device = device

    @abstractmethod
    def get_label_scores(
        self, batch_prompts: List[List[Dict[str, Any]]], target_labels: List[str]
    ) -> np.ndarray:
        """
        Calculates the normalized log-probabilities (scores) for each target label
        for every prompt in the batch.

        Args:
            batch_prompts: A list of prompts, where each prompt is in a chat-like format.
            target_labels: A list of strings representing the possible output labels (e.g., ['no', 'yes']).

        Returns:
            A numpy array of shape (batch_size, num_target_labels) containing the scores.
        """
        pass


class GemmaModel(BaseModel):
    """A wrapper for Gemma `transformers` implementation."""

    def __init__(self, model_path: str, device: str):
        super().__init__(model_path, device)
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.model.eval()
        self.tokenizer = self.processor.tokenizer

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.eot_token = "<end_of_turn>"
        self.eot_token_id = self.tokenizer.encode(
            self.eot_token, add_special_tokens=False
        )

    def get_label_scores(
        self, batch_prompts: List[List[Dict[str, Any]]], target_labels: List[str]
    ) -> np.ndarray:
        label_sequences = []
        for label_text in target_labels:
            token_ids = self.tokenizer.encode(label_text, add_special_tokens=False)
            label_sequences.append(token_ids + self.eot_token_id)

        inputs = self.processor.apply_chat_template(
            batch_prompts,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        batch_scores = []
        with torch.inference_mode():
            for seq in label_sequences:
                temp_inputs = inputs.copy()
                for token_id in seq:
                    # Append each token ID to the input sequence
                    temp_inputs["input_ids"] = torch.cat(
                        [
                            temp_inputs["input_ids"],
                            torch.tensor(
                                [[token_id]] * temp_inputs["input_ids"].shape[0],
                                device=self.device,
                            ),
                        ],
                        dim=-1,
                    )
                    temp_inputs["attention_mask"] = torch.cat(
                        [
                            temp_inputs["attention_mask"],
                            torch.ones(
                                (temp_inputs["attention_mask"].shape[0], 1),
                                device=self.device,
                            ),
                        ],
                        dim=-1,
                    )
                    temp_inputs["token_type_ids"] = torch.cat(
                        [
                            temp_inputs["token_type_ids"],
                            torch.zeros(
                                (temp_inputs["token_type_ids"].shape[0], 1),
                                device=self.device,
                            ),
                        ],
                        dim=-1,
                    )
                # Get logits for the full sequence
                full_sequence_logits = self.model(**temp_inputs).logits

                # Extract logits for the sequence after the prompt
                prompt_len = inputs.input_ids.shape[1]
                sequence_logits = full_sequence_logits[:, prompt_len - 1 : -1, :]
                log_probs = torch.log_softmax(
                    sequence_logits, dim=-1, dtype=torch.float32
                )

                # Gather log probabilities for the target sequence
                seq_tensor = (
                    torch.tensor(seq, device=self.device)
                    .unsqueeze(0)
                    .expand(log_probs.shape[0], -1)
                )
                gathered_log_probs = torch.gather(
                    log_probs, 2, seq_tensor.unsqueeze(-1)
                ).squeeze(-1)

                # Calculate average log probability for the sequence
                avg_log_prob = gathered_log_probs.mean(dim=-1)
                batch_scores.append(avg_log_prob)

        # Stack scores for all labels
        scores_tensor = torch.stack(batch_scores, dim=1)
        probs = torch.softmax(scores_tensor, dim=-1)
        return probs.cpu().numpy()
