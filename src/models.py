"""
Model handlers for different vision-language models.
Each handler encapsulates model-specific loading and inference logic.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type
import numpy as np
import torch
from transformers import (
    PreTrainedModel,
    AutoProcessor,
    AutoModelForCausalLM,
    Qwen3VLForConditionalGeneration,
    InternVLForConditionalGeneration,
)
from .utils import extract_option_label


class BaseModel(ABC):
    """Abstract base class for Vision Language Models."""

    def __init__(
        self,
        model_path: str,
        device: str,
        model_cls: Type[PreTrainedModel] = AutoModelForCausalLM,
        eot_token: str = "<|im_end|>",
        **kwargs,
    ):
        """
        Initializes the model, tokenizer, and processor.

        Args:
            model_path: Path to the pretrained model.
            device: Device to load the model on (e.g., 'cuda').
            model_cls: The specific Transformers model class to load.
            eot_token: The string representation of the End-Of-Turn token.
        """
        self.model_path = model_path
        self.device = device
        self.eot_token = eot_token

        # Load Processor
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        self.tokenizer = self.processor.tokenizer

        # Ensure pad token exists
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load Model
        self.model = (
            model_cls.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.bfloat16, **kwargs
            )
            .eval()
            .to(self.device)
        )

        # Cache EOT token ID
        self.eot_token_id = self.tokenizer.encode(
            self.eot_token, add_special_tokens=False
        )

    def _prepare_inference_inputs(self, base_inputs, label_seq_ids):
        """
        Dynamically constructs the input dictionary for the model by concatenating
        the prompt inputs with the label sequence inputs.

        Handles generic keys (input_ids, attention_mask) and model-specific keys
        (token_type_ids, image_grid_thw) automatically.
        """
        batch_size = base_inputs.input_ids.shape[0]

        # Create the label tensor repeated for the batch
        label_tensor = (
            torch.tensor(label_seq_ids, device=self.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        new_inputs = {}
        # Iterate over all keys returned by the processor
        for k, v in base_inputs.items():
            if k == "input_ids":
                new_inputs[k] = torch.cat([v, label_tensor], dim=1)
            elif k == "attention_mask":
                ones = torch.ones_like(label_tensor, device=self.device)
                new_inputs[k] = torch.cat([v, ones], dim=1)
            elif k == "token_type_ids":
                zeros = torch.zeros_like(label_tensor, device=self.device)
                new_inputs[k] = torch.cat([v, zeros], dim=1)
            else:
                # Pass through non-sequence args (pixel_values, image_grid_thw, etc.)
                new_inputs[k] = v

        return new_inputs

    def get_log_probs(
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
        # Pre-calculate token IDs for all target labels
        label_sequences = []
        for label_text in target_labels:
            token_ids = self.tokenizer.encode(label_text, add_special_tokens=False)
            if isinstance(token_ids[0], list):
                token_ids = [item for sublist in token_ids for item in sublist]
            label_sequences.append(token_ids + self.eot_token_id)

        # Process the prompts (images + text)
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
                # Prepare inputs for this specific label candidate
                seq_inputs = self._prepare_inference_inputs(inputs, seq)

                # Get logits for the full sequence
                full_sequence_logits = self.model(**seq_inputs).logits

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

                # Calculate sum of log probabilities for the sequence
                avg_log_prob = gathered_log_probs.sum(dim=-1)
                batch_scores.append(avg_log_prob)

        # Stack scores for all labels and normalize
        scores_tensor = torch.stack(batch_scores, dim=1)
        probs = torch.softmax(scores_tensor, dim=-1)
        return probs.cpu().numpy()

    def get_mcq_predictions(
        self,
        batch_prompts: List[List[Dict[str, Any]]],
        option_labels: List[str],
        options_text: List[str],
    ) -> List[Optional[str]]:
        """
        Generates text for a batch of prompts and parses the output to find
        the selected multiple-choice option.

        Args:
            batch_prompts: A list of prompts, where each prompt is in a chat-like format.
            option_labels: A list of strings representing the option labels (e.g., ['A', 'B', 'C', 'D']).
            options_text: A list of strings representing the full text of each option.

        Returns:
            A list of selected option labels for each prompt.
        """
        inputs = self.processor.apply_chat_template(
            batch_prompts,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                top_k=None,
                top_p=None,
                temperature=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode generated sequences, skipping prompt tokens
        decoded_outputs = self.tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        # Parse each decoded output string
        parsed_predictions = [
            extract_option_label(output, option_labels, options_text)
            for output in decoded_outputs
        ]

        return parsed_predictions


class Gemma3Model(BaseModel):
    """A wrapper for Gemma `transformers` implementation."""

    def __init__(self, model_path: str, device: str):
        super().__init__(
            model_path,
            device,
            model_cls=AutoModelForCausalLM,
            eot_token="<end_of_turn>",
        )


class Qwen3VLModel(BaseModel):
    """A wrapper for Qwen3-VL models."""

    def __init__(self, model_path: str, device: str):
        super().__init__(
            model_path,
            device,
            model_cls=Qwen3VLForConditionalGeneration,
            eot_token="<|im_end|>",
        )


class InternVL3_5Model(BaseModel):
    """A wrapper for InternVL3.5 models."""

    def __init__(self, model_path: str, device: str):
        super().__init__(
            model_path,
            device,
            model_cls=InternVLForConditionalGeneration,
            eot_token="<|im_end|>",
        )
