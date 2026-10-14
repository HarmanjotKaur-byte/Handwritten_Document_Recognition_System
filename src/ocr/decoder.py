"""Connectionist Temporal Classification (CTC) decoding and character mapping.

Maps character strings to label IDs and decodes raw network log-probabilities
into readable text sequences using greedy collapsing.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch

from src.config import OCR_CHARACTERS


class CTCDecoder:
    """Encodes text to integer labels and decodes sequence outputs from CTC models."""

    def __init__(self, characters: str = OCR_CHARACTERS):
        """Initialize CTCDecoder.

        Args:
            characters (str): Distinct characters recognized by the model.
        """
        self.characters = characters
        # Index 0 is reserved for the CTC Blank token
        self.blank_idx = 0

        # Mapping: character -> index (1-indexed)
        self.char_to_idx: Dict[str, int] = {
            char: idx + 1 for idx, char in enumerate(self.characters)
        }
        # Mapping: index -> character
        self.idx_to_char: Dict[int, str] = {
            idx + 1: char for idx, char in enumerate(self.characters)
        }

    @property
    def num_classes(self) -> int:
        """Total number of classes including the CTC blank token."""
        return len(self.characters) + 1

    def encode(self, text: str) -> List[int]:
        """Convert a character string to a list of integer class labels.

        Characters not in the vocabulary are skipped.

        Args:
            text (str): Input text string.

        Returns:
            List[int]: List of integer labels.
        """
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode_greedy(
        self, logits: torch.Tensor, time_major: bool = True
    ) -> Tuple[List[str], List[float]]:
        """Greedy CTC decode from model output logits.

        Collapses consecutive repeating characters and removes blank tokens.

        Args:
            logits (torch.Tensor): Tensor of shape (Time, Batch, Num_Classes) when time_major=True,
                                   or (Batch, Time, Num_Classes) when time_major=False.
            time_major (bool): Whether input dimension 0 is Time (default True from PyTorch CRNN).

        Returns:
            Tuple[List[str], List[float]]: Decoded strings and average confidence scores for each item.
        """
        # Ensure shape is (Batch, Time, Num_Classes)
        if time_major and logits.dim() == 3:
            logits = logits.permute(1, 0, 2)

        probs = torch.softmax(logits, dim=-1)
        max_probs, preds = torch.max(probs, dim=-1)

        preds_np = preds.detach().cpu().numpy()
        probs_np = max_probs.detach().cpu().numpy()

        batch_size, time_steps = preds_np.shape
        decoded_texts: List[str] = []
        confidences: List[float] = []

        for b in range(batch_size):
            char_list: List[str] = []
            score_list: List[float] = []
            prev_token = self.blank_idx

            for t in range(time_steps):
                token = preds_np[b, t]
                if token != self.blank_idx:
                    if token != prev_token:
                        char = self.idx_to_char.get(token, "")
                        if char:
                            char_list.append(char)
                            score_list.append(float(probs_np[b, t]))
                prev_token = token

            text = "".join(char_list)
            avg_score = float(np.mean(score_list)) if score_list else 0.0
            decoded_texts.append(text)
            confidences.append(avg_score)

        return decoded_texts, confidences
