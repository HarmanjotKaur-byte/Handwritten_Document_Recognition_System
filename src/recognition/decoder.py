"""Greedy CTC Decoder for converting recurrent sequence logits into recognized text strings."""

from typing import List, Tuple

import numpy as np
import torch

from src.recognition.vocabulary import Vocabulary


class GreedyCTCDecoder:
    """Decodes raw network logits using the Connectionist Temporal Classification (CTC) greedy rule.

    Decoding rules:
    1. At each time step t, choose the index with maximum logit / probability.
    2. Collapse identical consecutive indices: [a, a, b] -> [a, b].
    3. Remove the CTC blank token.
    4. Convert surviving indices to character tokens using the Vocabulary.
    """

    def __init__(self, vocabulary: Vocabulary, blank_index: int = Vocabulary.BLANK_INDEX):
        """Initialize GreedyCTCDecoder.

        Args:
            vocabulary: Character Vocabulary mapping indices to characters.
            blank_index: CTC blank index (must match training convention, default 0).
        """
        self.vocabulary = vocabulary
        self.blank_index = blank_index

    def decode(
        self,
        logits: torch.Tensor,
        input_lengths: torch.Tensor = None,
    ) -> List[Tuple[str, float]]:
        """Greedy decode a batch of sequence logits into text strings with confidence scores.

        Args:
            logits: Tensor of shape (Time, Batch, num_classes) or (Batch, Time, num_classes).
            input_lengths: Optional 1D Tensor of active time steps for each batch sample.

        Returns:
            List[Tuple[str, float]]: List of (predicted_text, average_confidence) pairs for each sample.
        """
        # Ensure logits shape is (Batch, Time, num_classes)
        if logits.dim() == 3:
            # Check if logits are in (Time, Batch, num_classes) where Batch is dimension 1
            # If dim 2 is num_classes:
            # - If dim 0 > dim 1 and dim 1 <= 32 (typical batch size), it might be (Time, Batch, C)
            # But if already (Batch, Time, num_classes) where Batch=1, Time=373, do NOT permute!
            if logits.shape[2] == len(self.vocabulary) and logits.shape[1] < logits.shape[0]:
                logits = logits.permute(1, 0, 2)

        probs = torch.softmax(logits, dim=-1)
        max_probs, preds = torch.max(probs, dim=-1)

        preds_np = preds.detach().cpu().numpy()
        probs_np = max_probs.detach().cpu().numpy()

        batch_size, max_time = preds_np.shape
        results: List[Tuple[str, float]] = []

        for b in range(batch_size):
            valid_time = int(input_lengths[b]) if input_lengths is not None else max_time
            valid_time = min(valid_time, max_time)

            char_list: List[str] = []
            score_list: List[float] = []
            prev_token = self.blank_index

            for t in range(valid_time):
                token = int(preds_np[b, t])
                prob = float(probs_np[b, t])

                # CTC Rule:
                # If current token is not blank, and not identical to immediate predecessor, emit character
                if token != self.blank_index:
                    if token != prev_token:
                        char = self.vocabulary.idx_to_char.get(token, "")
                        if char and char != self.vocabulary.UNK_TOKEN:
                            char_list.append(char)
                            score_list.append(prob)
                prev_token = token

            text = "".join(char_list)
            avg_score = float(np.mean(score_list)) if score_list else 0.0
            results.append((text, avg_score))

        return results

    def decode_single(self, logits: torch.Tensor) -> Tuple[str, float]:
        """Decode a single sample logits tensor.

        Args:
            logits: Tensor of shape (Time, 1, num_classes) or (Time, num_classes).

        Returns:
            Tuple[str, float]: (predicted_text, confidence).
        """
        if logits.dim() == 2:
            logits = logits.unsqueeze(1)  # (Time, 1, num_classes)
        results = self.decode(logits)
        return results[0] if results else ("", 0.0)
