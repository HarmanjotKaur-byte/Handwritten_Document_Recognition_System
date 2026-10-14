"""Character vocabulary builder and encoder for handwriting recognition.

Builds a character-level mapping from training transcriptions only,
reserving index 0 for the dedicated CTC blank token.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Union


class Vocabulary:
    """Character-level vocabulary mapping characters to integer class labels."""

    BLANK_TOKEN = "<BLANK>"
    UNK_TOKEN = "<UNK>"
    BLANK_INDEX = 0

    def __init__(
        self,
        characters: Optional[Iterable[str]] = None,
        use_unk: bool = True,
    ):
        """Initialize Vocabulary.

        Args:
            characters: Iterable of distinct individual characters to include.
            use_unk: Whether to include an unknown character token.
        """
        self.use_unk = use_unk

        # Index 0 is strictly reserved for CTC Blank token
        self.char_to_idx: Dict[str, int] = {self.BLANK_TOKEN: self.BLANK_INDEX}
        self.idx_to_char: Dict[int, str] = {self.BLANK_INDEX: self.BLANK_TOKEN}

        current_idx = 1
        if self.use_unk:
            self.char_to_idx[self.UNK_TOKEN] = current_idx
            self.idx_to_char[current_idx] = self.UNK_TOKEN
            self.unk_index = current_idx
            current_idx += 1
        else:
            self.unk_index = None

        if characters is not None:
            # Sort characters for deterministic index assignment
            sorted_chars = sorted(set(characters))
            for char in sorted_chars:
                if char not in self.char_to_idx:
                    self.char_to_idx[char] = current_idx
                    self.idx_to_char[current_idx] = char
                    current_idx += 1

    def __len__(self) -> int:
        """Total number of classes including BLANK and UNK."""
        return len(self.char_to_idx)

    @classmethod
    def build_from_texts(
        cls, texts: Iterable[str], use_unk: bool = True
    ) -> "Vocabulary":
        """Extract all unique characters from training texts.

        Args:
            texts: Iterable of training text strings.
            use_unk: Whether to allocate an unknown character index.

        Returns:
            Vocabulary: Built vocabulary.
        """
        unique_chars: Set[str] = set()
        for text in texts:
            if text:
                unique_chars.update(list(str(text)))

        return cls(characters=unique_chars, use_unk=use_unk)

    def encode(self, text: str) -> List[int]:
        """Convert a character string into a list of integer indices.

        Args:
            text: Ground-truth transcription.

        Returns:
            List[int]: Encoded integer sequence.
        """
        encoded: List[int] = []
        for char in str(text):
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
            elif self.use_unk and self.unk_index is not None:
                encoded.append(self.unk_index)
            # If use_unk is False, unseen characters are skipped
        return encoded

    def decode_indices(self, indices: Iterable[int]) -> str:
        """Convert an integer index sequence to a string without CTC collapsing.

        Args:
            indices: List of token indices.

        Returns:
            str: Decoded string.
        """
        chars: List[str] = []
        for idx in indices:
            if idx == self.BLANK_INDEX:
                continue
            char = self.idx_to_char.get(idx, "")
            if char and char != self.UNK_TOKEN:
                chars.append(char)
        return "".join(chars)

    def to_dict(self) -> dict:
        """Serialize vocabulary mappings to a dictionary."""
        return {
            "blank_token": self.BLANK_TOKEN,
            "blank_index": self.BLANK_INDEX,
            "use_unk": self.use_unk,
            "unk_index": self.unk_index,
            "char_to_idx": self.char_to_idx,
            "num_classes": len(self),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        """Reconstruct Vocabulary from a dictionary."""
        vocab = cls(use_unk=data.get("use_unk", True))
        vocab.char_to_idx = {str(k): int(v) for k, v in data["char_to_idx"].items()}
        vocab.idx_to_char = {int(v): str(k) for k, v in data["char_to_idx"].items()}
        vocab.unk_index = data.get("unk_index")
        return vocab

    def save(self, file_path: Union[str, Path]) -> None:
        """Save vocabulary to a JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "Vocabulary":
        """Load vocabulary from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
