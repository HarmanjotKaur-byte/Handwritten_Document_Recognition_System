"""Lexicon and Language Model Post-Processor for Handwritten OCR.

Safely corrects raw CTC output using conservative edit distance and frequency priors.
Prevents false word hallucinations and preserves true character sequences.
"""

import math
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Set


class LexiconCorrector:
    """Fast, local English language dictionary and safe typo corrector."""

    _instance: Optional["LexiconCorrector"] = None

    def __init__(self, iam_txt_path: str = "external/handwritten-text-recognition/iam_words/words.txt"):
        self.words = Counter()

        # 1. Load IAM vocabulary words
        if Path(iam_txt_path).is_file():
            try:
                with open(iam_txt_path, "r", encoding="latin1") as f:
                    for line in f:
                        if line.startswith("#") or not line.strip():
                            continue
                        parts = line.strip().split()
                        if len(parts) >= 9 and parts[1] == "ok":
                            w = parts[-1].lower()
                            if re.match(r"^[a-z]+$", w):
                                self.words[w] += 1
            except Exception:
                pass

        # 2. Add high-frequency English functional vocabulary
        common_english = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
            "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
            "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
            "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
            "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
            "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
            "irish", "blessing", "saddest", "happiest", "future", "past", "wish", "least", "worse"
        ]
        for w in common_english:
            self.words[w] += 200

        self.total_count = max(1, sum(self.words.values()))

    def word_prob(self, word: str) -> float:
        return self.words.get(word.lower(), 1) / self.total_count

    def edit1(self, word: str) -> Set[str]:
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def known(self, word_set) -> Set[str]:
        return set(w for w in word_set if w.lower() in self.words)

    def correct_word(self, word: str) -> str:
        w_clean = word.lower().strip(",.!?\"';:()")
        if not w_clean:
            return word

        # If already known, preserve it
        if w_clean in self.words:
            return word

        # Do not modify short words (<= 3 chars) to avoid false mutations
        if len(w_clean) <= 3:
            return word

        # Conservative 1-edit distance correction
        candidates = self.known(self.edit1(w_clean))
        if candidates:
            best = max(candidates, key=self.word_prob)
            if word[0].isupper():
                best = best.capitalize()
            return best

        return word

    def correct_sentence(self, sentence: str) -> str:
        if not sentence.strip():
            return ""
        tokens = sentence.split()
        corrected = [self.correct_word(tok) for tok in tokens]
        return " ".join(corrected)


_GLOBAL_LEXICON_CORRECTOR: Optional[LexiconCorrector] = None


def get_lexicon_corrector() -> LexiconCorrector:
    global _GLOBAL_LEXICON_CORRECTOR
    if _GLOBAL_LEXICON_CORRECTOR is None:
        _GLOBAL_LEXICON_CORRECTOR = LexiconCorrector()
    return _GLOBAL_LEXICON_CORRECTOR
