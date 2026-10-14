"""Pure Python Word Beam Search CTC Decoder for Handwriting Recognition.

Uses a Prefix Trie and Lexicon Word Prior to ensure output words are valid English words.
Extremely fast, requires no C++ extensions, and runs completely locally.
"""

import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple


class PrefixTrie:
    """Trie structure for fast word and prefix validation."""

    def __init__(self):
        self.root = {}
        self.word_end = "__END__"

    def insert(self, word: str):
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node[self.word_end] = True

    def is_prefix(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node:
                return False
            node = node[char]
        return True

    def is_word(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return self.word_end in node


class WordBeamSearchDecoder:
    """Word Beam Search Decoder constraining CTC paths to valid English words."""

    def __init__(
        self,
        vocabulary: Dict[int, str],
        word_list: List[str],
        beam_width: int = 25,
        blank_index: int = 0,
        lm_weight: float = 1.2,
        word_penalty: float = 0.5,
    ):
        self.vocab = vocabulary
        self.blank_index = blank_index
        self.beam_width = beam_width
        self.lm_weight = lm_weight
        self.word_penalty = word_penalty

        self.trie = PrefixTrie()
        self.word_freq = defaultdict(lambda: 1)
        for w in word_list:
            w_clean = w.strip().lower()
            if w_clean:
                self.trie.insert(w_clean)
                self.word_freq[w_clean] += 1

        self.total_words = sum(self.word_freq.values())

    def decode(self, log_probs: list) -> str:
        """Decode CTC log probabilities matrix of shape (Time, num_classes).

        Args:
            log_probs: 2D list or numpy array of log probabilities.

        Returns:
            str: Decoded sentence text.
        """
        # Beam entry: text -> (prob_blank, prob_non_blank, current_word_prefix)
        # We start with empty string at t = 0
        beams = {"": (0.0, -float("inf"), "")}

        time_steps = len(log_probs)

        for t in range(time_steps):
            frame_probs = log_probs[t]
            new_beams = defaultdict(lambda: (-float("inf"), -float("inf"), ""))

            # Top K character candidates for this frame to keep speed fast
            top_chars = sorted(
                range(len(frame_probs)), key=lambda c: frame_probs[c], reverse=True
            )[:12]

            for text, (p_b, p_nb, current_word) in beams.items():
                p_total = math.log(math.exp(p_b) + math.exp(p_nb)) if (p_b > -100 and p_nb > -100) else max(p_b, p_nb)

                # 1. Blank token emitted
                b_prob = frame_probs[self.blank_index]
                old_nb_b, old_b_b, cur_w = new_beams[text]
                # Emission of blank does not change text
                new_p_b = math.log(math.exp(old_nb_b) + math.exp(p_total + b_prob)) if (old_nb_b > -100) else (p_total + b_prob)
                new_beams[text] = (new_p_b, old_b_b, current_word)

                # 2. Non-blank characters emitted
                for char_idx in top_chars:
                    if char_idx == self.blank_index:
                        continue

                    char = self.vocab.get(char_idx, "")
                    if not char:
                        continue

                    c_prob = frame_probs[char_idx]

                    if char == " ":
                        # Word boundary: check if current_word is a valid word
                        lm_score = 0.0
                        if current_word:
                            if self.trie.is_word(current_word.lower()):
                                freq = self.word_freq[current_word.lower()]
                                lm_score = math.log(freq / self.total_words) * self.lm_weight + self.word_penalty
                            else:
                                lm_score = -5.0  # Penalty for non-word

                        new_text = text + " "
                        n_b, n_nb, _ = new_beams[new_text]
                        new_p_nb = (p_total + c_prob + lm_score)
                        comb_nb = math.log(math.exp(n_nb) + math.exp(new_p_nb)) if n_nb > -100 else new_p_nb
                        new_beams[new_text] = (n_b, comb_nb, "")
                    else:
                        # Character extension: check if prefix is valid in Trie
                        new_word = current_word + char
                        # If prefix is valid, or word is short (< 3)
                        is_valid_prefix = len(new_word) <= 2 or self.trie.is_prefix(new_word.lower())
                        prefix_penalty = 0.0 if is_valid_prefix else -4.0

                        if text and text[-1] == char:
                            # Repeating character: only extends if coming from blank
                            # from p_b: text + char
                            new_text = text + char
                            n_b, n_nb, _ = new_beams[new_text]
                            cand = p_b + c_prob + prefix_penalty
                            comb = math.log(math.exp(n_nb) + math.exp(cand)) if n_nb > -100 else cand
                            new_beams[new_text] = (n_b, comb, new_word)

                            # from p_nb: stays as text
                            n_b2, n_nb2, _ = new_beams[text]
                            cand2 = p_nb + c_prob
                            comb2 = math.log(math.exp(n_nb2) + math.exp(cand2)) if n_nb2 > -100 else cand2
                            new_beams[text] = (n_b2, comb2, current_word)
                        else:
                            new_text = text + char
                            n_b, n_nb, _ = new_beams[new_text]
                            cand = p_total + c_prob + prefix_penalty
                            comb = math.log(math.exp(n_nb) + math.exp(cand)) if n_nb > -100 else cand
                            new_beams[new_text] = (n_b, comb, new_word)

            # Prune to beam_width
            scored_beams = []
            for t_text, (pb, pnb, cur_w) in new_beams.items():
                tot = math.log(math.exp(pb) + math.exp(pnb)) if (pb > -100 and pnb > -100) else max(pb, pnb)
                scored_beams.append((tot, t_text, pb, pnb, cur_w))

            scored_beams.sort(key=lambda x: x[0], reverse=True)
            beams = {
                item[1]: (item[2], item[3], item[4])
                for item in scored_beams[: self.beam_width]
            }

        # Final best beam
        if not beams:
            return ""

        best_text = max(
            beams.keys(),
            key=lambda k: max(beams[k][0], beams[k][1]),
        )
        return best_text.strip()
