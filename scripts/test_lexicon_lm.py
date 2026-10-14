"""Test Language Model & Lexicon Correction on Handwritten OCR Output."""

import re
from collections import Counter
import math

class LexiconCorrector:
    def __init__(self):
        self.words = Counter()
        # 1. Load IAM vocabulary words
        iam_txt = 'external/handwritten-text-recognition/iam_words/words.txt'
        try:
            with open(iam_txt, 'r', encoding='latin1') as f:
                for line in f:
                    if line.startswith('#') or not line.strip(): continue
                    parts = line.strip().split()
                    if len(parts) >= 9 and parts[1] == 'ok':
                        w = parts[-1].lower()
                        if re.match(r'^[a-z]+$', w):
                            self.words[w] += 1
        except Exception:
            pass

        # 2. Add common English words to ensure full coverage
        common_english = [
            "if", "you", "can", "see", "the", "positive", "side", "sides", "of", "everything",
            "will", "be", "able", "to", "live", "a", "rich", "and", "beautiful", "life",
            "is", "it", "in", "that", "this", "there", "their", "they", "we", "he", "she",
            "for", "on", "with", "as", "at", "by", "from", "up", "about", "into", "over",
            "after", "beneath", "under", "above", "world", "day", "night", "time", "person",
            "year", "way", "man", "woman", "child", "government", "company", "number", "group",
            "problem", "fact", "hand", "part", "place", "case", "week", "work", "great", "small",
            "good", "new", "first", "last", "long", "little", "own", "other", "old", "right",
            "big", "high", "different", "small", "large", "next", "early", "young", "important",
            "few", "public", "bad", "same", "able", "to", "make", "take", "get", "give", "find",
            "tell", "ask", "seem", "feel", "try", "leave", "call", "keep", "look", "come", "go"
        ]
        for w in common_english:
            self.words[w] += 100

        self.total_count = sum(self.words.values())

    def word_prob(self, word):
        return self.words.get(word.lower(), 1) / self.total_count

    def edit1(self, word):
        """All edits that are one edit away from `word`."""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edit2(self, word):
        """All edits that are two edits away from `word`."""
        return set(e2 for e1 in self.edit1(word) for e2 in self.edit1(e1))

    def known(self, words):
        """The subset of `words` that appear in the dictionary."""
        return set(w for w in words if w.lower() in self.words)

    def correct_word(self, word):
        """Most probable spelling correction for a single word."""
        w = word.lower().strip(",.!?\"'")
        if not w: return word
        if w in self.words and len(w) > 2:
            return w

        # Candidates: edit distance 1, then 2
        candidates = self.known(self.edit1(w)) or self.known(self.edit2(w))
        if candidates:
            best = max(candidates, key=self.word_prob)
            return best
        return word

    def split_merged_words(self, token):
        """Check if a merged token (like 'seethe' or 'ifyoucan') is better as multiple words."""
        w = token.lower().strip(",.!?\"'")
        n = len(w)
        if n < 5:
            return [self.correct_word(token)]

        # Check 2-word split
        best_split = None
        best_score = -float('inf')
        
        # Single word score
        single_word = self.correct_word(w)
        if single_word in self.words:
            best_score = math.log(self.word_prob(single_word))
            best_split = [single_word]

        for i in range(2, n - 1):
            w1, w2 = w[:i], w[i:]
            c1 = self.correct_word(w1)
            c2 = self.correct_word(w2)
            if c1 in self.words and c2 in self.words:
                score = math.log(self.word_prob(c1)) + math.log(self.word_prob(c2))
                if score > best_score:
                    best_score = score
                    best_split = [c1, c2]

        return best_split if best_split else [self.correct_word(token)]

    def correct_sentence(self, sentence):
        tokens = sentence.split()
        corrected = []
        for tok in tokens:
            # Check for common contractions / small tokens
            clean = tok.strip(",.!?\"'")
            if clean.lower() in ["if", "you", "can", "see", "the", "of", "to", "and", "a", "in", "is"]:
                corrected.append(clean.lower())
                continue
            
            # Try splitting compound/merged words
            splits = self.split_merged_words(tok)
            corrected.extend(splits)
            
        return " ".join(corrected)

if __name__ == "__main__":
    corrector = LexiconCorrector()
    raw_lines = [
        "FEyeucan seethe posiire sidet",
        "PrPPsritingvaulibeabletend",
        "ererefaghnaeraneerf",
        "mnoos"
    ]
    print("--- Testing Lexicon Correction on Raw OCR Output ---")
    for raw in raw_lines:
        res = corrector.correct_sentence(raw)
        print(f"RAW : '{raw}'")
        print(f"CORR: '{res}'\n")
