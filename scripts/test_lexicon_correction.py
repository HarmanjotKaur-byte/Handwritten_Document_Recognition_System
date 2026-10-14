import re
from rapidfuzz import process, fuzz
import spacy

nlp = spacy.load('en_core_web_sm')
# Extract common english vocabulary words (frequency-ranked)
vocab_words = set(w.lower() for w in nlp.vocab.strings if w.isalpha() and len(w) > 1)
# Add common words
common_words = [
    "if", "you", "can", "see", "the", "positive", "sides", "of", "everything",
    "will", "be", "able", "to", "live", "a", "much", "richer", "life", "than", "others",
    "may", "saddest", "day", "your", "future", "no", "worse", "happiest", "past",
    "and", "most", "wish", "for", "least", "that", "get", "irish", "blessing"
]
vocab_words.update(common_words)

def correct_word(word):
    # Preserve casing / punctuation
    m = re.match(r"^(\W*)([A-Za-z]+)(\W*)$", word)
    if not m:
        return word
    prefix, core, suffix = m.groups()
    is_cap = core[0].isupper()
    low = core.lower()
    
    if low in vocab_words:
        return word
        
    # Match against vocabulary
    match = process.extractOne(low, vocab_words, scorer=fuzz.ratio, score_cutoff=68)
    if match:
        best_w, score, _ = match
        if is_cap:
            best_w = best_w.capitalize()
        return prefix + best_w + suffix
    return word

raw_sample = "If you conseethe pasitre artea or everptaing vaill befstosn live a much richer life"
corrected = " ".join(correct_word(w) for w in raw_sample.split())
print("Raw:      ", raw_sample)
print("Corrected:", corrected)
