"""Unit and integration tests for the handwriting recognition module."""

import numpy as np
import pytest
import torch
from PIL import Image

from src.recognition.dataset import iam_collate_fn
from src.recognition.decoder import GreedyCTCDecoder
from src.recognition.model import CRNNRecognitionModel
from src.recognition.preprocessing import LineImagePreprocessor
from src.recognition.vocabulary import Vocabulary


# -------------------------------------------------------------
# Vocabulary Tests
# -------------------------------------------------------------

def test_vocabulary_blank_index_and_encoding():
    texts = ["hello world", "ABC 123!"]
    vocab = Vocabulary.build_from_texts(texts, use_unk=True)

    # Blank token must be strictly at index 0
    assert vocab.char_to_idx[Vocabulary.BLANK_TOKEN] == Vocabulary.BLANK_INDEX
    assert vocab.idx_to_char[Vocabulary.BLANK_INDEX] == Vocabulary.BLANK_TOKEN

    # Unknown token must be at index 1
    assert vocab.char_to_idx[Vocabulary.UNK_TOKEN] == 1

    # Encode known text
    encoded = vocab.encode("hello")
    assert len(encoded) == 5
    assert all(isinstance(x, int) for x in encoded)
    assert all(x > 0 for x in encoded)

    # Encode unknown character
    encoded_unk = vocab.encode("~")  # not in training texts
    assert encoded_unk == [vocab.unk_index]

    # Serialization roundtrip
    data_dict = vocab.to_dict()
    reconstructed = Vocabulary.from_dict(data_dict)
    assert len(reconstructed) == len(vocab)
    assert reconstructed.encode("hello") == encoded


# -------------------------------------------------------------
# Preprocessor Tests
# -------------------------------------------------------------

def test_line_preprocessor_resizing_and_normalization():
    prep = LineImagePreprocessor(target_height=32, min_width=32, invert_colors=True)

    # Create dummy white image with a black stroke (100 H x 400 W)
    dummy_img = Image.new("L", (400, 100), color=255)
    tensor = prep.preprocess(dummy_img, augment=False)

    # Height must be exactly 32
    assert tensor.shape[1] == 32
    # Width must be rescaled: 400 * (32 / 100) = 128, padded to multiple of 4 -> 128
    assert tensor.shape[2] == 128
    assert tensor.shape[0] == 1  # 1 channel
    assert tensor.dtype == torch.float32

    # After inversion, white paper (255) becomes ~0.0
    assert tensor.min().item() >= 0.0
    assert tensor.max().item() <= 1.0


# -------------------------------------------------------------
# CRNN Model Architecture Tests
# -------------------------------------------------------------

def test_crnn_forward_pass_dimensions():
    vocab_size = 80
    model = CRNNRecognitionModel(num_classes=vocab_size, in_channels=1, lstm_hidden_size=128)
    model.eval()

    # Batch of 2, 1 channel, height 32, width 128
    dummy_x = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        logits = model(dummy_x)

    # Expected shape: (Batch, Time, num_classes) where Batch = 2, Time = (128 // 4) - 1 = 31
    assert logits.shape == (2, 31, vocab_size)

    # Test log_probs
    log_probs = model.compute_log_probs(dummy_x)
    assert log_probs.shape == (2, 31, vocab_size)
    assert torch.all(log_probs <= 0.0)  # Log probabilities are non-positive


# -------------------------------------------------------------
# Greedy CTC Decoder Tests
# -------------------------------------------------------------

def test_greedy_ctc_decoder_collapsing_and_blanks():
    vocab = Vocabulary(["a", "b", "c", "l", "o"], use_unk=False)
    decoder = GreedyCTCDecoder(vocabulary=vocab, blank_index=Vocabulary.BLANK_INDEX)

    blank = Vocabulary.BLANK_INDEX
    l_idx = vocab.char_to_idx["l"]
    o_idx = vocab.char_to_idx["o"]

    # Test 1: Repeated consecutive characters WITHOUT blank: [l, l, l] -> "l"
    seq1 = [l_idx, l_idx, l_idx]
    logits1 = torch.zeros(len(seq1), 1, len(vocab))
    for t, idx in enumerate(seq1):
        logits1[t, 0, idx] = 10.0

    res1 = decoder.decode(logits1)
    assert res1[0][0] == "l"

    # Test 2: Consecutive identical characters WITH blank in between: [l, blank, l] -> "ll" (like in "hello")
    seq2 = [l_idx, blank, l_idx]
    logits2 = torch.zeros(len(seq2), 1, len(vocab))
    for t, idx in enumerate(seq2):
        logits2[t, 0, idx] = 10.0

    res2 = decoder.decode(logits2)
    assert res2[0][0] == "ll"

    # Test 3: Word "look": [l, o, blank, o, k] -> "loo" (k is missing from small vocab)
    seq3 = [l_idx, o_idx, blank, o_idx]
    logits3 = torch.zeros(len(seq3), 1, len(vocab))
    for t, idx in enumerate(seq3):
        logits3[t, 0, idx] = 10.0

    res3 = decoder.decode(logits3)
    assert res3[0][0] == "loo"

    # Test 4: Empty / All blanks -> ""
    seq4 = [blank, blank, blank]
    logits4 = torch.zeros(len(seq4), 1, len(vocab))
    for t, idx in enumerate(seq4):
        logits4[t, 0, idx] = 10.0

    res4 = decoder.decode(logits4)
    assert res4[0][0] == ""


# -------------------------------------------------------------
# Collate Function Tests
# -------------------------------------------------------------

def test_iam_collate_fn_padding_and_lengths():
    sample1 = {
        "image": torch.ones((1, 32, 100)),
        "target": torch.tensor([1, 2, 3]),
        "raw_text": "abc",
        "input_length": 25,
        "target_length": 3,
    }
    sample2 = {
        "image": torch.ones((1, 32, 160)),
        "target": torch.tensor([4, 5]),
        "raw_text": "de",
        "input_length": 40,
        "target_length": 2,
    }

    padded_imgs, targets, input_lens, target_lens, raw_texts = iam_collate_fn([sample1, sample2])

    assert padded_imgs.shape == (2, 1, 32, 160)
    assert targets.tolist() == [1, 2, 3, 4, 5]
    # Downsampled sequence length through Stage 7 Conv2d unpadded is (W // 4) - 1:
    # 100 // 4 - 1 = 24, 160 // 4 - 1 = 39
    assert input_lens.tolist() == [24, 39]
    assert target_lens.tolist() == [3, 2]
    assert raw_texts == ["abc", "de"]
