"""Tests for CRNN model architecture and CTC decoder."""

import torch
import pytest

from src.ocr.decoder import CTCDecoder
from src.ocr.model import CRNN
from src.ocr.recognizer import HandwritingRecognizer


def test_crnn_forward_pass_shape():
    decoder = CTCDecoder()
    num_classes = decoder.num_classes
    model = CRNN(num_classes=num_classes)
    model.eval()

    # Batch of 2, 1 channel, height 32, width 128
    dummy_input = torch.randn(2, 1, 32, 128)
    with torch.no_grad():
        output = model(dummy_input)

    # Output shape should be (Time, Batch, num_classes)
    assert output.dim() == 3
    assert output.shape[1] == 2
    assert output.shape[2] == num_classes


def test_ctc_decoder_encode_decode():
    decoder = CTCDecoder()
    sample_text = "Hello 123"
    encoded = decoder.encode(sample_text)
    assert len(encoded) == len(sample_text)

    # Simulate model logits of shape (Time, Batch, Num_Classes)
    time_steps = len(encoded) * 2
    logits = torch.zeros(time_steps, 1, decoder.num_classes)
    # Set alternating non-blank and blank to test collapsing
    for i, token in enumerate(encoded):
        t = i * 2
        logits[t, 0, token] = 10.0  # high confidence on token
        logits[t + 1, 0, decoder.blank_idx] = 10.0  # blank

    decoded_texts, confidences = decoder.decode_greedy(logits, time_major=True)
    assert len(decoded_texts) == 1
    assert decoded_texts[0] == sample_text
    assert confidences[0] > 0.9


def test_recognizer_diagnostics_reports_uninitialized():
    recognizer = HandwritingRecognizer(model_path=None)
    diag = recognizer.get_status_diagnostics()
    assert diag["is_trained"] is False
