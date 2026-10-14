"""Handwriting Recognition Package using PyTorch CRNN + CTC on IAM-line."""

from src.recognition.dataset import IAMLineDataset, iam_collate_fn
from src.recognition.decoder import GreedyCTCDecoder
from src.recognition.model import CRNNRecognitionModel
from src.recognition.predictor import HandwritingPredictor, get_predictor, recognize_line
from src.recognition.preprocessing import LineImagePreprocessor
from src.recognition.vocabulary import Vocabulary

__all__ = [
    "IAMLineDataset",
    "iam_collate_fn",
    "GreedyCTCDecoder",
    "CRNNRecognitionModel",
    "LineImagePreprocessor",
    "Vocabulary",
    "HandwritingPredictor",
    "get_predictor",
    "recognize_line",
]
