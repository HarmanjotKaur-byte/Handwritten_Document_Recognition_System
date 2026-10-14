"""Convolutional Recurrent Neural Network (CRNN) for handwritten text recognition.

Architecture:
    CNN: Extracts sequence of visual feature representations from line image crops.
    Map-to-Sequence: Reshapes feature maps into a 1D sequence over time.
    BiLSTM: Models bidirectional temporal context across handwriting strokes.
    Linear Projection: Maps hidden states to character class log-probabilities for CTC.
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """CRNN architecture for handwritten line image recognition."""

    def __init__(self, num_classes: int, in_channels: int = 1, hidden_size: int = 256):
        """Initialize the CRNN network.

        Args:
            num_classes (int): Vocabulary size including the CTC blank token.
            in_channels (int): Input image channels (1 for grayscale).
            hidden_size (int): Hidden dimension for bidirectional LSTM layers.
        """
        super().__init__()
        self.num_classes = num_classes

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (H: 32 -> 16)
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (H: 16 -> 8)
            # Layer 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Layer 4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),  # (H: 8 -> 4)
            # Layer 5
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Layer 6
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),  # (H: 4 -> 2)
            # Layer 7: Final collapse of vertical height to 1
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),  # (H: 2 -> 1)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Bidirectional Recurrent Context
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
            dropout=0.2,
        )

        # Linear Prediction Head: (hidden_size * 2) -> num_classes
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Batch of line crops of shape (Batch, 1, 32, Width).

        Returns:
            torch.Tensor: Logits of shape (Time, Batch, num_classes) for CTCLoss.
        """
        # Feature extraction: (B, 1, 32, W) -> (B, 512, 1, Time)
        features = self.cnn(x)

        # Collapse vertical dimension: (B, 512, 1, Time) -> (B, 512, Time)
        features = features.squeeze(2)

        # Permute to (Time, Batch, Features) for RNN
        features = features.permute(2, 0, 1)

        # Recurrent sequence modeling: (Time, Batch, 512)
        recurrent_out, _ = self.rnn(features)

        # Linear projection: (Time, Batch, num_classes)
        logits = self.fc(recurrent_out)

        return logits
