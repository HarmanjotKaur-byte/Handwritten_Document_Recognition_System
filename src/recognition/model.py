"""CRNN (CNN + BiLSTM + CTC Projection) Model Architecture for Handwriting Recognition.

Architecture Overview:
    Input: (B, 1, 32, W) Grayscale line images.
    CNN: Extracts spatial/stroke features and reduces height to 1 while downsampling width by 4x.
         Feature Map: (B, 512, 1, W // 4)
    Map-to-Sequence: (B, 512, 1, W // 4) -> (W // 4, B, 512)
    BiLSTM: 2-layer Bidirectional LSTM capturing left-to-right and right-to-left stroke context.
            Output: (W // 4, B, 512)
    Linear Projection: Maps contextual features to character class logits.
            Output Logits: (Time, Batch, num_classes) where Time = W // 4.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn


class CRNNRecognitionModel(nn.Module):
    """Convolutional Recurrent Neural Network for handwritten text line recognition.

    Architecture Overview:
        Input: (B, 1, 128, W) Grayscale line images.
        CNN: 7-stage feature extractor with Conv2D, BatchNorm2D, ReLU, and MaxPool2D.
             Reduces height to 7 and downsamples width by 4x.
        Map-to-Sequence: Linear projection from (512 * 7) down to hidden_size (256).
        BiLSTM: 2-layer Bidirectional LSTM capturing left-to-right and right-to-left stroke context.
        Linear Classification: Maps hidden state to character class logits (including CTC blank).
    """

    def __init__(
        self,
        num_classes: int = 75,
        in_channels: int = 1,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
        img_height: int = 128,
    ):
        """Initialize CRNN model.

        Args:
            num_classes: Total vocabulary size including the CTC blank token.
            in_channels: Input image channels (1 for grayscale).
            lstm_hidden_size: Hidden dimension of each directional LSTM.
            lstm_num_layers: Number of stacked BiLSTM layers.
            lstm_dropout: Dropout probability between recurrent layers.
            img_height: Fixed input height (128 for IAM-line).
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.lstm_hidden_size = lstm_hidden_size
        self.img_height = img_height
        self.downsample_factor = 4

        # ---------------------------------------------------------------------
        # CNN Feature Extractor
        # ---------------------------------------------------------------------
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (B, 64, 64, W/2)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (B, 128, 32, W/4)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # (B, 256, 16, W/4)

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # (B, 512, 8, W/4)

            nn.Conv2d(512, 512, kernel_size=2),  # (B, 512, 7, W/4 - 1)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.map2seq = nn.Linear(512 * 7, lstm_hidden_size)
        self.rnn = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(lstm_hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input batch of shape (Batch, 1, Height, Width).
               If Height != 128, automatically resamples height to 128.

        Returns:
            torch.Tensor: Log probabilities of shape (Batch, Time, num_classes)
        """
        if x.shape[2] != self.img_height:
            x = torch.nn.functional.interpolate(
                x, size=(self.img_height, x.shape[3]), mode="bilinear", align_corners=False
            )

        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.permute(0, 3, 1, 2).reshape(b, w, c * h)
        seq = self.map2seq(conv)
        rnn_out, _ = self.rnn(seq)
        output = self.fc(rnn_out)
        return torch.nn.functional.log_softmax(output, dim=2)

    def compute_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log-softmax probabilities for CTC decoding / loss.

        Args:
            x: Input batch of shape (Batch, 1, 128, Width).

        Returns:
            torch.Tensor: Log probabilities of shape (Batch, Time, num_classes).
        """
        return self.forward(x)

    def get_config(self) -> dict:
        """Return architecture configuration for checkpoint serialization."""
        return {
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "lstm_hidden_size": self.lstm_hidden_size,
            "img_height": self.img_height,
            "downsample_factor": self.downsample_factor,
        }


class PyLaiaConvBlock(nn.Module):
    """Convolutional building block with BatchNorm, LeakyReLU, and optional MaxPool2D."""

    def __init__(self, in_channels: int, out_channels: int, pool: Tuple[int, int]):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.pool = nn.MaxPool2d(pool) if pool[0] > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.batchnorm(self.conv(x))))


class StandaloneLaiaCRNN(nn.Module):
    """Benchmark PyLaia CRNN model for IAM handwriting recognition.

    Architecture:
        - 4 ConvBlocks downsampling height from 128 to 16 and width by 8x.
        - 3-layer Bidirectional LSTM (hidden size 256, dropout 0.5).
        - Linear projection to 81 vocabulary classes.
    """

    def __init__(self, num_classes: int = 81):
        super().__init__()
        self.num_classes = num_classes
        self.downsample_factor = 8
        self.img_height = 128

        self.conv = nn.Sequential(
            PyLaiaConvBlock(1, 12, (2, 2)),
            PyLaiaConvBlock(12, 24, (2, 2)),
            PyLaiaConvBlock(24, 48, (0, 0)),
            PyLaiaConvBlock(48, 48, (2, 2)),
        )
        self.rnn = nn.LSTM(768, 256, num_layers=3, bidirectional=True, dropout=0.5)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (Batch, 1, Height, Width).

        Returns:
            torch.Tensor: Log probabilities of shape (Batch, Time, num_classes).
        """
        if x.shape[2] != self.img_height:
            x = torch.nn.functional.interpolate(
                x, size=(self.img_height, x.shape[3]), mode="bilinear", align_corners=False
            )
        c = self.conv(x)  # (Batch, 48, 16, Time)
        b, ch, h, w = c.shape
        c = c.permute(3, 0, 1, 2).reshape(w, b, ch * h)  # (Time, Batch, 768)
        r, _ = self.rnn(c)  # (Time, Batch, 512)
        logits = self.linear(r)  # (Time, Batch, num_classes)
        return torch.nn.functional.log_softmax(logits.permute(1, 0, 2), dim=2)

    def compute_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def get_config(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "downsample_factor": self.downsample_factor,
            "img_height": self.img_height,
            "arch": "CNN-BiLSTM-CTC",
        }


class CodeMasterCRNN(nn.Module):
    """CRNN Model from CodeMaster-AJ/handwritten-text-recognition repository.

    Architecture:
        - 4 Convolutional stages with ReLU and MaxPool2D
          reduces (Batch, 1, 32, 128) -> (Batch, 256, 4, 32)
        - Height flattening: (Batch, 32, 1024)
        - Linear projection: Linear(1024, 128)
        - 2-layer Bidirectional LSTM: LSTM(128, 128, num_layers=2, batch_first=True, dropout=0.2)
        - Final linear projection: Linear(256, num_classes)
        - Output: (Batch, Time=32, num_classes)
    """

    def __init__(self, num_classes: int = 77):
        super().__init__()
        self.num_classes = num_classes
        self.img_height = 32
        self.img_width = 128
        self.time_steps = 32

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )
        self.linear = nn.Linear(256 * 4, 128)
        self.lstm = nn.LSTM(
            128, 128, bidirectional=True, num_layers=2, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (Batch, 1, 32, 128).

        Returns:
            torch.Tensor: Log probabilities of shape (Batch, Time, num_classes).
        """
        features = self.cnn(x)
        features = features.permute(0, 3, 1, 2)
        b, t, c, h = features.size()
        features = features.contiguous().view(b, t, c * h)
        seq_out = self.linear(features)
        seq_out, _ = self.lstm(seq_out)
        logits = self.fc(seq_out)
        return torch.nn.functional.log_softmax(logits, dim=2)

    def compute_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def get_config(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "img_height": self.img_height,
            "img_width": self.img_width,
            "arch": "CodeMaster-CRNN",
        }


