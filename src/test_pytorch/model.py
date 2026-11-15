"""CNN model definition for MNIST."""

import torch
from torch import nn


class MnistMLP(nn.Module):
    """Fully connected network for classifying handwritten digit images."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(images)


class MnistCNN(nn.Module):
    """Compact CNN for classifying handwritten digit images."""

    def __init__(self) -> None:
        super().__init__()
        # Two convolutional layers learn visual patterns; pooling reduces the
        # image dimensions between them to lower the cost of later computations.
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # After the convolutions, features are flattened and mapped to ten
        # scores, one for each digit from 0 to 9.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
