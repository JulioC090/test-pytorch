"""Package for training a CNN on the MNIST dataset."""

from test_pytorch.cli import main
from test_pytorch.model import MnistCNN, MnistMLP
from test_pytorch.training import evaluate, train_one_epoch

__all__ = ["MnistCNN", "MnistMLP", "evaluate", "main", "train_one_epoch"]
