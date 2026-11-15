"""Train a simple convolutional neural network on the MNIST dataset."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from test_pytorch.model import MnistCNN, MnistMLP
from test_pytorch.training import evaluate, train_one_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="examples per batch")
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="directory for MNIST data"
    )
    parser.add_argument(
        "--model",
        choices=("cnn", "mlp"),
        default="cnn",
        help="model architecture (default: %(default)s)",
    )
    parser.add_argument("--output", type=Path, help="weights file")
    args = parser.parse_args()

    # Use a GPU when available; otherwise, train on the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Convert each image to a tensor and normalize its pixels to aid
    # optimization. torchvision downloads MNIST to the specified directory if needed.
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_data = datasets.MNIST(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root=args.data_dir, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    model_class = MnistCNN if args.model == "cnn" else MnistMLP
    model = model_class().to(device)
    output_path = args.output or Path(f"data/mnist_{args.model}.pt")
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Each epoch processes all training examples; afterward, measure accuracy
    # on test images that were not used to update the weights.
    for epoch in range(1, args.epochs + 1):
        average_loss = train_one_epoch(
            model, train_loader, loss_function, optimizer, device
        )
        accuracy = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs} — "
            f"loss: {average_loss:.4f}, test accuracy: {accuracy:.2%}"
        )

    # Save the trained weights, creating the destination directory if needed.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Weights saved to: {output_path}")
