# -*- coding: utf-8 -*-
"""Copy of EVA4 - Session 2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jOUIjtw8CvnmMqT494jZI3nYm2QOipz7
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torchvision import datasets, transforms
from tqdm import tqdm  # Add this import


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First block - reduced initial channels
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # Changed from 32 to 8
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)  # Changed from 64 to 16
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.1)  # Reduced dropout

        # Second block - reduced channels
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)  # Changed from 128 to 16
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)  # Changed from 256 to 32
        self.bn4 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.1)

        # Final block - now using GAP
        self.conv5 = nn.Conv2d(32, 10, 1)  # 1x1 convolution to get 10 channels
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling layer

    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Final block with GAP
        x = self.conv5(x)
        x = self.gap(x)
        x = x.view(-1, 10)  # Safe to use -1 here as GAP ensures fixed dimensions
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(desc=f"loss={loss.item()} batch_id={batch_idx}")


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    # Set random seed for reproducibility
    torch.manual_seed(1)

    # Check CUDA availability
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Training hyperparameters
    batch_size = 128
    epochs = 10
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    # Initialize model and move to device
    model = Net().to(device)

    # Print model summary
    summary(model, input_size=(1, 28, 28))

    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training loop
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)


if __name__ == "__main__":
    main()
