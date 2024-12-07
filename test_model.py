import pytest
import torch
import torch.nn as nn

from train import Net, get_train_val_loaders, test, train


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def model(device):
    model = Net().to(device)
    return model


@pytest.fixture
def data_loaders():
    use_cuda = torch.cuda.is_available()
    return get_train_val_loaders(batch_size=128, use_cuda=use_cuda)


def test_model_accuracy(model, device, data_loaders):
    train_loader, val_loader = data_loaders
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    best_train_acc = 0
    best_val_acc = 0

    for epoch in range(20):
        train(model, device, train_loader, optimizer, epoch)

        train_acc = test(model, device, train_loader)
        best_train_acc = max(best_train_acc, train_acc)

        val_acc = test(model, device, val_loader)
        best_val_acc = max(best_val_acc, val_acc)

    assert (
        best_train_acc >= 99.4
    ), f"Train accuracy {best_train_acc:.2f}% is below 99.4%"
    assert (
        best_val_acc >= 99.4
    ), f"Validation accuracy {best_val_acc:.2f}% is below 99.4%"


def test_parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    assert (
        total_params < 20000
    ), f"Model has {total_params} parameters, which exceeds the limit of 20000"


def test_batch_normalization_usage(model):
    has_batch_norm = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert has_batch_norm, "Model does not use Batch Normalization"

    # Check if batch norm is used after conv layers
    conv_followed_by_bn = False
    prev_layer = None
    for layer in model.modules():
        if isinstance(prev_layer, nn.Conv2d) and isinstance(layer, nn.BatchNorm2d):
            conv_followed_by_bn = True
            break
        prev_layer = layer

    assert (
        conv_followed_by_bn
    ), "Batch Normalization is not properly used after convolution layers"


def test_dropout_usage(model):
    has_dropout = any(isinstance(m, nn.Dropout2d) for m in model.modules())
    assert has_dropout, "Model does not use Dropout"

    # Check dropout values
    dropout_values = [m.p for m in model.modules() if isinstance(m, nn.Dropout2d)]
    assert all(
        0.0 < p <= 0.5 for p in dropout_values
    ), "Dropout values should be between 0 and 0.5"


def test_gap_vs_fc_usage(model):
    # Check for GAP
    has_gap = any(isinstance(m, nn.AdaptiveAvgPool2d) for m in model.modules())
    # Check for FC layers
    has_fc = any(isinstance(m, nn.Linear) for m in model.modules())

    # Model should use either GAP or FC, but not both
    assert has_gap != has_fc, "Model should use either GAP or FC layer, but not both"
    # In this case, we expect GAP
    assert has_gap, "Model should use Global Average Pooling (GAP)"
    assert not has_fc, "Model should not use Fully Connected (FC) layers"
