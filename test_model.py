import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from train import Net, get_train_val_loaders, test, train


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def model(device):
    return Net().to(device)


@pytest.fixture
def data_loaders():
    batch_size = 64
    use_cuda = torch.cuda.is_available()
    train_loader, val_loader = get_train_val_loaders(batch_size, use_cuda)
    return train_loader, val_loader


def test_model_accuracy(model, device, data_loaders):
    train_loader, val_loader = data_loaders

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.003,  # Standard Adam learning rate
        betas=(0.9, 0.999),  # Default Adam betas
        eps=1e-08,  # Default epsilon
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=4,
        gamma=0.6,
    )

    best_val_acc = 0
    best_epoch = 0

    for epoch in range(20):
        train(model, device, train_loader, optimizer)
        val_acc = test(model, device, val_loader, save_misclassified=False, epoch=epoch)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), "best_model_test.pth")

    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    assert (
        best_val_acc >= 99.4
    ), f"Validation accuracy {best_val_acc:.2f}% is below 99.4%. Best epoch {best_epoch}"


def test_parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    assert (
        total_params < 20000
    ), f"Model has {total_params} parameters, which exceeds the limit of 20000"


def test_batch_normalization_usage(model):
    has_batch_norm = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert has_batch_norm, "Model should use BatchNormalization"


def test_dropout_usage(model):
    has_dropout = any(isinstance(m, nn.Dropout2d) for m in model.modules())
    assert has_dropout, "Model should use Dropout"


def test_gap_vs_fc_usage(model):
    has_gap = any(isinstance(m, nn.AdaptiveAvgPool2d) for m in model.modules())
    assert has_gap, "Model should use Global Average Pooling"
