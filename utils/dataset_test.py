
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# ── CIFAR-10 test loader ─────────────────────────────────────────────
def get_cifar10_test_loader(data_path, batch_size=32):
    """
    Loads CIFAR-10 test images.
    No augmentation — only normalize for final evaluation.

    Args:
        data_path  : path to cifar10/test folder
        batch_size : images per batch (default 32)
    Returns:
        test_loader : DataLoader ready for final evaluation
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4925, 0.4828, 0.4478],
            std= [0.2470, 0.2438, 0.2618]
        )
    ])

    dataset = ImageFolder(data_path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = True
    )

    print(f"CIFAR-10 test loader ready!")
    print(f"  Images    : {len(dataset)}")
    print(f"  Classes   : {dataset.classes}")
    print(f"  Batches   : {len(loader)}")
    print(f"  Batch size: {batch_size}")

    return loader


# ── Tiny ImageNet test loader ────────────────────────────────────────
def get_tinyimagenet_test_loader(data_path, batch_size=32):
    """
    Loads Tiny ImageNet test images.
    No augmentation — only normalize for final evaluation.

    Args:
        data_path  : path to tiny-imagenet-10/test folder
        batch_size : images per batch (default 32)
    Returns:
        test_loader : DataLoader ready for final evaluation
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4674, 0.4490, 0.3984],
            std= [0.2712, 0.2587, 0.2671]
        )
    ])

    dataset = ImageFolder(data_path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = True
    )

    print(f"Tiny ImageNet test loader ready!")
    print(f"  Images    : {len(dataset)}")
    print(f"  Classes   : {dataset.classes}")
    print(f"  Batches   : {len(loader)}")
    print(f"  Batch size: {batch_size}")

    return loader
