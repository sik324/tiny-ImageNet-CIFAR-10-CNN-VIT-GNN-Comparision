
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# ── CIFAR-10 training loader ─────────────────────────────────────────
def get_cifar10_train_loader(data_path, batch_size=64):
    """
    Loads CIFAR-10 training images.
    Applies augmentation + normalization.

    Args:
        data_path  : path to cifar10/train folder
        batch_size : images per batch (default 64)
    Returns:
        train_loader : DataLoader ready for CNN
    """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
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
        shuffle     = True,
        num_workers = 2,
        pin_memory  = True
    )

    print(f"CIFAR-10 train loader ready!")
    print(f"  Images    : {len(dataset)}")
    print(f"  Classes   : {dataset.classes}")
    print(f"  Batches   : {len(loader)}")
    print(f"  Batch size: {batch_size}")

    return loader


# ── Tiny ImageNet training loader ────────────────────────────────────
def get_tinyimagenet_train_loader(data_path, batch_size=64):
    """
    Loads Tiny ImageNet training images.
    Applies augmentation + normalization.
    Note: Tiny ImageNet has extra images/ subfolder inside each class.

    Args:
        data_path  : path to tiny-imagenet-10/train folder
        batch_size : images per batch (default 64)
    Returns:
        train_loader : DataLoader ready for CNN
    """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(64, padding=8),
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
        shuffle     = True,
        num_workers = 2,
        pin_memory  = True
    )

    print(f"Tiny ImageNet train loader ready!")
    print(f"  Images    : {len(dataset)}")
    print(f"  Classes   : {dataset.classes}")
    print(f"  Batches   : {len(loader)}")
    print(f"  Batch size: {batch_size}")

    return loader
