from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ImageNet normalisation constants — matches the DeiT pretrained weights.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

_VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


def get_cifar100_loaders(
    batch_size: int = 32,
    data_root: str = "/home/arooba/compute-aware-vit-thesis/data/",
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    root = Path(data_root)
    train_ds = datasets.CIFAR100(root, train=True,  transform=_TRAIN_TRANSFORMS, download=False)
    val_ds   = datasets.CIFAR100(root, train=False, transform=_VAL_TRANSFORMS,   download=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
