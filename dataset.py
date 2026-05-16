import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import ssl

# Fix for macOS SSL certificate issue
ssl._create_default_https_context = ssl._create_unverified_context


def get_transforms(input_size=224):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_dataset(root='./data', split='trainval', download=True, input_size=224):
    """
    Loads the Oxford-IIIT Pet dataset.
    split: 'trainval' or 'test'
    """
    transform = get_transforms(input_size)
    dataset = datasets.OxfordIIITPet(root=root, split=split, target_types='category', download=download, transform=transform)
    return dataset

def get_subset(dataset, num_classes=10, images_per_class=None):
    """
    Creates a subset of the dataset with a specific number of classes.
    """
    # Get indices for the first num_classes
    targets = np.array(dataset._labels)
    classes = np.unique(targets)
    
    if num_classes > len(classes):
        num_classes = len(classes)
        
    selected_classes = classes[:num_classes]
    indices = []
    
    for cls in selected_classes:
        cls_indices = np.where(targets == cls)[0]
        if images_per_class:
             cls_indices = cls_indices[:images_per_class]
        indices.extend(cls_indices)
        
    return Subset(dataset, indices)

def get_dataloader(dataset, batch_size=32, shuffle=True, num_workers=2):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

if __name__ == "__main__":
    # Test the dataset loader
    print("Testing dataset loader...")
    ds = get_dataset(download=True)
    print(f"Dataset size: {len(ds)}")
    subset = get_subset(ds, num_classes=5)
    print(f"Subset size: {len(subset)}")
    loader = get_dataloader(subset, batch_size=4)
    images, labels = next(iter(loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
