import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import os

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

# Load dataset from folders
train_dataset_full = ImageFolder(root='../data/train', transform=train_transform)
test_dataset = ImageFolder(root='../data/test', transform=test_transform)

# Train/validation split
train_size = int(0.8 * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size
train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

print(f"✓ Training dataset loaded: {len(train_dataset_full)} images")
print(f"✓ Test dataset loaded: {len(test_dataset)} images")
print(f"✓ Classes found: {train_dataset_full.classes}")

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Class names
class_names = train_dataset_full.classes 