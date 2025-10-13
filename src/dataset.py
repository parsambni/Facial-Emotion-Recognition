import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import numpy as np

# STRONGER data augmentation for better generalization
train_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    # More aggressive augmentation
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    # Add noise through random erasing
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
])

# Clean test transform
test_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load datasets
print("Loading datasets...")
train_dataset_full = ImageFolder(root='../data/FER2013/train', transform=train_transform)
test_dataset = ImageFolder(root='../data/FER2013/test', transform=test_transform)

# Split into train and validation (85/15 split for more training data)
train_size = int(0.85 * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size
train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

print(f"✓ Training samples: {len(train_dataset)}")
print(f"✓ Validation samples: {len(val_dataset)}")
print(f"✓ Test samples: {len(test_dataset)}")

# Calculate class weights
class_counts = np.zeros(len(train_dataset_full.classes))
for _, label in train_dataset_full:
    class_counts[label] += 1

print("\nClass distribution:")
for i, (name, count) in enumerate(zip(train_dataset_full.classes, class_counts)):
    print(f"  {name}: {int(count)} images")

# Create class weights (stronger for minority classes)
class_weights = torch.FloatTensor(1.0 / class_counts)
class_weights = class_weights / class_weights.sum() * len(class_weights)

# Larger batch size for more stable training
BATCH_SIZE = 128

# Create data loaders with more workers for faster loading
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                         num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                       num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=4, pin_memory=True)

class_names = train_dataset_full.classes

print(f"\n✓ Data loaders ready with batch size {BATCH_SIZE}!")
