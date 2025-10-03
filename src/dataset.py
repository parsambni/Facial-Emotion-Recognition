import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

transform = transforms.Compose([
    transforms.Resize((40, 40)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

train_dataset = ImageFolder(root='data/train', transform=transform)
test_dataset = ImageFolder(root='data/test', transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
