import torch
import torch.nn as nn
import torch.nn.functional as F

class FER_CNN(nn.Module):
    def __init__(self, num_classes=7):
        super(FER_CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Batch normalization
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)

        # Regularization
        self.dropout = nn.Dropout(p=0.5)

        # Calculate the size after convolution and pooling
        # Input: 48x48
        # After conv1 + pool1: 48x48 -> 24x24
        # After conv2 + pool2: 24x24 -> 12x12
        # After conv3 + pool3: 12x12 -> 6x6
        # So: 128 * 6 * 6 = 4608
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # First conv block
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Second conv block
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Third conv block
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

# Test the model dimensions
if __name__ == "__main__":
    model = FER_CNN(num_classes=7)
    
    # Test with a dummy input (batch_size=1, channels=1, height=48, width=48)
    test_input = torch.randn(1, 1, 48, 48)
    output = model(test_input)
    
    print(f"Model output shape: {output.shape}")
    print(f"Expected shape: [1, 7]")
    print("✓ Model architecture is correct!")
