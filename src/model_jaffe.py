import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Simple residual block with skip connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)  # Add skip connection
        out = F.relu(out)
        return out


class JAFFE_CNN(nn.Module):
    """
    CNN model optimized for JAFFE dataset (Japanese Female Facial Expression)
    
    Key differences from FER2013 model:
    - Input size: 128x128 (JAFFE images are higher resolution than FER2013's 48x48)
    - Smaller model to prevent overfitting (only 213 images total)
    - Less aggressive dropout due to small dataset
    - Fewer filters in early layers
    """
    def __init__(self, num_classes=7):
        super(JAFFE_CNN, self).__init__()
        
        # Initial convolution (smaller kernel for higher resolution images)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks (fewer channels to prevent overfitting)
        self.layer1 = self._make_layer(32, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier with less dropout (small dataset needs less regularization)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class JAFFE_SimpleCNN(nn.Module):
    """
    Simpler CNN model for JAFFE dataset
    Good for when you have very limited data
    """
    def __init__(self, num_classes=7):
        super(JAFFE_SimpleCNN, self).__init__()

        # Convolutional layers (smaller than FER2013 version)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Batch normalization
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.batch_norm3 = nn.BatchNorm2d(64)

        # Less dropout for small dataset
        self.dropout = nn.Dropout(p=0.3)
        
        # Fully connected layers (adjusted for 128x128 input)
        # After 3 pooling layers: 128 -> 64 -> 32 -> 16
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

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


# Test the models
if __name__ == "__main__":
    print("Testing JAFFE models...\n")
    
    # Test simple model
    print("Testing JAFFE_SimpleCNN...")
    model1 = JAFFE_SimpleCNN(num_classes=7)
    test_input = torch.randn(1, 1, 128, 128)
    output1 = model1(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output1.shape}")
    
    # Test residual model
    print("\nTesting JAFFE_CNN (with residual blocks)...")
    model2 = JAFFE_CNN(num_classes=7)
    output2 = model2(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output2.shape}")
    
    # Count parameters
    params1 = sum(p.numel() for p in model1.parameters())
    params2 = sum(p.numel() for p in model2.parameters())
    print(f"\nSimple model parameters: {params1:,}")
    print(f"Residual model parameters: {params2:,}")
    print("\n✓ Both JAFFE models work correctly!")
