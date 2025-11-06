import torch
import torch.nn as nn
import torch.nn.functional as F


class JAFFE_CNN(nn.Module):
    """
    Optimized CNN for JAFFE dataset based on successful research approaches.
    
    Design based on papers achieving 70-95% on JAFFE:
    - Moderate depth (4 conv layers)
    - Progressive channel expansion: 64->128->256->512
    - Strong regularization (BatchNorm + Dropout)
    - Global pooling to reduce parameters
    
    Target: 70-90% accuracy with person-independent split
    """
    def __init__(self, num_classes=7):
        super(JAFFE_CNN, self).__init__()
        
        # Feature extraction layers
        # Block 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 128 -> 64
        
        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 64 -> 32
        
        # Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)  # 32 -> 16
        
        # Block 4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)  # 16 -> 8
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Global pooling
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x


# Test the model
if __name__ == "__main__":
    print("="*60)
    print("Testing JAFFE_CNN Model")
    print("="*60)
    
    model = JAFFE_CNN(num_classes=7)
    test_input = torch.randn(2, 1, 128, 128)  # Batch of 2 images
    
    print(f"\nInput shape: {test_input.shape}")
    
    output = model(test_input)
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Model size in MB
    param_size = total_params * 4 / (1024 ** 2)  # 4 bytes per float32
    print(f"Model size: {param_size:.2f} MB")
    
    print("\n✓ Model architecture validated successfully!")
    print("="*60)
