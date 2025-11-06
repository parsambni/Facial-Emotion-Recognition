import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset_jaffe import create_jaffe_dataloaders
from model_jaffe import JAFFE_CNN, JAFFE_SimpleCNN
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*60}")
print(f"JAFFE EMOTION RECOGNITION TRAINING")
print(f"{'='*60}")
print(f"Using device: {device}\n")

# Load JAFFE dataset
train_loader, val_loader, test_loader, class_weights, class_names = create_jaffe_dataloaders(
    data_dir='../data/JAFFE/jaffe',
    batch_size=16,  # Smaller batch size for small dataset
    person_independent=True  # More realistic evaluation
)

print(f"Classes: {class_names}\n")

# Initialize model - Choose one:
# model = JAFFE_SimpleCNN(num_classes=7).to(device)  # Simpler model
model = JAFFE_CNN(num_classes=7).to(device)  # Residual model (recommended)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {model.__class__.__name__}")
print(f"Total parameters: {total_params:,}\n")

# Loss with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# Optimizer - Lower learning rate and weight decay for small dataset
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)

# ReduceLROnPlateau scheduler - reduce LR when validation plateaus
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

# Training parameters
num_epochs = 100  # More epochs due to small dataset and slow learning rate
best_val_acc = 0.0
patience = 20  # More patience for small dataset
patience_counter = 0

# For tracking history
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

# Create models directory if it doesn't exist
os.makedirs('../models', exist_ok=True)

print(f"Training for up to {num_epochs} epochs with early stopping...")
print(f"{'='*60}\n")

for epoch in range(num_epochs):
    # ========== TRAINING ==========
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{num_epochs} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    train_loss = train_loss / len(train_loader)
    train_acc = 100.0 * train_correct / train_total
    
    # ========== VALIDATION ==========
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:3d}/{num_epochs} [Val]  ")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_acc = 100.0 * val_correct / val_total
    
    # Update learning rate based on validation accuracy
    scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # Print epoch results
    print(f"\nEpoch [{epoch+1:3d}/{num_epochs}]")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    print(f"  Learning Rate: {current_lr:.6f}")
    
    # Save best model and early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'class_names': class_names
        }, '../models/best_model_jaffe.pth')
        print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n  Early stopping triggered after {patience} epochs without improvement.")
            break
    print()

print(f"\n{'='*60}")
print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
print(f"{'='*60}\n")

# ========== FINAL TEST EVALUATION ==========
print("Loading best model and evaluating on test set...")
checkpoint = torch.load('../models/best_model_jaffe.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_correct = 0
test_total = 0
class_correct = [0] * 7
class_total = [0] * 7

# Confusion matrix
confusion_matrix = torch.zeros(7, 7, dtype=torch.int32)

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        
        # Per-class accuracy
        for i in range(len(labels)):
            label = labels[i]
            pred = predicted[i]
            class_correct[label] += (pred == label).item()
            class_total[label] += 1
            confusion_matrix[label][pred] += 1

test_acc = 100.0 * test_correct / test_total

print(f"\n{'='*60}")
print(f"FINAL RESULTS ON TEST SET (UNSEEN PEOPLE)")
print(f"{'='*60}")
print(f"\n✓ Overall Test Accuracy: {test_acc:.2f}%\n")

print("Per-class accuracy:")
for i, class_name in enumerate(class_names):
    if class_total[i] > 0:
        class_acc = 100.0 * class_correct[i] / class_total[i]
        print(f"  {class_name:10s}: {class_acc:5.2f}% ({class_correct[i]}/{class_total[i]})")

print(f"\n{'='*60}")

# ========== PLOT TRAINING HISTORY ==========
print("\nGenerating training history plots...")

plt.figure(figsize=(12, 4))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc', linewidth=2)
plt.plot(history['val_acc'], label='Val Acc', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../models/jaffe_training_history.png', dpi=150, bbox_inches='tight')
print(f"✓ Training history saved to '../models/jaffe_training_history.png'")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(confusion_matrix.numpy(), interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
plt.colorbar()
tick_marks = range(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

# Add text annotations
thresh = confusion_matrix.max() / 2.
for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        plt.text(j, i, format(confusion_matrix[i, j].item(), 'd'),
                ha="center", va="center",
                color="white" if confusion_matrix[i, j] > thresh else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('../models/jaffe_confusion_matrix.png', dpi=150, bbox_inches='tight')
print(f"✓ Confusion matrix saved to '../models/jaffe_confusion_matrix.png'")

print("\n✓ JAFFE training complete!")
