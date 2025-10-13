import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import train_loader, val_loader, test_loader, class_weights, class_names
from model import ImprovedFER_CNN
from tqdm import tqdm
import os

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*60}")
print(f"IMPROVED TRAINING FOR 90%+ ACCURACY")
print(f"{'='*60}")
print(f"Using device: {device}")
print(f"Classes: {class_names}\n")

# Initialize IMPROVED model
model = ImprovedFER_CNN(num_classes=7).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}\n")

# Loss with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# Better optimizer with lower learning rate
optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)

# Cosine annealing scheduler (smooth decay)
num_epochs = 50  # More epochs for better convergence
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# For saving best model
best_val_acc = 0.0
patience = 10
patience_counter = 0

# Create models directory if it doesn't exist
os.makedirs('../models', exist_ok=True)

print(f"Training for {num_epochs} epochs...")
print(f"{'='*60}\n")

for epoch in range(num_epochs):
    # ========== TRAINING ==========
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
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
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]  ")
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
    
    # Update learning rate
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # Print epoch results
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")
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
        }, '../models/best_model_improved.pth')
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
checkpoint = torch.load('../models/best_model_improved.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_correct = 0
test_total = 0
class_correct = [0] * 7
class_total = [0] * 7

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
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1

test_acc = 100.0 * test_correct / test_total

print(f"\n{'='*60}")
print(f"FINAL RESULTS")
print(f"{'='*60}")
print(f"\n✓ Overall Test Accuracy: {test_acc:.2f}%\n")

print("Per-class accuracy:")
for i, class_name in enumerate(class_names):
    if class_total[i] > 0:
        class_acc = 100.0 * class_correct[i] / class_total[i]
        print(f"  {class_name:10s}: {class_acc:.2f}%")

print(f"\n{'='*60}")
