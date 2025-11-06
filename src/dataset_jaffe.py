import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split

class JAFFEDataset(Dataset):
    """
    Custom Dataset for JAFFE (Japanese Female Facial Expression) Database
    
    JAFFE has 213 images of 7 facial expressions from 10 Japanese female subjects.
    Format: PersonID.Emotion.ImageNumber.tiff
    
    Emotions:
    - AN: Angry
    - DI: Disgust
    - FE: Fear
    - HA: Happy
    - NE: Neutral
    - SA: Sad
    - SU: Surprise
    """
    
    def __init__(self, root_dir, transform=None, file_list=None):
        """
        Args:
            root_dir: Path to JAFFE image directory
            transform: Optional transform to be applied on images
            file_list: Optional list of specific files to use (for train/val/test split)
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Emotion mapping (alphabetical order for consistency)
        self.emotion_map = {
            'AN': 0,  # Angry
            'DI': 1,  # Disgust
            'FE': 2,  # Fear
            'HA': 3,  # Happy
            'NE': 4,  # Neutral
            'SA': 5,  # Sad
            'SU': 6   # Surprise
        }
        
        self.class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Get all image files
        if file_list is not None:
            self.image_files = file_list
        else:
            self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.tiff')]
            self.image_files.sort()
        
        # Parse labels from filenames
        self.labels = []
        for filename in self.image_files:
            # Format: PersonID.Emotion.Number.tiff (e.g., KA.AN1.39.tiff)
            parts = filename.split('.')
            emotion_code = parts[1][:2]  # Extract 'AN', 'HA', etc.
            self.labels.append(self.emotion_map[emotion_code])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        
        # Load TIFF image and convert to RGB (will be converted to grayscale in transform)
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_person_id(self, idx):
        """Get the person ID for an image (useful for person-independent splits)"""
        filename = self.image_files[idx]
        return filename.split('.')[0]


def create_jaffe_dataloaders(data_dir='../data/JAFFE/jaffe', 
                             batch_size=16, 
                             val_split=0.2, 
                             test_split=0.2,
                             person_independent=True):
    """
    Creates train, validation, and test data loaders for JAFFE dataset.
    
    Args:
        data_dir: Path to JAFFE images
        batch_size: Batch size for data loaders
        val_split: Proportion for validation set
        test_split: Proportion for test set
        person_independent: If True, ensures different people in train/val/test sets
    
    Returns:
        train_loader, val_loader, test_loader, class_weights, class_names
    """
    
    # Data augmentation for training (less aggressive than FER2013 due to small dataset)
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # JAFFE images are larger, resize to 128x128
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Test/validation transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Get all files
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.tiff')]
    all_files.sort()
    
    if person_independent:
        # Split by person for more realistic evaluation
        person_files = {}
        for f in all_files:
            person_id = f.split('.')[0]
            if person_id not in person_files:
                person_files[person_id] = []
            person_files[person_id].append(f)
        
        people = list(person_files.keys())
        # 10 people total: 6 for train, 2 for val, 2 for test
        train_people = people[:6]
        val_people = people[6:8]
        test_people = people[8:10]
        
        train_files = [f for p in train_people for f in person_files[p]]
        val_files = [f for p in val_people for f in person_files[p]]
        test_files = [f for p in test_people for f in person_files[p]]
    else:
        # Random split
        train_files, temp_files = train_test_split(all_files, test_size=(val_split + test_split), random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=(test_split / (val_split + test_split)), random_state=42)
    
    print("="*60)
    print("JAFFE Dataset Loading")
    print("="*60)
    print(f"Split method: {'Person-independent' if person_independent else 'Random'}")
    
    # Create datasets
    train_dataset = JAFFEDataset(data_dir, transform=train_transform, file_list=train_files)
    val_dataset = JAFFEDataset(data_dir, transform=test_transform, file_list=val_files)
    test_dataset = JAFFEDataset(data_dir, transform=test_transform, file_list=test_files)
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    
    # Calculate class distribution and weights
    class_counts = np.zeros(7)
    for _, label in train_dataset:
        class_counts[label] += 1
    
    print("\nClass distribution (training set):")
    for i, (name, count) in enumerate(zip(train_dataset.class_names, class_counts)):
        print(f"  {name:10s}: {int(count):3d} images")
    
    # Create class weights for balanced training
    class_weights = torch.FloatTensor(1.0 / (class_counts + 1e-6))  # Add small epsilon to avoid division by zero
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    # Create data loaders (smaller batch size due to small dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=2, pin_memory=True)
    
    print(f"\n✓ Data loaders ready with batch size {batch_size}!")
    print("="*60 + "\n")
    
    return train_loader, val_loader, test_loader, class_weights, train_dataset.class_names


if __name__ == "__main__":
    # Test the dataset
    print("Testing JAFFE Dataset loader...\n")
    
    train_loader, val_loader, test_loader, class_weights, class_names = create_jaffe_dataloaders(
        data_dir='../data/JAFFE/jaffe',
        batch_size=16,
        person_independent=True
    )
    
    # Get one batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"\nClass weights: {class_weights}")
    print(f"\nClass names: {class_names}")
    print("\n✓ JAFFE dataset loaded successfully!")
