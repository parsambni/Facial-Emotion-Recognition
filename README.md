# 🎭 Facial Emotion Recognition

A deep learning project for real-time emotion detection from facial expressions using PyTorch and Convolutional Neural Networks (CNNs).

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Models](#models)
- [Datasets](#datasets)
- [Usage](#usage)
- [Training](#training)
- [Results](#results)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

This project implements state-of-the-art deep learning models for recognizing seven basic human emotions from facial images:

- 😠 **Angry**
- 🤢 **Disgust**
- 😨 **Fear**
- 😊 **Happy**
- 😐 **Neutral**
- 😢 **Sad**
- 😲 **Surprise**

The system supports both static image prediction and real-time webcam emotion detection with an easy-to-use interface.

## ✨ Features

- 🎯 **High Accuracy**: Up to 70-75% accuracy on FER2013 dataset
- 🚀 **Real-time Detection**: Webcam-based emotion recognition
- 🖼️ **Image Prediction**: Analyze emotions from static images
- 🔄 **Multiple Models**: Choose between JAFFE and FER2013 trained models
- 📊 **Interactive Selection**: User-friendly model selection interface
- 🎨 **Visual Feedback**: Color-coded emotion labels and probability bars
- 🔧 **Easy Training**: Simple scripts for training custom models
- 📈 **Progress Tracking**: Real-time training metrics and visualization

## 🎬 Demo

### Image Prediction
```bash
cd src/inference
python predict_image.py ../../test_images/your_image.jpg --show-all
```

### Webcam Detection
```bash
cd src/inference
python predict_webcam.py
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/parsambni/Facial-Emotion-Recognition.git
   cd Facial-Emotion-Recognition
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   ```

## 🚀 Quick Start

### Option 1: Use Pre-trained Model

1. Download a pre-trained model (if available):
   ```bash
   # Model will be in models/best_model_fer2013.pth or models/best_model_jaffe.pth
   ```

2. Run prediction:
   ```bash
   cd src/inference
   python predict_image.py ../../test_images/sample.jpg --show-all
   ```

### Option 2: Train Your Own Model

1. Ensure dataset is in place (see [Datasets](#datasets))

2. Train FER2013 model:
   ```bash
   cd src/training
   python train_fer2013.py
   ```

3. Test the trained model:
   ```bash
   cd ../inference
   python predict_image.py ../../data/FER2013/test/happy/sample.jpg
   ```

## 📁 Project Structure

```
Facial-Emotion-Recognition/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
│
├── data/                    # Datasets
│   ├── FER2013/            # FER2013 dataset (35,887 images)
│   │   ├── train/          # Training images by emotion
│   │   └── test/           # Test images by emotion
│   └── JAFFE/              # JAFFE dataset (213 images)
│       └── jaffe/          # Japanese female facial expressions
│
├── models/                  # Saved trained models
│   ├── best_model_fer2013.pth
│   └── best_model_jaffe.pth
│
├── src/                     # Source code
│   ├── models/             # Model architectures
│   │   ├── fer2013_model.py    # ImprovedFER_CNN (ResNet-style)
│   │   └── jaffe_model.py      # JAFFE_CNN
│   │
│   ├── data/               # Dataset loaders
│   │   ├── fer2013_dataset.py  # FER2013 data pipeline
│   │   └── jaffe_dataset.py    # JAFFE data pipeline
│   │
│   ├── training/           # Training scripts
│   │   ├── train_fer2013.py    # Train FER2013 model
│   │   └── train_jaffe.py      # Train JAFFE model
│   │
│   └── inference/          # Prediction scripts
│       ├── predict_image.py    # Static image prediction
│       └── predict_webcam.py   # Real-time webcam detection
```

## 🧠 Models

### 1. ImprovedFER_CNN (FER2013)
- **Architecture**: ResNet-style with residual blocks
- **Input Size**: 48×48 grayscale
- **Parameters**: ~2-3M
- **Dataset**: FER2013 (35,887 images)
- **Expected Accuracy**: 70-75%
- **Best For**: Production use, diverse faces

### 2. JAFFE_CNN
- **Architecture**: 4-layer CNN with global pooling
- **Input Size**: 128×128 grayscale
- **Parameters**: ~1.6M
- **Dataset**: JAFFE (213 images)
- **Accuracy**: 56% validation / 39% test
- **Best For**: Japanese female faces, research

For detailed comparison, see [MODEL_COMPARISON.md](MODEL_COMPARISON.md)

## 📊 Datasets

### FER2013 (Recommended)
- **Size**: 35,887 images
- **Split**: 28,709 train / 7,178 test
- **Resolution**: 48×48 pixels
- **Source**: Kaggle FER2013 competition
- **Diversity**: Multiple ethnicities, ages, genders

### JAFFE
- **Size**: 213 images
- **Subjects**: 10 Japanese female models
- **Resolution**: 256×256 pixels
- **Source**: Japanese Female Facial Expression database
- **Use Case**: Research, controlled conditions

## 💻 Usage

### Image Prediction

**Interactive mode** (choose model):
```bash
cd src/inference
python predict_image.py path/to/image.jpg --show-all
```

**Specify model directly**:
```bash
python predict_image.py path/to/image.jpg --model ../../models/best_model_fer2013.pth --show-all
```

### Webcam Detection

**Start real-time detection**:
```bash
cd src/inference
python predict_webcam.py
```

**Use specific model**:
```bash
python predict_webcam.py --model ../../models/best_model_fer2013.pth
```

**Controls**:
- Press `q` to quit
- Press `s` to save screenshot

### Python API

```python
from models.fer2013_model import ImprovedFER_CNN
import torch
from PIL import Image

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedFER_CNN(num_classes=7).to(device)
checkpoint = torch.load('models/best_model_fer2013.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
image = Image.open('path/to/image.jpg')
# ... preprocess image ...
with torch.no_grad():
    output = model(image_tensor)
    emotion = torch.argmax(output, dim=1)
```

## 🎓 Training

### Train FER2013 Model

```bash
cd src/training
python train_fer2013.py
```

**Training Features**:
- ✅ Cosine annealing learning rate schedule
- ✅ Class weight balancing for imbalanced data
- ✅ Strong data augmentation (flip, rotation, affine, color jitter)
- ✅ Early stopping with patience
- ✅ Automatic best model checkpointing
- ✅ Real-time progress bars with tqdm
- ✅ Validation monitoring every epoch

**Expected Training Time**:
- CPU: 4-6 hours
- GPU: 30-60 minutes

### Train JAFFE Model

```bash
cd src/training
python train_jaffe.py
```

For detailed training instructions, see [TRAIN_FER2013.md](TRAIN_FER2013.md)

## 📈 Results

### FER2013 Model Performance

| Metric | Value |
|--------|-------|
| **Training Accuracy** | ~75-80% |
| **Validation Accuracy** | ~70-75% |
| **Test Accuracy** | ~70-75% |
| **Parameters** | 2-3M |
| **Training Time (GPU)** | ~45 min |

### JAFFE Model Performance

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 56.1% |
| **Test Accuracy** | 39.5% |
| **Parameters** | 1.6M |
| **Training Time** | ~10 min |

### Per-Class Accuracy (FER2013)

| Emotion | Accuracy |
|---------|----------|
| Happy | ~85% |
| Surprise | ~80% |
| Neutral | ~75% |
| Sad | ~70% |
| Angry | ~65% |
| Fear | ~60% |
| Disgust | ~55% |

## 📚 Documentation

- **[MODEL_COMPARISON.md](MODEL_COMPARISON.md)**: Detailed architecture comparison
- **[FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md)**: Project organization
- **[QUICK_START.md](QUICK_START.md)**: Getting started guide
- **[TRAIN_FER2013.md](TRAIN_FER2013.md)**: Training instructions

## 🔧 Configuration

### Modify Training Parameters

Edit `src/training/train_fer2013.py`:

```python
# Training hyperparameters
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
PATIENCE = 15

# Data augmentation strength
transforms.RandomRotation(degrees=15)  # Adjust rotation
transforms.RandomErasing(p=0.2)        # Adjust erasing probability
```

### Modify Model Architecture

Edit `src/models/fer2013_model.py`:

```python
# Add more residual blocks
self.layer3 = self._make_layer(256, 512, num_blocks=3)  # Increase depth

# Adjust dropout
self.dropout = nn.Dropout(0.5)  # Increase for more regularization
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to functions
- Include unit tests for new features
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FER2013 Dataset**: [Kaggle Competition](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
- **JAFFE Dataset**: [Japanese Female Facial Expression Database](https://zenodo.org/record/3451524)
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision library

## 📧 Contact

**Parsa Sambni** - [@parsambni](https://github.com/parsambni)

Project Link: [https://github.com/parsambni/Facial-Emotion-Recognition](https://github.com/parsambni/Facial-Emotion-Recognition)

## 🎯 Future Work

- [ ] Add more emotion categories (contempt, embarrassment)
- [ ] Implement attention mechanisms
- [ ] Add ensemble model support
- [ ] Create REST API for cloud deployment
- [ ] Mobile app integration
- [ ] Real-time video stream processing
- [ ] Multi-face detection support
- [ ] Emotion intensity scoring
- [ ] Cross-dataset evaluation
- [ ] Model compression for edge devices

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/parsambni">Parsa Mobini Dehkordi</a>
</p>

<p align="center">
  If you find this project helpful, please give it a ⭐!
</p>
