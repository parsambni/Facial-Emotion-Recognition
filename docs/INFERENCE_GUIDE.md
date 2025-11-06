# JAFFE Emotion Recognition - Inference Guide

This guide shows you how to use the trained model to predict emotions from images or webcam.

## Prerequisites

Make sure you have trained the model first:
```bash
python train.py
```

This will create `../../models/best_model_jaffe.pth`

## Option 1: Predict Emotion from an Image

Use `predict_image.py` to analyze a single image:

### Basic Usage:
```bash
python predict_image.py path/to/your/image.jpg
```

### Show All Emotion Probabilities:
```bash
python predict_image.py path/to/your/image.jpg --show-all
```

### Examples:
```bash
# Predict emotion from a JAFFE test image
python predict_image.py ../../data/JAFFE/jaffe/KA.HA1.30.tiff

# With detailed probabilities
python predict_image.py ../../data/JAFFE/jaffe/YM.SA1.213.tiff --show-all

# Use a custom model
python predict_image.py myface.jpg --model path/to/model.pth
```

### Output Example:
```
Using device: cpu

Loading model...
Model loaded successfully!

Processing image: test.jpg

============================================================
EMOTION PREDICTION RESULT
============================================================

🎭 Predicted Emotion: HAPPY
📊 Confidence: 87.45%

All emotion probabilities:
----------------------------------------
  happy     : 87.45% ███████████████████████████████████████████
  surprise  : 6.32%  ███
  neutral   : 3.21%  █
  disgust   : 1.54%  
  angry     : 0.89%  
  fear      : 0.43%  
  sad       : 0.16%  

============================================================
```

## Option 2: Real-time Webcam Emotion Detection

Use `predict_webcam.py` for real-time emotion detection:

### Basic Usage:
```bash
python predict_webcam.py
```

### Advanced Options:
```bash
# Use a different camera (e.g., external webcam)
python predict_webcam.py --camera 1

# Skip face detection and use full frame
python predict_webcam.py --no-face-detection

# Use custom model
python predict_webcam.py --model path/to/model.pth
```

### Controls:
- **'q'** - Quit the application
- **'s'** - Save current frame with emotion label

### Features:
- ✅ Automatic face detection
- ✅ Real-time emotion prediction
- ✅ Color-coded emotion labels:
  - 🔴 Red: Angry
  - 🟣 Purple: Disgust
  - 🟠 Orange: Fear
  - 🟢 Green: Happy
  - ⚪ White: Neutral
  - 🔵 Blue: Sad
  - 🟡 Yellow: Surprise
- ✅ Confidence percentage display
- ✅ Multiple face detection support

## Troubleshooting

### "Model file not found"
Train the model first:
```bash
python train.py
```

### "Could not open camera"
1. Check if your camera is connected
2. Try a different camera index: `--camera 1`
3. Check camera permissions

### "No faces detected"
1. Make sure you're facing the camera
2. Ensure good lighting
3. Try `--no-face-detection` to analyze full frame

### Low accuracy
The model works best with:
- Good lighting conditions
- Clear frontal face view
- Similar facial features to Japanese females (trained on JAFFE dataset)
- For better accuracy, consider retraining with more diverse data

## Supported Image Formats

- JPEG/JPG
- PNG
- TIFF
- BMP
- And most common image formats supported by PIL

## Requirements

```bash
pip install torch torchvision opencv-python pillow numpy
```

## Model Information

- **Architecture:** JAFFE_CNN (4 convolutional layers)
- **Input Size:** 128x128 grayscale
- **Classes:** 7 emotions (angry, disgust, fear, happy, neutral, sad, surprise)
- **Training Dataset:** JAFFE (Japanese Female Facial Expression)

## Tips for Best Results

1. **Image Quality:** Use clear, well-lit images
2. **Face Position:** Frontal face view works best
3. **Face Size:** Larger faces in the image give better results
4. **Lighting:** Even lighting without harsh shadows
5. **Expression:** Clear, pronounced facial expressions

## Example Workflow

```bash
# 1. Train the model (if not already trained)
python train.py

# 2. Test on a single image
python predict_image.py test_face.jpg --show-all

# 3. Try real-time detection
python predict_webcam.py

# 4. Save interesting frames
# (Press 's' while running webcam script)
```

## Notes

- The webcam script processes every 5th frame to maintain smooth performance
- Face detection uses OpenCV's Haar Cascade classifier
- The model was trained on JAFFE dataset (Japanese female faces), so it may work better on similar demographics
- For production use, consider fine-tuning on your target demographic

---

For questions or issues, check the main README or open an issue on GitHub.
