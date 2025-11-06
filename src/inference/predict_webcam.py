import torch
import torch.nn.functional as F
import sys
import os
# Add parent directory to path to import models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.fer2013_model import ImprovedFER_CNN
from models.jaffe_model import JAFFE_CNN
import torchvision.transforms as transforms
import cv2
import numpy as np
import argparse
from PIL import Image

def load_model(model_path, device):
    """Load the trained model"""
    # Determine model type from checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get('class_names', ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
    
    # Check if it's FER2013 or JAFFE based on file name
    if 'fer2013' in model_path.lower():
        model = ImprovedFER_CNN(num_classes=7).to(device)
        input_size = 48
    else:
        model = JAFFE_CNN(num_classes=7).to(device)
        input_size = 128
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, class_names, input_size

def preprocess_face(face_image, input_size=48):
    """Preprocess face image for model"""
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    face_tensor = transform(face_pil).unsqueeze(0)
    
    return face_tensor

def predict_emotion(model, face_tensor, class_names, device):
    """Predict emotion from face tensor"""
    face_tensor = face_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(face_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    emotion = class_names[predicted.item()]
    conf = confidence.item() * 100
    
    return emotion, conf

def draw_emotion_label(frame, face_rect, emotion, confidence):
    """Draw emotion label on frame"""
    x, y, w, h = face_rect
    
    # Colors for different emotions (BGR format)
    emotion_colors = {
        'angry': (0, 0, 255),      # Red
        'disgust': (128, 0, 128),  # Purple
        'fear': (0, 165, 255),     # Orange
        'happy': (0, 255, 0),      # Green
        'neutral': (255, 255, 255), # White
        'sad': (255, 0, 0),        # Blue
        'surprise': (0, 255, 255)  # Yellow
    }
    
    color = emotion_colors.get(emotion, (255, 255, 255))
    
    # Draw rectangle around face
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    # Prepare text
    label = f"{emotion.upper()}: {confidence:.1f}%"
    
    # Get text size for background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    
    # Draw background rectangle for text
    cv2.rectangle(
        frame,
        (x, y - text_height - 10),
        (x + text_width, y),
        color,
        -1
    )
    
    # Draw text
    cv2.putText(
        frame,
        label,
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),  # Black text
        2
    )

def main():
    parser = argparse.ArgumentParser(description='Real-time emotion detection using webcam')
    parser.add_argument('--model', type=str, default='../../models/best_model_fer2013.pth',
                        help='Path to the trained model (default: FER2013 model)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index (default: 0)')
    parser.add_argument('--no-face-detection', action='store_true',
                        help='Skip face detection and use full frame')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        print("Available models:")
        if os.path.exists('../../models/best_model_jaffe.pth'):
            print("  - ../../models/best_model_jaffe.pth (JAFFE)")
        if os.path.exists('../../models/best_model_fer2013.pth'):
            print("  - ../../models/best_model_fer2013.pth (FER2013)")
        print("\nPlease train the model first by running:")
        print("  cd ../training && python train_fer2013.py")
        return
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model, class_names, input_size = load_model(args.model, device)
    print(f"Model loaded successfully! (Input size: {input_size}x{input_size})")
    
    # Load face detector (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return
    
    print("\n" + "="*60)
    print("REAL-TIME EMOTION DETECTION")
    print("="*60)
    print("\nInstructions:")
    print("  - Look at the camera")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("\nStarting camera...")
    print("="*60 + "\n")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        frame_count += 1
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if not args.no_face_detection:
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]
                
                # Predict emotion every 5 frames to reduce lag
                if frame_count % 5 == 0:
                    face_tensor = preprocess_face(face_roi, input_size)
                    emotion, confidence = predict_emotion(model, face_tensor, class_names, device)
                    
                    # Store for display
                    if not hasattr(main, 'current_emotion'):
                        main.current_emotion = {}
                    main.current_emotion[(x, y, w, h)] = (emotion, confidence)
                
                # Draw label using stored emotion
                if hasattr(main, 'current_emotion') and (x, y, w, h) in main.current_emotion:
                    emotion, confidence = main.current_emotion[(x, y, w, h)]
                    draw_emotion_label(frame, (x, y, w, h), emotion, confidence)
            
            # Display number of faces detected
            cv2.putText(
                frame,
                f"Faces: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        else:
            # Use full frame for prediction
            if frame_count % 5 == 0:
                face_tensor = preprocess_face(frame, input_size)
                emotion, confidence = predict_emotion(model, face_tensor, class_names, device)
                main.current_emotion = (emotion, confidence)
            
            if hasattr(main, 'current_emotion'):
                emotion, confidence = main.current_emotion
                label = f"{emotion.upper()}: {confidence:.1f}%"
                cv2.putText(
                    frame,
                    label,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )
        
        # Display FPS
        cv2.putText(
            frame,
            f"Press 'q' to quit",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1
        )
        
        # Show frame
        cv2.imshow('Emotion Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            # Save current frame
            filename = f'emotion_capture_{saved_count}.jpg'
            cv2.imwrite(filename, frame)
            saved_count += 1
            print(f"Saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nCamera closed.")

if __name__ == "__main__":
    main()
