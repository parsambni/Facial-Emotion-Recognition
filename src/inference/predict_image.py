import torch
import torch.nn.functional as F
import sys
import os
# Add parent directory to path to import models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.fer2013_model import ImprovedFER_CNN
from models.jaffe_model import JAFFE_CNN
from PIL import Image
import torchvision.transforms as transforms
import argparse

def load_model(model_path, device):
    """Load the trained model"""
    # Determine model type from checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get('class_names', ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
    
    # Check if it's FER2013 or JAFFE based on input size in state dict or file name
    if 'fer2013' in model_path.lower():
        model = ImprovedFER_CNN(num_classes=7).to(device)
        input_size = 48
    else:
        model = JAFFE_CNN(num_classes=7).to(device)
        input_size = 128
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, class_names, input_size

def preprocess_image(image_path, input_size=48):
    """Preprocess image for inference"""
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def predict_emotion(model, image_tensor, class_names, device):
    """Predict emotion from image tensor"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    emotion = class_names[predicted.item()]
    conf = confidence.item() * 100
    
    # Get all probabilities
    all_probs = probabilities[0].cpu().numpy()
    
    return emotion, conf, all_probs

def main():
    parser = argparse.ArgumentParser(description='Predict emotion from an image using trained model')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to the trained model')
    parser.add_argument('--show-all', action='store_true',
                        help='Show probabilities for all emotions')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found!")
        return
    
    # Check available models
    available_models = {}
    if os.path.exists('../../models/best_model_jaffe.pth'):
        available_models['1'] = ('../../models/best_model_jaffe.pth', 'JAFFE (128x128, trained on 213 images)')
    if os.path.exists('../../models/best_model_fer2013.pth'):
        available_models['2'] = ('../../models/best_model_fer2013.pth', 'FER2013 (48x48, trained on 35,887 images)')
    
    if not available_models:
        print("Error: No trained models found!")
        print("\nPlease train a model first by running:")
        print("  cd ../training && python train_jaffe.py")
        print("  or")
        print("  cd ../training && python train_fer2013.py")
        return
    
    # If model not specified, ask user to choose
    if args.model is None:
        print("\n" + "="*60)
        print("SELECT MODEL")
        print("="*60)
        for key, (path, desc) in available_models.items():
            print(f"  [{key}] {desc}")
        print("="*60)
        
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice not in available_models:
            print(f"Error: Invalid choice '{choice}'!")
            return
        
        args.model = available_models[choice][0]
        print(f"\n✓ Selected: {available_models[choice][1]}\n")
    
    # Check if selected model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        return
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load model
    print("Loading model...")
    model, class_names, input_size = load_model(args.model, device)
    print(f"Model loaded successfully! (Input size: {input_size}x{input_size})\n")
    
    # Preprocess image
    print(f"Processing image: {args.image_path}")
    image_tensor = preprocess_image(args.image_path, input_size)
    
    # Predict
    emotion, confidence, all_probs = predict_emotion(model, image_tensor, class_names, device)
    
    # Display results
    print("\n" + "="*60)
    print("EMOTION PREDICTION RESULT")
    print("="*60)
    print(f"\n🎭 Predicted Emotion: {emotion.upper()}")
    print(f"📊 Confidence: {confidence:.2f}%\n")
    
    if args.show_all:
        print("All emotion probabilities:")
        print("-" * 40)
        # Sort by probability
        emotion_probs = [(class_names[i], all_probs[i] * 100) for i in range(len(class_names))]
        emotion_probs.sort(key=lambda x: x[1], reverse=True)
        
        for emo, prob in emotion_probs:
            bar = "█" * int(prob / 2)  # Scale to 50 chars max
            print(f"  {emo:10s}: {prob:5.2f}% {bar}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
