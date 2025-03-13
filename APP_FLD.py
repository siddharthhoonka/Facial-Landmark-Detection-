import cv2
import torch
import numpy as np
from torchvision import transforms
from model import XceptionNet  # Ensure this matches your model architecture

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
def load_model(filepath):
    model = XceptionNet()
    model = model.to(device)
    
    checkpoint = torch.load(filepath, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError("Invalid checkpoint format: Missing 'state_dict'")
    
    model.eval()
    return model

# Preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Post-processing function to adjust output to original image size
def postprocess_landmarks(landmarks, original_shape):
    h, w = original_shape
    landmarks = landmarks.view(-1, 2).detach().cpu().numpy()
    landmarks[:, 0] *= w
    landmarks[:, 1] *= h
    return landmarks

# Inference function
def inference_image(image_np, model):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_np)
    print(f"Input shape: {preprocessed_image.shape}")

    # Ensure tensor is valid
    if torch.isnan(preprocessed_image).any() or torch.isinf(preprocessed_image).any():
        raise ValueError("Invalid tensor values detected in input")

    # Predict landmarks
    with torch.no_grad():
        landmarks_predictions = model(preprocessed_image)
    
    # Post-process landmarks to original image scale
    landmarks = postprocess_landmarks(landmarks_predictions, image_np.shape[:2])
    return landmarks

# Function to draw landmarks on the image
def draw_landmarks(image, landmarks):
    for (x, y) in landmarks:
        cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)
    return image

# Main function
def main():
    # Load the model
    model = load_model("model_checkpoint.pth")

    # Read the input image
    image_path = "input_image.jpg"
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    
    # Convert image to RGB
    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run inference
    landmarks = inference_image(image_np, model)

    # Draw landmarks on the image
    output_image = draw_landmarks(image_np, landmarks)

    # Save and display the output
    output_path = "output_image.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()
