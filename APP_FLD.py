import cv2
import torch
import torchvision.transforms.functional as TF
import streamlit as st
from PIL import Image
import numpy as np
import requests
from io import BytesIO  
from file_loader import load_model as download_model

# Load the model
model_path = download_model()
if model_path:
    model = XceptionNet(middle_block_n=6)  
    model = load_model(model_path, model)
    print("Model loaded and ready.")
else:
    print("Failed to download model.")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(image):
    image = TF.to_pil_image(image)
    image = TF.resize(image, (128, 128))
    image = TF.to_tensor(image)
    image = (image - image.min()) / (image.max() - image.min())
    image = (2 * image) - 1
    return image.unsqueeze(0)

def draw_landmarks_on_faces(image, faces_landmarks):
    image = image.copy()
    for landmarks, (left, top, height, width) in faces_landmarks:
        landmarks = landmarks.view(-1, 2).numpy()
        for x, y in landmarks:
            try:
                cv2.circle(image, (int((x * width) + left), int((y * height) + top)), 2, [0, 255, 255], -1)
            except Exception as e:
                print(f"Error drawing landmark: {e}")
    return image

@torch.no_grad()
def inference_image(image):
    image_np = np.array(image)
    st.image(image_np, caption='Original Image', use_column_width=True)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    outputs = []
    if len(faces) == 0:
        st.warning("No faces detected.")
    for (x, y, w, h) in faces:
        crop_img = gray[y: y + h, x: x + w]
        preprocessed_image = preprocess_image(crop_img)
        landmarks_predictions = model(preprocessed_image.cuda() if torch.cuda.is_available() else preprocessed_image)
        outputs.append((landmarks_predictions.cpu(), (x, y, h, w)))
    return draw_landmarks_on_faces(image_bgr, outputs)

def main():
    st.markdown(
        """
        <style>
            .title { text-align: center; font-size: 40px; font-weight: bold; color: #FFA500; }
            .subtext { text-align: center; font-size: 20px; color: #666; }
            .footer { font-size: 14px; text-align: center; margin-top: 50px; color: #999; }
            .stTextInput > div > div > input { border-radius: 8px; border: 1px solid #ccc; padding: 12px; font-size: 16px; }
            .stButton > button { background-color: #FFA500; color: white; border-radius: 6px; padding: 12px; font-size: 16px; }
            .stFileUploader > div > div { border-radius: 10px; border: 2px dashed #FFA500; padding: 12px; }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<p class='title'>Facial Landmark Detection</p>", unsafe_allow_html=True)
    st.markdown("<p class='subtext'>Upload an image, enter an image URL, or take a photo using your camera:</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], help="Upload an image for landmark detection")
    image_url = st.text_input("Or enter Image URL:")
    camera_image = st.camera_input("Or take a photo")
    
    image = None
    if image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            st.error(f"Error loading image from URL: {e}")
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Error loading uploaded image: {e}")
    if camera_image is not None:
        try:
            image = Image.open(camera_image)
        except Exception as e:
            st.error(f"Error loading camera image: {e}")
    
    if image is not None:
        image_np = np.array(image)
        output_image = inference_image(image_np)
        if output_image is not None:
            st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), caption="Landmarks Detected", use_column_width=True)
    
    st.markdown("<p class='footer'>Siddharth Hoonka</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
