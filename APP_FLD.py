import cv2
import torch
import torchvision.transforms.functional as TF
import streamlit as st
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO  
from file_loader import load_model as download_model



class depthwiseseperableconv2d(nn.Module):
  def __init__(self,input_channels,output_channels,kernel_size,**kwargs):
    super(depthwiseseperableconv2d,self).__init__()


# Depthwise convolution: Applies a separate convolutional filter to each input channel
# 'groups=input_channels' means each channel is convolved independently (depthwise).
    self.depthwise=nn.Conv2d(input_channels,input_channels,kernel_size,groups=input_channels,bias=False,**kwargs)

# Pointwise convolution: Combines the output of the depthwise layer, 
# by applying a 1x1 convolution to reduce or increase the number of channels.

    self.pointwise=nn.Conv2d(input_channels,output_channels,1,bias=False)


  def forward(self,x):
    x=self.depthwise(x)
    x=self.pointwise(x)
    return x

class entryblock(nn.Module):
  def __init__(self):
    super(entryblock,self).__init__()
# First convolution block: 1 input channel to 32 channels, followed by BatchNorm and LeakyReLU
    self.conv1= nn.Sequential(
        nn.Conv2d(1,32,3,padding=1,bias=False),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.2)
    )
# Second convolution block: 32 input channels to 64 channels
    self.conv2=nn.Sequential(
        nn.Conv2d(32,64,3,padding=1,bias=False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2)
    )
 # Third block (Residual + Direct Path)
    self.conv3_residual=nn.Sequential(
        depthwiseseperableconv2d(64,64,3,padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2),
        depthwiseseperableconv2d(64,128,3,padding=1),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(3,stride=2,padding=1)
    )

# Direct path: 1x1 convolution to match the output of residual (input 64 channels, output 128 channels), and downsample using stride=2

    self.conv3direct=nn.Sequential(
        nn.Conv2d(64,128,1,stride=2),
        nn.BatchNorm2d(128)
    )
# Fourth block (Residual + Direct Path)
    self.conv4_residual=nn.Sequential(
        depthwiseseperableconv2d(128,128,3,padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2),
        depthwiseseperableconv2d(128,256,3,padding=1),
        nn.BatchNorm2d(256),
        nn.MaxPool2d(3,stride=2,padding=1)
    )
# Direct path: 1x1 convolution to match the output of residual (input 128 channels, output 256 channels), and downsample using stride=2
    self.conv4direct=nn.Sequential(
        nn.Conv2d(128,256,1,stride=2),
        nn.BatchNorm2d(256)
    )

# Fifth block (Residual + Direct Path)

    self.conv5_residual=nn.Sequential(
        depthwiseseperableconv2d(256,256,3,padding=1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2),
        depthwiseseperableconv2d(256,728,3,padding=1),
        nn.BatchNorm2d(728),
        nn.MaxPool2d(3,stride=2,padding=1)
    )
    
# Direct path: 1x1 convolution to match the output of residual (input 256 channels, output 728 channels), and downsample using stride=2
    self.conv5direct=nn.Sequential(
        nn.Conv2d(256,728,1,stride=2),
        nn.BatchNorm2d(728)
    )

  def forward(self,x):
    x=self.conv1(x)
    x=self.conv2(x)

    residual=self.conv3_residual(x)
    direct=self.conv3direct(x)
    x=residual+direct

    residual=self.conv4_residual(x)
    direct=self.conv4direct(x)
    x=residual+direct

    residual=self.conv5_residual(x)
    direct=self.conv5direct(x)
    x=residual+direct

    return x

class middleblock(nn.Module):
  def __init__(self):
    super(middleblock,self).__init__()

    self.conv1=nn.Sequential(
        nn.LeakyReLU(0.2),
        depthwiseseperableconv2d(728,728,3,padding=1),
        nn.BatchNorm2d(728),
    )

    self.conv2=nn.Sequential(
        nn.LeakyReLU(0.2),
        depthwiseseperableconv2d(728,728,3,padding=1),
        nn.BatchNorm2d(728),
    )
    self.conv3=nn.Sequential(
        nn.LeakyReLU(0.2),
        depthwiseseperableconv2d(728,728,3,padding=1),
        nn.BatchNorm2d(728),
    )

  def forward(self,x):
    residual=self.conv1(x)
    residual=self.conv2(residual)
    residual=self.conv3(residual)

    return x+residual
  
class middleiterativeblock(nn.Module):
  def __init__(self,numblocks):
    super(middleiterativeblock,self).__init__()
    
    #By using *, you are unpacking the list created by the list comprehension ([middleblock() for _ in range(numblocks)])
    #so that each element of the list is passed as an individual argument to nn.Sequential().

    self.block=nn.Sequential(*[middleblock() for _ in range(numblocks)])

  def forward(self,x):
    return self.block(x)

class exitblock(nn.Module):
  def __init__(self):
    super(exitblock,self).__init__()

# Residual path: Depthwise separable convolutions followed by BatchNorm, LeakyReLU, and max pooling
    self.residualconv= nn.Sequential(
        nn.LeakyReLU(0.2),
        depthwiseseperableconv2d(728,728,3,padding=1),
        nn.BatchNorm2d(728),
        nn.LeakyReLU(0.2),
        depthwiseseperableconv2d(728,1024,3,padding=1),
        nn.BatchNorm2d(1024),
        nn.MaxPool2d(3,stride=2,padding=1)
    )

# Direct path: 1x1 convolution to match the number of channels and downsample (stride=2)
    self.direct=nn.Sequential(
        nn.Conv2d(728,1024,1,stride=2),
        nn.BatchNorm2d(1024)
    )
# Convolution block after residual and direct paths are combined
    self.conv=nn.Sequential(
        depthwiseseperableconv2d(1024,1536,3,padding=1),
        nn.BatchNorm2d(1536),
        nn.LeakyReLU(0.2),
        depthwiseseperableconv2d(1536,2048,3,padding=1),
        nn.BatchNorm2d(2048),
        nn.LeakyReLU(0.2),
    )
# Dropout layer for regularization to prevent overfitting    
    self.dropout =nn.Dropout(0.3)

# Global average pooling to reduce each channel to a single value (1x1 output per channel)    
    self.globlavgpool=nn.AdaptiveAvgPool2d((1,1))


  def forward(self,x):
      direct =self.direct(x)
      residual=self.residualconv(x)
      x=residual+direct
      x=self.conv(x)
      x=self.globlavgpool(x)
      x=self.dropout(x)
      return x

class Network(nn.Module):
  def __init__(self,num_of_blocks):
    super(Network,self).__init__()
# Entry block: Initial layers to extract low-level features
    self.entry_block =entryblock()
# Middle block: A stack of `num_of_blocks` residual blocks (iterative layers) for deep feature extraction
    self.middle_block=middleiterativeblock(num_of_blocks)
# Exit block: Final layers to extract high-level features and reduce spatial dimensions    
    self.exit_block=exitblock()

# Linear layer that outputs 136 values (68 keypoint pairs)

    self.fc=nn.Linear(2048,136)

  def forward(self,x):
    x=self.entry_block(x)
    x=self.middle_block(x)
    x=self.exit_block(x)
    x=x.view(x.size(0),-1)
    x=self.fc(x)

    return x


def load_model(filepath, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filepath, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model

# Step 1: Ensure model file is downloaded if it doesn’t exist
model_path = download_model()  # This will download 'best_model (3).pt' if it's not present

# Step 2: Load model with PyTorch if the download was successful
if model_path:
    model = Network(8)  
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
        landmarks = landmarks.view(-1, 2)
        landmarks = (landmarks + 0.5)
        landmarks = landmarks.numpy()
        
        for i, (x, y) in enumerate(landmarks, 1):
            try:
                cv2.circle(image, (int((x * width) + left), int((y * height) + top)), 2, [255, 0, 0], -1)
            except Exception as e:
                print(f"Error drawing landmark: {e}")
    
    return image

@torch.no_grad()
def inference_image(image):
    if image is None:
        print("No image provided!")
        return None

    image_np = np.array(image)
    
    st.image(image_np, caption='Original Image')
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    outputs = []

    if len(faces) == 0:
        st.warning("No faces detected.")

    for (x, y, w, h) in faces:
        crop_img = gray[y: y + h, x: x + w]
        preprocessed_image = preprocess_image(crop_img)
        
        if torch.cuda.is_available():
            landmarks_predictions = model(preprocessed_image.cuda())
        else:
            landmarks_predictions = model(preprocessed_image)


        outputs.append((landmarks_predictions.cpu(), (x, y, h, w)))

    output_image = draw_landmarks_on_faces(image_bgr, outputs)
    
    return output_image

def main():
    st.markdown(
        """
        <h1 style="text-align: center; color: #90EE90;">Facial Landmark Detection</h1>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
    """
    <p style="font-size:18px;">Upload an image, enter an image URL, or take a photo using your camera:</p>
    """,
    unsafe_allow_html=True
)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image_url = st.text_input("Or enter Image URL:")
    camera_image = st.camera_input("Or take a photo")
    image = None

    if image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            st.error(f"⚠️ Error loading image from URL: {e}")

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"⚠️ Error loading uploaded image: {e}")

    if camera_image is not None:
        try:
            image = Image.open(camera_image)
        except Exception as e:
            st.error(f"⚠️ Error loading camera image: {e}")

    if image is not None:
        image_np = np.array(image)
        output_image = inference_image(image_np)
        if output_image is not None:
            st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), caption="Landmarks Detected")

    st.markdown(
        """
        <hr>
        <p style="text-align: center; font-size: 14px;">🚀 Developed by <b>Siddharth Hoonka</b></p>
        """,
        unsafe_allow_html=True
    )
if __name__ == "__main__":
    main()
