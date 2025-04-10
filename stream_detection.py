import os
import torch
import torchvision
from torchvision import models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
import cv2  # Added for video capture
import numpy as np

# Define the dataset path and classes
dataset_path = 'fruit'  # FIX ME

# Load classes from the dataset annotations
classes = ['__background__', 'cat', 'dog']
num_classes = len(classes)

# Function to load the trained model
def load_trained_model(model_path, num_classes):
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda:0')))
    model.eval()
    return model

# Function to perform inference
def predict(image, model, device):
    model.to(device)
    image = [image.to(device)]
    model.eval()
    with torch.no_grad():
        prediction = model(image)
    return prediction

# Function to visualize prediction
def visualize_prediction(image, prediction, threshold=0.4):
    boxes = prediction[0]['boxes']
    scores = prediction[0]['scores']
    labels = prediction[0]['labels']
    keep = scores >= threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    class_names = [classes[i] for i in labels]
    text = [(f"{name}: {score:.2f}") for name, score in zip(class_names, scores)]
    drawn_image = draw_bounding_boxes(image.mul(255).byte(), boxes, text, width=4)
    return drawn_image

# Device setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Preprocess image from path (unused in video, but preserved for structure)
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image)

# =================== VIDEO STREAM CODE STARTS HERE ===================

# Load the model
model_path = os.path.expanduser('cat_dog.pth')  # FIX ME
trained_model = load_trained_model(model_path, num_classes)

# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for default camera

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Transform for webcam frames
transform = transforms.Compose([
    transforms.ToTensor(),
])

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert BGR to RGB and PIL to tensor
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tensor = transform(image_pil)

    # Run prediction
    prediction = predict(image_tensor, trained_model, device)

    # Visualize prediction
    drawn = visualize_prediction(image_tensor, prediction)
    drawn_np = drawn.permute(1, 2, 0).cpu().numpy()

    # Convert RGB back to BGR for OpenCV display
    drawn_bgr = cv2.cvtColor(drawn_np, cv2.COLOR_RGB2BGR)

    # Show the result
    cv2.imshow('Object Detection', drawn_bgr)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# =================== VIDEO STREAM CODE ENDS HERE ===================
