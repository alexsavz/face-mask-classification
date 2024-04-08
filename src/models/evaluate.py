"""Model inference and helper codes"""

import os
from io import BytesIO

import torch
import torch.nn as nn


from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from PIL import Image

# Set CUDA_VISIBLE_DEVICES to avoid GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Function to get the prediction model
def get_prediction_model(local_file_name):
    # Load the ML model
    device = "cpu" # Set device to CPU
    weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1 # Load pre-trained weights
    vit_model = vit_b_16(weights=weights) # Initialize ViT model with pre-trained weights

    # Number of input features to the final layer of the ViT model
    num_features = vit_model.heads.head.in_features

    # Freeze all layers of the model
    for param in vit_model.parameters():
        param.requires_grad = False

    # Modify the final layer of the model
    vit_model.heads.head = nn.Linear(num_features, 1) # Replace the final layer with a Linear layer
    vit_model = vit_model.to(device) # Move model to device (CPU)

    # Load model state from a general checkpoint for inference
    checkpoint = torch.load(local_file_name, map_location=torch.device("cpu"))
    vit_model.load_state_dict(checkpoint["model_state_dict"])
    vit_model.eval()
    return vit_model

# Function to read image from bytes
def read_image(contents):
    image = Image.open(BytesIO(contents)).convert("RGB")
    return image

# Function to preprocess input image
def preprocess(input_image):
    data_transform = T.Compose(
        [
            transforms.Resize(384, InterpolationMode.BICUBIC),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = data_transform(input_image)
    input_image = input_tensor.unsqueeze(0)
    return input_image

# Function to perform prediction
def predict(input_image, model):
    if torch.cuda.is_available():
        input_image = input_image.to("cuda")
        model.to("cuda")
    with torch.no_grad(): # Disabling gradient calculation
        output = model(input_image)
    # Apply sigmoid function to logits
    pred = torch.sigmoid(output)
    # Round predictions to 0 or 1
    pred = torch.round(pred)

    print(f"Résultat de la prédiction : {pred.item()}")
    return int(pred.item())
