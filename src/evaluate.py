"""Model inference and helper codes"""

import os
from io import BytesIO

import torch
import torch.nn as nn

#from azure.storage.blob import BlobServiceClient

from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# def download_model_from_azure_blob(storage_connection_string, container_name, blob_name, local_file_name):
#     blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
#     blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

#     with open(local_file_name, "wb") as download_file:
#         download_file.write(blob_client.download_blob().readall())



def get_prediction_model(local_file_name):
    # Load the ML model
    # Activation des calculs sur GPU
    # device = torch.device("cuda") if torch.cuda.is_available else "cpu"
    device = "cpu"

    weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    vit_model = vit_b_16(weights=weights)

    # Nombre de features en entrée de la dernière couche du modèle ViT
    num_features = vit_model.heads.head.in_features

    # Gèle de toutes les couches du modèle
    for param in vit_model.parameters():
        param.requires_grad = False

    # Modifions la dernière couche du modèle
    vit_model.heads.head = nn.Linear(num_features, 1)
    vit_model = vit_model.to(device)

    checkpoint = torch.load(local_file_name, map_location=torch.device("cpu"))
    vit_model.load_state_dict(checkpoint["model_state_dict"])
    vit_model.eval()
    return vit_model


def read_image(contents):
    image = Image.open(BytesIO(contents)).convert("RGB")
    return image


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


def predict(input_image, model):
    if torch.cuda.is_available():
        input_image = input_image.to("cuda")
        model.to("cuda")
    with torch.no_grad():
        output = model(input_image)
    # Utiliser une fonction sigmoide sur les logits
    pred = torch.sigmoid(output)
    # Arrondir les prédiction à 0 ou à 1
    pred = torch.round(pred)

    print(f"Résultat de la prédiction : {pred.item()}")
    return int(pred.item())
