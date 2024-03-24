from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile

from src.evaluate import (
    read_image,
    preprocess,
    predict,
    get_prediction_model,
    #download_model_from_azure_blob,
)

vit_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Utilisation des fonctions
    # storage_connection_string = "BlobEndpoint=https://facemask2477918544.blob.core.windows.net/;QueueEndpoint=https://facemask2477918544.queue.core.windows.net/;FileEndpoint=https://facemask2477918544.file.core.windows.net/;TableEndpoint=https://facemask2477918544.table.core.windows.net/;SharedAccessSignature=sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2024-12-05T06:39:11Z&st=2023-12-04T22:39:11Z&spr=https&sig=1COs7OvSKGEROzBdAg9C0j22Dp3OKAzj5vH1DzADuY0%3D"
    # container_name = "facemask"
    # blob_name = "best_model1.pt"
    # local_file_name = "modele_local.pt"
    # download_model_from_azure_blob(
    #     storage_connection_string, container_name, blob_name, local_file_name
    # )
    global vit_model  # Utiliser le mot-clé global pour modifier la variable globale
    vit_model = get_prediction_model("models/best_model1.pt")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/api/predict")
async def predict_image(file: UploadFile):
    # Lire l'image du fichier temporaire
    contents = await file.read()
    # Read image
    image = read_image(contents)
    # Preprocess image
    image = preprocess(image)
    # Predict
    predictions = predict(image, vit_model)
    return predictions
