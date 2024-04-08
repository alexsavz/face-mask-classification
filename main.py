from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Importing necessary configurations and functions
import src.data.configs as configs
from src.models.evaluate import (
    read_image,
    preprocess,
    predict,
    get_prediction_model,
)

# Importing YAML configuration file
config = configs.import_yaml_config("./configs/config.yaml")
URL_1 = config["path"]["origins"]["local_1"]
URL_2 = config["path"]["origins"]["local_2"]
URL_3 = config["path"]["origins"]["vercel_app"]

# Global variable to store the model
vit_model = None

# Async context manager for application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vit_model  # le mot-clé global modifie la variable globale
    vit_model = get_prediction_model("pymodels/best_model1.pt")
    yield

# Creating the FastAPI application instance with lifespan management
app = FastAPI(lifespan=lifespan)

# Cross-Origin Resource Sharing config
origins = [URL_1, URL_2, URL_3]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint to verify the application is running
@app.get("/")
def read_root():
    return {"Message": "Hello world"}

# Endpoint to receive an image and obtain predictions
@app.post("/api/predict")
async def predict_image(file: UploadFile):
    # Reading the image from the temporary file
    contents = await file.read()
    # Reading image
    image = read_image(contents)
    # Preprocessing image
    image = preprocess(image)
    # Making prediction
    predictions = predict(image, vit_model)
    return predictions
