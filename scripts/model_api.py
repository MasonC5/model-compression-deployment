# model_api.py
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import io
import os

from scripts.train_and_quantize import SmallCNN

app = FastAPI()
templates = Jinja2Templates(directory="scripts/templates")

# Load the distilled model
model = SmallCNN()
model.load_state_dict(torch.load("distilled_model.pth", map_location=torch.device('cpu')))
model.eval()

# CIFAR-10 preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        class_name = classes[predicted.item()]

    return {"prediction": class_name}

@app.get("/download/baseline")
async def download_baseline():
    return FileResponse("baseline_model.pth", media_type='application/octet-stream', filename="baseline_model.pth")

@app.get("/download/distilled")
async def download_distilled():
    return FileResponse("distilled_model.pth", media_type='application/octet-stream', filename="distilled_model.pth")
