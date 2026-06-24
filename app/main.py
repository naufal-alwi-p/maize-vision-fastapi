from typing import Annotated, Literal
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import filetype

from PIL import Image
import io
import torch

from utils import class_names
from model.response_model import InferenceResponse
from ml import load_convnext, load_maxvit, grad_cam_convnext, grad_cam_maxvit, inference_grad_cam

BASE_DIR = Path(__file__).resolve().parent
CONVNEXT_WEIGHTS_PATH = BASE_DIR.parent / "model_weights" / "convnext_weights.pth"
MAXVIT_WEIGHTS_PATH = BASE_DIR.parent / "model_weights" / "maxvit_weights.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cam_convnext = grad_cam_convnext(load_convnext(weight_path=CONVNEXT_WEIGHTS_PATH, num_classes=5, device=DEVICE))
cam_maxvit = grad_cam_maxvit(load_maxvit(weight_path=MAXVIT_WEIGHTS_PATH, num_classes=5, device=DEVICE))

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.get("/")
async def welcome():
    return "Hello User!"

@app.post('/predict')
async def predict(image: Annotated[UploadFile, File(description="Hanya menerima satu file foto. Jika mengirim lebih dari 1 file maka file terakhir yang akan diambil")], model: Annotated[Literal['convnext', 'maxvit'] | None, Form(description="Opsional, pilih model deep learning yang digunakan 'convnext' (default) & 'maxvit'")] = None):
    image_bytes = await image.read()

    file_type = filetype.guess_mime(image_bytes)

    if file_type not in {'image/jpeg', 'image/png', 'image/webp'}:
        raise HTTPException(status_code=422, detail="File harus berupa jpg, jpeg, webp, dan png")

    img = Image.open(io.BytesIO(image_bytes))

    if img.mode != "RGB":
        img = img.convert("RGB")

    result = inference_grad_cam(cam_maxvit if model == 'maxvit' else cam_convnext, img, DEVICE)

    return InferenceResponse(
        original_image=result.original_image,
        grad_cam_image=result.grad_cam_image,
        scores=result.scores,
        class_names=class_names,
    )
