from typing import Annotated, Literal

from fastapi import FastAPI, File, UploadFile, Form, HTTPException

import filetype

from PIL import Image
import io

app = FastAPI()

@app.get("/")
async def welcome():
    return "Hello User!"

@app.post('/predict')
async def predict(model: Annotated[Literal['convnext', 'maxvit'], Form(description="Opsi model deep learning yang digunakan 'convnext' & 'maxvit'")], image: Annotated[UploadFile, File(description="Hanya menerima satu file foto. Jika mengirim lebih dari 1 file maka file terakhir yang akan diambil")]):
    image_bytes = await image.read()

    file_type = filetype.guess_mime(image_bytes)

    if file_type not in {'image/jpeg', 'image/png', 'image/webp'}:
        raise HTTPException(status_code=422, detail="File harus berupa jpg, jpeg, webp, dan png")

    img = Image.open(io.BytesIO(image_bytes))

    if img.mode != "RGB":
        img = img.convert("RGB")

    width, height = img.size

    return {
        "status": "success",
        "model": 'ConvNeXt' if model == 'convnext' else 'MaxViT',
        "width": width,
        "height": height,
        "format": img.format,
        "type": image.content_type,
        "file type": file_type,
    }
