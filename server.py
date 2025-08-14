import base64
from io import BytesIO
from typing import Union

import requests
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import env
from vl.detection import QwenVLDetection

app = FastAPI()

app.mount("/images", StaticFiles(directory="output_images"), name="output_images")


det = QwenVLDetection()


@app.post("/detect")
async def detect(image: Union[str, UploadFile], target: str = ""):
    if isinstance(image, str):
        if image.startswith("http"):
            response = requests.get(image)
            image = Image.open(BytesIO(response.content))
        elif image.startswith("data:image/"):
            image_data = image.split("data:image/")[1].split(";base64,")[1]
            image_data = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_data))
    elif image.content_type.startswith("image/"):
        contents = await image.read()
        image = Image.open(BytesIO(contents))
    else:
        return JSONResponse(content={"error": "invalid parameters"})

    json_output, bboxes_only, image_name = det.detect(image=image, target=target)
    image_url = f"{env.OUTPUT_HTTP_PREFIX}/images/{image_name}"
    return JSONResponse(
        content={
            "json_output": json_output,
            "bboxes_only": bboxes_only,
            "image": image_url,
        }
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
