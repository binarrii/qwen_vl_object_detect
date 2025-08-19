import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Union

import requests
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from rich.pretty import pprint

from config import config
from utils.image import b64_to_pil
from vl.detection import QwenVLDetection


@asynccontextmanager
async def lifespan(_: FastAPI):
    asyncio.create_task(post_start_task())
    yield


async def post_start_task():
    await asyncio.sleep(1)
    print(f"\n{'#' * 30} {'#' * 30} {'#' * 30}\n")
    pprint(config)
    print(f"\n{'#' * 30} {'#' * 30} {'#' * 30}\n")


app = FastAPI(lifespan=lifespan)

app.mount("/images", StaticFiles(directory="output_images"), name="output_images")


det = QwenVLDetection()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/detect")
async def detect(image: Union[str, UploadFile], target: str = ""):
    if isinstance(image, str):
        if image.startswith("http"):
            response = requests.get(image)
            image = Image.open(BytesIO(response.content))
        elif image.startswith("data:image/"):
            image = b64_to_pil(image)
    elif image.content_type.startswith("image/"):
        contents = await image.read()
        image = Image.open(BytesIO(contents))
    else:
        return JSONResponse(content={"error": "invalid parameters"})

    _prompt = det.expand_default_caption_prompt(target)
    _, json_boxes, bboxes_only, image_name = det.detect(image=image, usr_prompt=_prompt)
    _output_url_prefix = os.getenv("OUTPUT_HTTP_PREFIX", config.output_http_prefix)
    image_url = f"{_output_url_prefix}/images/{image_name}"
    print(f"output_image_url: {image_url}")
    return JSONResponse(
        content={
            "json_output": json_boxes,
            "bboxes_only": bboxes_only,
            "image": image_url,
        }
    )


@app.post("/v1/chat/completions")
async def chatCompletions(req: Request):
    req_json = await req.json()
    print(req_json["messages"][0])
    print(req_json["messages"][1]["content"][0])
    print(f"{req_json['messages'][1]['content'][1]['image_url']['url'][:30]}...")

    messages = req_json.get("messages", [])

    system_prompt, caption_prompt, image = None, None, None
    for message in messages:
        if message.get("role", None) == "system":
            system_prompt = message.get("content", None)
            continue
        for content in message.get("content", []):
            if content.get("type", None) == "text":
                caption_prompt = content.get("text", None)
            if content.get("type", None) == "image_url":
                image_url = content.get("image_url", None)
                image_b64 = image_url.get("url", None) if image_url else None
                image = b64_to_pil(image_b64)

    model = req_json.get("model", None)

    json_output, _, _, _ = det.detect(
        image=image,
        usr_prompt=caption_prompt,
        sys_prompt=system_prompt,
        model=model,
        max_tokens=req_json.get("max_tokens", None),
        temperature=req_json.get("temperature", None),
        top_p=req_json.get("top_p", None),
        frequency_penalty=req_json.get("frequency_penalty", None),
        presence_penalty=req_json.get("presence_penalty", None),
    )

    raw_json = req_json.get("raw_json", False)
    output_content = json_output if raw_json else f"```json\n{json_output}\n```"

    return {
        "id": f"chatcmpl-{str(uuid.uuid4()).replace('-', '')}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": output_content,
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
    }
