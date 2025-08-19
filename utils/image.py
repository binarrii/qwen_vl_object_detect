import base64
import io
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torch import Tensor


def b64_to_pil(image: str):
    if image.startswith("data:image/"):
        image_data = image.split("data:image/")[1].split(";base64,")[1]
    else:
        image_data = image

    image_data = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_data))
    return image


def pil_to_b64(image: Image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return image_b64


def pil_to_tensor(image: Image):
    image_array = np.array(image)
    image_tensor = torch.from_numpy(image_array)
    if image_tensor.shape[-1] == 4:
        image_tensor = image_tensor[..., :3]
    image_tensor = image_tensor.permute(2, 0, 1)
    return image_tensor


def tensor_to_pil(tensor: Tensor):
    if tensor.ndim == 3 and tensor.shape[0] in {1, 3}:
        tensor = tensor.permute(1, 2, 0)
    elif tensor.ndim == 4 and tensor.shape[1] in {1, 3}:
        tensor = tensor.permute(0, 2, 3, 1)
        tensor = tensor[0]

    if tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)

    image_array = tensor.numpy().astype(np.uint8)
    image = Image.fromarray(image_array)
    return image
