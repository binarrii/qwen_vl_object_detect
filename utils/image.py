import base64
from io import BytesIO

from PIL import Image


def b64_to_pil(image: str):
    if image.startswith("data:image/"):
        image_data = image.split("data:image/")[1].split(";base64,")[1]
    else:
        image_data = image

    image_data = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_data))
    return image
