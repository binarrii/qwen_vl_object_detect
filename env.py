import os

import dotenv

dotenv.load_dotenv(dotenv_path=f"{os.path.dirname(__file__)}/.env")

VL_MODEL = os.getenv("VL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
VL_MODEL_BASE_URL = os.getenv("VL_MODEL_BASE_URL", "").rstrip("/")
VL_MODEL_API_KEY = os.getenv("VL_MODEL_API_KEY", "")

OUTPUT_HTTP_PREFIX=os.getenv("OUTPUT_HTTP_PREFIX", "").rstrip("/")
