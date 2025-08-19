import os

import yaml
from pydantic import BaseModel


class VLModel(BaseModel):
    model_id: str
    base_url: str
    api_key: str


class Config(BaseModel):
    models: list[VLModel]
    output_http_prefix: str


_conf_path = f"{os.path.dirname(__file__)}/config.yaml"

with open(_conf_path, "r") as f:
    config = Config(**yaml.safe_load(f))
