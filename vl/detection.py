import json
import uuid

from PIL import Image, ImageDraw
from transformers import AutoImageProcessor

from config import config
from utils.boxes import parse_boxes
from vl.caption import VLCaption


_SYSTEM_PROMPT = """
- 你是个图片分割美工, 擅长分离切割图片中的主要角色或物体, 且角色之间最好不要相连
- 角色可能是 人物, 2D动画角色, 3D动画角色, 动物, 拟人角色, 占据主要画面的物品, 比如汽车, 机械, 食物等
- 输出JSON格式如下:
  [{"bbox_2d": [653, 527, 1058, 1020], "label": "lady"}, {"bbox_2d": [1135, 228, 1607, 1051], "label": "main character"}]
"""


def _prefix(model: str):
    return model if model.startswith("./") else f"./{model}"


class QwenVLDetection(VLCaption):
    models = config.models
    processors = {
        m.model_id: AutoImageProcessor.from_pretrained(
            _prefix(m.model_id), trust_remote_code=True
        )
        for m in models
    }

    def _check_model(self, model_id: str):
        return next((m for m in self.models if m.model_id == model_id), self.models[0])

    def detect(
        self,
        image,
        usr_prompt: str,
        sys_prompt: str = None,
        bbox_selection: str = "all",
        score_threshold: float = 0.6,
        merge_boxes: bool = False,
        model=None,
        max_tokens=2048,
        temperature=0.5,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    ):
        image = Image.open(image) if isinstance(image, str) else image
        _model = self._check_model(model)
        _sys_prompt = sys_prompt or _SYSTEM_PROMPT

        print(f"model: {_model}")

        processor = self.processors[_model.model_id]
        inputs = processor(images=[image], return_tensors="pt").to("cpu")
        try:
            input_h = inputs["image_grid_thw"][0][1] * 14
            input_w = inputs["image_grid_thw"][0][2] * 14
        except:  # noqa: E722
            # pixel_values = inputs['pixel_values']
            # print(pixel_values.shape)
            # n, c, h, w = pixel_values.shape
            input_h, input_w = image.height, image.width

        print(f"input_image_o: (h={image.height}, w={image.width})")
        print(f"input_image_p: (h={input_h}, w={input_w})")

        output_text = self.caption(
            image_in=image,
            protocol="openai",
            custom_model=_model.model_id,
            model=None,
            ollama_model=None,
            system_prompt=_sys_prompt,
            caption_prompt=usr_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            base_url=_model.base_url,
            api_key=_model.api_key,
        )
        items = parse_boxes(
            output_text,
            image.width,
            image.height,
            input_w,
            input_h,
            score_threshold,
        )

        selection = bbox_selection.strip().lower()
        boxes = items
        if selection != "all" and selection:
            idxs = []
            for part in selection.replace(",", " ").split():
                try:
                    idxs.append(int(part))
                except Exception:
                    continue
            boxes = [boxes[i] for i in idxs if 0 <= i < len(boxes)]

        if merge_boxes and boxes:
            x1 = min(b["bbox"][0] for b in boxes)
            y1 = min(b["bbox"][1] for b in boxes)
            x2 = max(b["bbox"][2] for b in boxes)
            y2 = max(b["bbox"][3] for b in boxes)
            score = max(b["score"] for b in boxes)
            label = boxes[0].get("label", usr_prompt)
            boxes = [{"bbox": [x1, y1, x2, y2], "score": score, "label": label}]

        json_boxes = [
            {"bbox_2d": b["bbox"], "label": b.get("label", usr_prompt)} for b in boxes
        ]
        json_output = json.dumps(json_boxes, ensure_ascii=False)
        bboxes_only = [b["bbox"] for b in boxes]
        print(f"json_output: {json_output}")
        print(f"bboxes_only: {bboxes_only}")

        # Draw each bounding box on the image
        new_image = image.convert("RGB")
        draw = ImageDraw.Draw(new_image)
        for box in boxes:
            bbox = box["bbox"]
            if processor.do_center_crop:
                if image.width > image.height:
                    bbox[0] += (image.width - image.height) / 2
                    bbox[2] += (image.width - image.height) / 2
                if image.height > image.width:
                    bbox[1] += (image.height - image.width) / 2
                    bbox[3] += (image.height - image.width) / 2
            draw.rectangle(bbox, outline="red", width=2)
            label = box.get("label", usr_prompt)
            draw.text((bbox[0], bbox[1]), label, fill="red")

        image_name = f"{str(uuid.uuid4()).replace('-', '')}.png"
        image_file = f"output_images/{image_name}"
        new_image.save(image_file)
        print(f"output_image_file: {image_file}")

        return (json_output, json_boxes, bboxes_only, image_name)
