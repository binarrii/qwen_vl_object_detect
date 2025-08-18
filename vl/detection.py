import json
import uuid

from PIL import Image, ImageDraw
from transformers import AutoProcessor

import env
from utils.boxes import parse_boxes
from vl.caption import VLCaption


_SYSTEM_PROMPT = """
- 你是个图片分割美工, 擅长分离切割图片中的主要角色或物体, 且角色之间最好不要相连
- 角色可能是 人物, 2D动画角色, 3D动画角色, 动物, 拟人角色, 占据主要画面的物品, 比如汽车, 机械, 食物等
- 输出JSON格式如下:
  [{"bbox_2d": [653, 527, 1058, 1020], "label": "lady"}, {"bbox_2d": [1135, 228, 1607, 1051], "label": "main character"}]
"""


class QwenVLDetection(VLCaption):
    model_id: str = env.VL_MODEL
    processor = AutoProcessor.from_pretrained(f"./{model_id}", trust_remote_code=True)
    models = ["Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"]

    def _check_model(self, model: str):
        return model if model in self.models else self.models[0]

    def detect(
        self,
        image,
        target: str,
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

        prompt = f"定位 `{target}` 并以JSON格式输出边界框 bbox 坐标和标签"
        messages = [
            {"role": "system", "content": _sys_prompt},
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, {"image": image}],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        ).to("cpu")
        input_h = inputs["image_grid_thw"][0][1] * 14
        input_w = inputs["image_grid_thw"][0][2] * 14

        output_text = self.caption(
            image_in=image,
            protocol="openai",
            custom_model=self.model_id,
            model=_model,
            ollama_model=None,
            system_prompt=_sys_prompt,
            caption_prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            base_url=env.VL_MODEL_BASE_URL,
            api_key=env.VL_MODEL_API_KEY,
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
            label = boxes[0].get("label", target)
            boxes = [{"bbox": [x1, y1, x2, y2], "score": score, "label": label}]

        json_boxes = [
            {"bbox_2d": b["bbox"], "label": b.get("label", target)} for b in boxes
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
            draw.rectangle(bbox, outline="red", width=2)
            label = box.get("label", target)
            draw.text((bbox[0], bbox[1]), label, fill="red")

        image_name = f"{str(uuid.uuid4()).replace('-', '')}.png"
        image_file = f"output_images/{image_name}"
        new_image.save(image_file)

        return (json_output, json_boxes, bboxes_only, image_name)
