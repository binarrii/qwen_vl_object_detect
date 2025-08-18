import base64
import io

from openai import OpenAI


class VLCaption:
    def expand_default_caption_prompt(self, target: str):
        return f"定位 `{target}` 并以JSON格式输出边界框 bbox 坐标和标签"

    def caption(
        self,
        image_in,
        protocol,
        custom_model,
        model,
        ollama_model,
        system_prompt,
        caption_prompt,
        max_tokens,
        temperature,
        top_p,
        frequency_penalty,
        presence_penalty,
        base_url,
        api_key,
    ):
        # image to base64, image is bwhc tensor
        # Convert tensor to PIL Image
        # if isinstance(image_in, torch.Tensor):
        #     pil_image = Image.fromarray(
        #         np.clip(255.0 * image_in.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        #     )
        # else:
        pil_image = image_in

        if protocol == "ollama":
            if not custom_model:
                custom_model = ollama_model
            return self._caption_ollama(
                pil_image,
                custom_model,
                system_prompt,
                caption_prompt,
                max_tokens,
                temperature,
                base_url,
            )
        # openai style api
        else:
            if not custom_model:
                custom_model = model
            return self._caption_openai(
                pil_image,
                custom_model,
                system_prompt,
                caption_prompt,
                max_tokens,
                temperature,
                top_p,
                frequency_penalty,
                presence_penalty,
                base_url,
                api_key,
            )

    def _caption_openai(
        self,
        pil_image,
        model,
        system_prompt,
        caption_prompt,
        max_tokens,
        temperature,
        top_p,
        frequency_penalty,
        presence_penalty,
        base_url,
        api_key,
    ):
        # Convert PIL Image to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": caption_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_str}"},
                        },
                    ],
                },
            ],
            timeout=30,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        if response.choices[0].message.content is None:
            raise ValueError("No content in response")

        caption = response.choices[0].message.content.strip()
        return caption
