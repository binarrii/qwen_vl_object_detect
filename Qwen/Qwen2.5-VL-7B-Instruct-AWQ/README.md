---
license: apache-2.0
language:
- en
pipeline_tag: image-text-to-text
tags:
- multimodal
library_name: transformers
base_model:
- Qwen/Qwen2.5-VL-7B-Instruct
---

# Qwen2.5-VL-7B-Instruct-AWQ
<a href="https://chat.qwenlm.ai/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/%F0%9F%92%9C%EF%B8%8F%20Qwen%20Chat%20-536af5" style="display: inline-block; vertical-align: middle;"/>
</a>

## Introduction

In the past five months since Qwen2-VL’s release, numerous developers have built new models on the Qwen2-VL vision-language models, providing us with valuable feedback. During this period, we focused on building more useful vision-language models. Today, we are excited to introduce the latest addition to the Qwen family: Qwen2.5-VL.

#### Key Enhancements:
* **Understand things visually**: Qwen2.5-VL is not only proficient in recognizing common objects such as flowers, birds, fish, and insects, but it is highly capable of analyzing texts, charts, icons, graphics, and layouts within images.

* **Being agentic**: Qwen2.5-VL directly plays as a visual agent that can reason and dynamically direct tools, which is capable of computer use and phone use.

* **Understanding long videos and capturing events**: Qwen2.5-VL can comprehend videos of over 1 hour, and this time it has a new ability of cpaturing event by pinpointing the relevant video segments.

* **Capable of visual localization in different formats**: Qwen2.5-VL can accurately localize objects in an image by generating bounding boxes or points, and it can provide stable JSON outputs for coordinates and attributes.

* **Generating structured outputs**: for data like scans of invoices, forms, tables, etc. Qwen2.5-VL supports structured outputs of their contents, benefiting usages in finance, commerce, etc.


#### Model Architecture Updates:

* **Dynamic Resolution and Frame Rate Training for Video Understanding**:

We extend dynamic resolution to the temporal dimension by adopting dynamic FPS sampling, enabling the model to comprehend videos at various sampling rates. Accordingly, we update mRoPE in the time dimension with IDs and absolute time alignment, enabling the model to learn temporal sequence and speed, and ultimately acquire the ability to pinpoint specific moments.

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-VL/qwen2.5vl_arc.jpeg" width="80%"/>
<p>


* **Streamlined and Efficient Vision Encoder**

We enhance both training and inference speeds by strategically implementing window attention into the ViT. The ViT architecture is further optimized with SwiGLU and RMSNorm, aligning it with the structure of the Qwen2.5 LLM.


We have three models with 3, 7 and 72 billion parameters. This repo contains the instruction-tuned 7B Qwen2.5-VL model with AWQ. For more information, visit our [Blog](https://qwenlm.github.io/blog/qwen2.5-vl/) and [GitHub](https://github.com/QwenLM/Qwen2.5-VL).



## Evaluation



## Requirements
The code of Qwen2.5-VL has been in the latest Hugging face transformers and we advise you to build from source with command:
```
pip install git+https://github.com/huggingface/transformers accelerate
```
or you might encounter the following error:
```
KeyError: 'qwen2_5_vl'
```


## Quickstart

Below, we provide simple examples to show how to use Qwen2.5-VL with 🤖 ModelScope and 🤗 Transformers.

The code of Qwen2.5-VL has been in the latest Hugging face transformers and we advise you to build from source with command:
```
pip install git+https://github.com/huggingface/transformers accelerate
```
or you might encounter the following error:
```
KeyError: 'qwen2_5_vl'
```


We offer a toolkit to help you handle various types of visual input more conveniently, as if you were using an API. This includes base64, URLs, and interleaved images and videos. You can install it using the following command:

```bash
# It's highly recommanded to use `[decord]` feature for faster video loading.
pip install qwen-vl-utils[decord]==0.0.8
```

If you are not using Linux, you might not be able to install `decord` from PyPI. In that case, you can use `pip install qwen-vl-utils` which will fall back to using torchvision for video processing. However, you can still [install decord from source](https://github.com/dmlc/decord?tab=readme-ov-file#install-from-source) to get decord used when loading video.

### Using 🤗  Transformers to Chat

Here we show a code snippet to show you how to use the chat model with `transformers` and `qwen_vl_utils`:

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct-AWQ", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct-AWQ")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct-AWQ", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```


### 🤖 ModelScope
We strongly advise users especially those in mainland China to use ModelScope. `snapshot_download` can help you solve issues concerning downloading checkpoints.


### More Usage Tips

For input images, we support local files, base64, and URLs. For videos, we currently only support local files.

```python
# You can directly insert a local file path, a URL, or a base64-encoded image into the position where you want in the text.
## Local file path
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///path/to/your/image.jpg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
## Image URL
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "http://path/to/your/image.jpg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
## Base64 encoded image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "data:image;base64,/9j/..."},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
```
#### Image Resolution for performance boost

The model supports a wide range of resolution inputs. By default, it uses the native resolution for input, but higher resolutions can enhance performance at the cost of more computation. Users can set the minimum and maximum number of pixels to achieve an optimal configuration for their needs, such as a token count range of 256-1280, to balance speed and memory usage.

```python
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct-AWQ", min_pixels=min_pixels, max_pixels=max_pixels
)
```

Besides, We provide two methods for fine-grained control over the image size input to the model:

1. Define min_pixels and max_pixels: Images will be resized to maintain their aspect ratio within the range of min_pixels and max_pixels.

2. Specify exact dimensions: Directly set `resized_height` and `resized_width`. These values will be rounded to the nearest multiple of 28.

```python
# min_pixels and max_pixels
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "file:///path/to/your/image.jpg",
                "resized_height": 280,
                "resized_width": 420,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
# resized_height and resized_width
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "file:///path/to/your/image.jpg",
                "min_pixels": 50176,
                "max_pixels": 50176,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
```

### Processing Long Texts

The current `config.json` is set for context length up to 32,768 tokens.
To handle extensive inputs exceeding 32,768 tokens, we utilize [YaRN](https://arxiv.org/abs/2309.00071), a technique for enhancing model length extrapolation, ensuring optimal performance on lengthy texts.

For supported frameworks, you could add the following to `config.json` to enable YaRN:

{
	...,
    "type": "yarn",
    "mrope_section": [
        16,
        24,
        24
    ],
    "factor": 4,
    "original_max_position_embeddings": 32768
}

However, it should be noted that this method has a significant impact on the performance of temporal and spatial localization tasks, and is therefore not recommended for use.

At the same time, for long video inputs, since MRoPE itself is more economical with ids, the max_position_embeddings can be directly modified to a larger value, such as 64k.


### Benchmark
#### Performance of Quantized Models
This section reports the generation performance of quantized models (including GPTQ and AWQ) of the Qwen2.5-VL series. Specifically, we report:

- MMMU_VAL (Accuracy)
- DocVQA_VAL (Accuracy)
- MMBench_DEV_EN (Accuracy)
- MathVista_MINI (Accuracy)

We use [VLMEvalkit](https://github.com/open-compass/VLMEvalKit) to evaluate all models.

| Model Size | Quantization | MMMU_VAL | DocVQA_VAL | MMBench_EDV_EN | MathVista_MINI  |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL-72B-Instruct | BF16<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)[🤖](https://modelscope.cn/models/qwen/Qwen2.5-VL-72B-Instruct)) | 70.0 | 96.1 | 88.2 | 75.3 |
|  | AWQ<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ)[🤖](https://modelscope.cn/models/qwen/Qwen2.5-VL-72B-Instruct-AWQ)) | 69.1 | 96.0 | 87.9 | 73.8 |
| Qwen2.5-VL-7B-Instruct | BF16<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)[🤖](https://modelscope.cn/models/qwen/Qwen2.5-VL-7B-Instruct)) | 58.4 | 94.9 | 84.1 | 67.9 |
|  | AWQ<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ)[🤖](https://modelscope.cn/models/qwen/Qwen2.5-VL-7B-Instruct-AWQ)) | 55.6 | 94.6 | 84.2 | 64.7 |
| Qwen2.5-VL-3B-Instruct | BF16<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)[🤖](https://modelscope.cn/models/qwen/Qwen2.5-VL-3B-Instruct)) | 51.7 | 93.0 | 79.8 | 61.4 |
|  | AWQ<br><sup>([🤗](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ)[🤖](https://modelscope.cn/models/qwen/Qwen2.5-VL-3B-Instruct-AWQ)) | 49.1 | 91.8 | 78.0 | 58.8 |



## Citation

If you find our work helpful, feel free to give us a cite.

```
@misc{qwen2.5-VL,
    title = {Qwen2.5-VL},
    url = {https://qwenlm.github.io/blog/qwen2.5-vl/},
    author = {Qwen Team},
    month = {January},
    year = {2025}
}

@article{Qwen2VL,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Fan, Yang and Dang, Kai and Du, Mengfei and Ren, Xuancheng and Men, Rui and Liu, Dayiheng and Zhou, Chang and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}

@article{Qwen-VL,
  title={Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond},
  author={Bai, Jinze and Bai, Shuai and Yang, Shusheng and Wang, Shijie and Tan, Sinan and Wang, Peng and Lin, Junyang and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2308.12966},
  year={2023}
}
```