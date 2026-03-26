import os
import argparse
import glob
import base64
import re
import torch
import requests
from io import BytesIO
from PIL import Image
from qwen_vl_utils import process_vision_info
from clip_interrogator import Config, Interrogator
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)

try:
    import ollama
except ImportError:
    pass


class VisualDescriptionGenerator:
    def __init__(self, args):
        self.args = args
        self.device = self._get_device()

        self.base_prompt = """
Please provide a detailed visual description of this image.
Include key objects, their spatial relationships,
notable visual features, and any observable actions or events.
Respond in clear, structured English paragraphs.
""".strip()

        if self.args.model_type == 'api':
            self._validate_api_credentials()
        elif self.args.model_type == 'blip2':
            self.processor, self.model = self._init_BLIP2()
        elif self.args.model_type == 'llava':
            self.client = self._init_LLaVa()
        elif self.args.model_type == 'qwen2-vl':
            self.processor, self.model = self._init_Qwen2_VL()
        elif self.args.model_type == 'clip-interrogator':
            self.ci = self._init_CLIP_Interrogator()
        else:
            raise ValueError(f"Unsupported model type: {self.args.model_type}")

    def _get_device(self):
        if self.args.model_type in ['api', 'llava']:
            return None
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _validate_api_credentials(self):
        if not hasattr(self.args, 'api_key') or not self.args.api_key:
            raise ValueError("API key is required for API mode.")

    def _init_BLIP2(self):
        model_map = {
            'flan-t5': "Salesforce/blip2-flan-t5-xl",
            'opt': "Salesforce/blip2-opt-2.7b"
        }
        model_name = model_map[self.args.blip2_version]
        processor = AutoProcessor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        return processor, model

    def _init_LLaVa(self):
        return ollama.Client(host=f"http://localhost:{self.args.llava_port}")

    def _init_Qwen2_VL(self):
        model_map = {
            '2b': "Qwen/Qwen2-VL-2B-Instruct",
            '7b': "Qwen/Qwen2-VL-7B-Instruct",
            '72b': "Qwen/Qwen2-VL-72B-Instruct"
        }

        if self.args.use_quantization:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quant_config = None

        model_name = model_map[self.args.qwen2vl_version]
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            quantization_config=quant_config,
            trust_remote_code=True
        ).to(self.device)
        return processor, model

    def _init_CLIP_Interrogator(self):
        config = Config()
        if hasattr(self.args, 'clip_model') and self.args.clip_model:
            config.clip_model_name = self.args.clip_model
        return Interrogator(config)

    @staticmethod
    def _image_to_base64(image_path):
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.convert("RGB").save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def _sanitize_prefix(text):
        text = text.strip().lower()
        text = text.replace("/", "-")
        text = re.sub(r"[^a-z0-9._-]+", "-", text)
        text = re.sub(r"-+", "-", text).strip("-")
        return text

    def _get_model_prefix(self):
        if getattr(self.args, "prefix", None):
            return self._sanitize_prefix(self.args.prefix)

        if self.args.model_type == "clip-interrogator":
            return "clip-interrogator"
        if self.args.model_type == "qwen2-vl":
            return f"qwen2vl{self.args.qwen2vl_version}"
        if self.args.model_type == "blip2":
            return f"blip2-{self.args.blip2_version}"
        if self.args.model_type == "llava":
            return f"llava{self.args.llava_version}"
        if self.args.model_type == "api":
            return "api"

        return self._sanitize_prefix(self.args.model_type)

    def _get_output_path(self, image_path):
        previous_prefixes = self.args.previous_prefixes or []
        current_prefix = self._get_model_prefix()
        all_prefixes = previous_prefixes + [current_prefix]
        suffix = ".".join([self._sanitize_prefix(x) for x in all_prefixes])
        return f"{os.path.splitext(image_path)[0]}.{suffix}.txt"

    def _get_previous_output_path(self, image_path, prefixes):
        suffix = ".".join([self._sanitize_prefix(x) for x in prefixes])
        return f"{os.path.splitext(image_path)[0]}.{suffix}.txt"

    @staticmethod
    def _read_txt_content(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        text = re.sub(r"^\[Description\]\s*", "", text).strip()
        return text

    def _load_previous_descriptions(self, image_path):
        previous_prefixes = self.args.previous_prefixes or []
        descriptions = []

        for i in range(len(previous_prefixes)):
            prefix_chain = previous_prefixes[: i + 1]
            prev_txt_path = self._get_previous_output_path(image_path, prefix_chain)
            if not os.path.exists(prev_txt_path):
                raise FileNotFoundError(
                    f"Expected previous-stage txt not found: {prev_txt_path}"
                )
            descriptions.append(self._read_txt_content(prev_txt_path))

        return descriptions

    def _build_prompt(self, image_path):
        previous_prefixes = self.args.previous_prefixes or []

        if len(previous_prefixes) == 0:
            return self.base_prompt

        previous_descriptions = self._load_previous_descriptions(image_path)
        history_blocks = []
        for idx, desc in enumerate(previous_descriptions, start=1):
            history_blocks.append(f"Previous description {idx}:\n{desc}")

        history_text = "\n\n".join(history_blocks)

        if len(previous_prefixes) == 1:
            stage_instruction = """
You are given an image and one existing visual description.
Do NOT repeat the previous description.
Only add missing visual details that are clearly supported by the image.
Focus on objects not mentioned yet, attributes, spatial relations,
background elements, visible text, symbols, charts, or subtle actions.
Return only the additional information in clear English paragraphs.
""".strip()
        else:
            stage_instruction = """
You are given an image and multiple previous visual descriptions.
Do NOT repeat existing content unless absolutely necessary.
Your goal is to further enrich the description by adding missing,
image-grounded visual details and improving coverage.
Focus on complementary objects, attributes, relations, scene layout,
fine-grained cues, text in the image, diagrams, and overlooked elements.
Return only the newly added information in clear English paragraphs.
""".strip()

        prompt = f"""{stage_instruction}

{history_text}
"""
        print(f"\n[Debug] Constructed prompt for {image_path}:\n{prompt}\n")

        return prompt

    def generate_description(self, image_path, prompt):
        if self.args.model_type == 'api':
            return self._generate_API_description(image_path, prompt)
        elif self.args.model_type == 'blip2':
            return self._generate_BLIP2_description(image_path, prompt)
        elif self.args.model_type == 'llava':
            return self._generate_LLaVa_description(image_path, prompt)
        elif self.args.model_type == 'qwen2-vl':
            return self._generate_Qwen2_VL_description(image_path, prompt)
        elif self.args.model_type == 'clip-interrogator':
            return self._generate_CLIP_Interrogator_description(image_path, prompt)
        else:
            raise ValueError(f"Unsupported model type: {self.args.model_type}")

    def _generate_API_description(self, image_path, prompt):
        base64_image = self._image_to_base64(image_path)
        headers = {"Authorization": f"Bearer {self.args.api_key}"}
        data = {
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }]
        }
        response = requests.post(self.args.api_url, headers=headers, json=data, timeout=300)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()

    def _generate_BLIP2_description(self, image_path, prompt):
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_length=300,
            num_beams=5,
            temperature=0.7
        )
        return self.processor.decode(generated_ids[0], skip_special_tokens=True).strip()

    def _generate_LLaVa_description(self, image_path, prompt):
        base64_image = self._image_to_base64(image_path)
        response = self.client.chat(
            model=f"llava:{self.args.llava_version}",
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [base64_image]
            }]
        )
        return response['message']['content'].strip()

    def _generate_Qwen2_VL_description(self, image_path, prompt):
        base64_image = self._image_to_base64(image_path)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": f"data:image;base64,{base64_image}"}
            ]
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=1500,
            temperature=0.7
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        return response
        # return self.processor.decode(generated_ids[0], skip_special_tokens=True).strip()

    def _generate_CLIP_Interrogator_description(self, image_path, prompt):
        # clip-interrogator 不直接使用 prompt，但保留统一接口
        try:
            with Image.open(image_path).convert('RGB') as img:
                return self.ci.interrogate(img)
        except Exception as e:
            print(f"CLIP processing error: {str(e)}")
            return None

    def process(self):
        if os.path.isfile(self.args.input):
            self._process_single_image(self.args.input)
        elif os.path.isdir(self.args.input):
            self._process_batch_images(self.args.input)
        else:
            raise ValueError(f"Invalid input path: {self.args.input}")

    def _process_single_image(self, image_path):
        try:
            output_path = self._get_output_path(image_path)
            if os.path.exists(output_path):
                print(f"Output already exists, skip: {output_path}")
                return

            prompt = self._build_prompt(image_path)

            print(f"\nAnalyzing image: {image_path}")
            print(f"Output path: {output_path}")
            description = self.generate_description(image_path, prompt)

            if description is None or not str(description).strip():
                print(f"Empty description for {image_path}, skip saving.")
                return

            print(f"\n[Description]\n{description}")

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"[Description]\n{description}\n")

            print(f"Saved to {output_path}")

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

    def _process_batch_images(self, folder_path):
        extensions = ['*.jpg', '*.jpeg', '*.png']
        for ext in extensions:
            for image_path in glob.glob(os.path.join(folder_path, '**', ext), recursive=True):
                self._process_single_image(image_path)


def main():
    parser = argparse.ArgumentParser(
        description="CoE-style multimodel visual description generation tool"
    )
    parser.add_argument('--input', required=True, help="Image path or folder path")
    parser.add_argument(
        '--prefix',
        default=None,
        help="Custom prefix for current model output. If omitted, derive from model name."
    )
    parser.add_argument(
        '--previous_prefixes',
        type=str,
        default="",
        help=(
            "Comma-separated prefixes of previous CoE stages. "
            "Example: blip2-flan-t5 or blip2-flan-t5,qwen2vl2b"
        )
    )

    subparsers = parser.add_subparsers(dest='model_type', required=True)

    api_parser = subparsers.add_parser('api')
    api_parser.add_argument('--api_key', required=True, help="API key")
    api_parser.add_argument('--api_url', default="https://api.openai.com/v1/chat/completions")

    blip_parser = subparsers.add_parser('blip2')
    blip_parser.add_argument('--blip2_version', choices=['flan-t5', 'opt'], required=True)

    llava_parser = subparsers.add_parser('llava')
    llava_parser.add_argument('--llava_version', choices=['7b', '13b', '34b'], required=True)
    llava_parser.add_argument('--llava_port', type=int, default=11434)

    qwen_parser = subparsers.add_parser('qwen2-vl')
    qwen_parser.add_argument('--qwen2vl_version', choices=['2b', '7b', '72b'], required=True)
    qwen_parser.add_argument('--use_quantization', action='store_true')

    clip_parser = subparsers.add_parser('clip-interrogator')
    clip_parser.add_argument('--clip_model', default="ViT-L-14/openai", help="CLIP model type")

    args = parser.parse_args()

    if args.previous_prefixes:
        args.previous_prefixes = [x.strip() for x in args.previous_prefixes.split(",") if x.strip()]
    else:
        args.previous_prefixes = []

    try:
        generator = VisualDescriptionGenerator(args)
        generator.process()
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()