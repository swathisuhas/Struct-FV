import os
import json
import argparse
from typing import List, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from imagenet_info import return_classes


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_all_file_paths_from_filenames(directory: str) -> Tuple[List[str], List[int]]:
    """
    Expected layout:
        dataset_dir/
            t1/
                0.png
                1.png
                ...
                999.png
            t2/
                0.png
                ...
            t3/
                0.png
                ...

    Label is extracted from filename stem:
        0.png -> class 0
        999.png -> class 999
    """
    file_paths = []
    labels = []

    for root, _, files in os.walk(directory):
        for file in sorted(files):
            ext = os.path.splitext(file)[1].lower()
            if ext not in VALID_EXTS:
                continue

            stem = os.path.splitext(file)[0]
            if not stem.isdigit():
                continue

            label = int(stem)
            full_path = os.path.join(root, file)

            file_paths.append(full_path)
            labels.append(label)

    return file_paths, labels


def build_label_list(classnames: List[str]) -> str:
    return "\n".join([f"{i}: {name}" for i, name in enumerate(classnames)])


def build_prompt(classnames: List[str]) -> str:
    label_block = build_label_list(classnames)
    prompt = f"""
You are doing ImageNet-1K image classification.

Task:
Choose the 5 most likely labels from the numbered label list below for this image.

Rules:
- Output only valid JSON.
- Use exactly this format:
{{"indices": [i1, i2, i3, i4, i5], "labels": ["label1", "label2", "label3", "label4", "label5"]}}
- Return exactly 5 predictions.
- Rank them from most likely to least likely.
- Every index must be a valid label index from the list below.
- Every label must exactly match the label text corresponding to its index.
- Do not output any explanation.
- Do not output markdown.
- Do not output any extra text before or after the JSON.

ImageNet-1K labels:
{label_block}
""".strip()
    return prompt


def parse_prediction(text: str):
    text = text.strip()
    try:
        obj = json.loads(text)
        pred_indices = obj["indices"]
        pred_labels = obj["labels"]

        if not isinstance(pred_indices, list) or not isinstance(pred_labels, list):
            return None, None
        if len(pred_indices) != 5 or len(pred_labels) != 5:
            return None, None

        pred_indices = [int(x) for x in pred_indices]
        pred_labels = [str(x) for x in pred_labels]
        return pred_indices, pred_labels
    except Exception:
        return None, None


def classify_one_image(model, processor, image_path: str, prompt: str):
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs.pop("token_type_ids", None)

    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL ImageNet top-1/top-5 accuracy")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset root")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--max_images", type=int, default=-1)
    args = parser.parse_args()

    classnames = return_classes()
    prompt = build_prompt(classnames)

    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

    file_paths, labels = list_all_file_paths_from_filenames(args.dataset_dir)

    if args.max_images > 0:
        file_paths = file_paths[:args.max_images]
        labels = labels[:args.max_images]

    if len(file_paths) == 0:
        raise ValueError(f"No valid images found in {args.dataset_dir}")

    print(f"Found {len(file_paths)} images")

    top1_correct = 0
    top5_correct = 0
    total = 0
    parse_failures = 0

    for image_path, target in tqdm(list(zip(file_paths, labels)), total=len(file_paths)):
        raw_text = classify_one_image(model, processor, image_path, prompt)
        pred_indices, pred_labels = parse_prediction(raw_text)

        is_top1 = False
        is_top5 = False

        if pred_indices is not None and pred_labels is not None:
            valid = True

            for idx, lbl in zip(pred_indices, pred_labels):
                if idx < 0 or idx >= len(classnames):
                    valid = False
                    break
                if lbl != classnames[idx]:
                    valid = False
                    break

            if valid:
                is_top1 = (pred_indices[0] == target)
                is_top5 = (target in pred_indices)
            else:
                parse_failures += 1
        else:
            parse_failures += 1

        top1_correct += int(is_top1)
        top5_correct += int(is_top5)
        total += 1

    top1 = 100.0 * top1_correct / total
    top5 = 100.0 * top5_correct / total

    print(f"Total images: {total}")
    print(f"Parse failures: {parse_failures}")
    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")


if __name__ == "__main__":
    main()