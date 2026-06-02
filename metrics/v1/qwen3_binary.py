import os
import json
import random
import argparse
from typing import List, Dict, Tuple, Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from imagenet_info import return_classes


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def extract_label_from_filename(path: str) -> Optional[int]:
    filename = os.path.basename(path)
    stem, _ = os.path.splitext(filename)
    if stem.isdigit():
        return int(stem)
    return None


def list_dataset_t_subfolders(directory: str) -> List[Dict]:
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
                1.png
                ...
                999.png
            t3/
                0.png
                1.png
                ...
                999.png

    Returns a list of dicts:
        {
            "path": ...,
            "label": int
        }
    """
    samples = []

    for root, _, files in os.walk(directory):
        for file in sorted(files):
            ext = os.path.splitext(file)[1].lower()
            if ext not in VALID_EXTS:
                continue

            full_path = os.path.join(root, file)
            label = extract_label_from_filename(full_path)
            if label is None:
                continue

            samples.append({
                "path": full_path,
                "label": label,
            })

    return samples


def group_indices_by_class(samples: List[Dict]) -> Dict[int, List[int]]:
    by_class: Dict[int, List[int]] = {}
    for i, sample in enumerate(samples):
        by_class.setdefault(sample["label"], []).append(i)
    return by_class


def build_negative_index_pool(samples: List[Dict], by_class: Dict[int, List[int]]) -> Dict[int, List[int]]:
    """
    For each class c, build a list of sample indices whose class != c.
    """
    all_indices = list(range(len(samples)))
    neg_pool: Dict[int, List[int]] = {}

    for cls in by_class.keys():
        cls_set = set(by_class[cls])
        neg_pool[cls] = [idx for idx in all_indices if idx not in cls_set]

    return neg_pool


def build_prompt(class_name: str) -> str:
    return f"""
You are doing binary visual classification.

Question:
Is this an image of a {class_name}?

Rules:
- Answer with exactly one word.
- Output only: yes
- or output only: no
- Do not explain.
- Do not add punctuation.
- Do not add any other text.
""".strip()


def parse_yes_no(text: str) -> Optional[bool]:
    """
    Returns:
        True  -> yes
        False -> no
        None  -> parse failure
    """
    t = text.strip().lower()

    if t == "yes":
        return True
    if t == "no":
        return False

    t = t.replace(".", "").replace("!", "").replace(":", "").strip()
    if t == "yes":
        return True
    if t == "no":
        return False

    first = t.split()[0] if t else ""
    if first == "yes":
        return True
    if first == "no":
        return False

    return None


def ask_yes_no(model, processor, image_path: str, prompt: str) -> str:
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
            max_new_tokens=4,
            do_sample=False,
            num_beams=1,
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
    parser = argparse.ArgumentParser(description="Qwen3-VL yes/no ImageNet binary metric")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset root")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--max_images", type=int, default=-1, help="Use only first N images")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for negative sampling")
    parser.add_argument("--save_failures", type=str, default="failures.json")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    classnames = return_classes()

    print("Loading dataset...")
    samples = list_dataset_t_subfolders(args.dataset_dir)
    if len(samples) == 0:
        raise ValueError(f"No valid images found in {args.dataset_dir}")

    if args.max_images > 0:
        samples = samples[:args.max_images]

    print(f"Found {len(samples)} images")

    by_class = group_indices_by_class(samples)
    neg_pool = build_negative_index_pool(samples, by_class)

    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

    positive_total = 0
    positive_correct = 0

    negative_total = 0
    negative_correct = 0

    parse_failures = 0
    total_questions = 0

    failures = []
    progress = tqdm(samples, total=len(samples))

    for sample in progress:
        image_path = sample["path"]
        label = sample["label"]
        class_name = classnames[label]
        prompt = build_prompt(class_name)

        # Positive question on the true image: should be YES
        raw_pos = ask_yes_no(model, processor, image_path, prompt)
        pred_pos = parse_yes_no(raw_pos)

        total_questions += 1
        positive_total += 1

        if pred_pos is None:
            parse_failures += 1
            failures.append({
                "type": "positive",
                "image": image_path,
                "class_id": label,
                "class_name": class_name,
                "expected": "yes",
                "prediction": None,
                "raw_output": raw_pos,
            })
        else:
            if pred_pos is True:
                positive_correct += 1
            else:
                failures.append({
                    "type": "positive",
                    "image": image_path,
                    "class_id": label,
                    "class_name": class_name,
                    "expected": "yes",
                    "prediction": "no",
                    "raw_output": raw_pos,
                })

        # Negative question: same class prompt, but on random image from a different class
        pool = neg_pool[label]
        if len(pool) == 0:
            raise ValueError(f"No negative pool available for class {label}")

        neg_idx = random.choice(pool)
        neg_image_path = samples[neg_idx]["path"]

        raw_neg = ask_yes_no(model, processor, neg_image_path, prompt)
        pred_neg = parse_yes_no(raw_neg)

        total_questions += 1
        negative_total += 1

        if pred_neg is None:
            parse_failures += 1
            failures.append({
                "type": "negative",
                "image": neg_image_path,
                "class_id": label,
                "class_name": class_name,
                "expected": "no",
                "prediction": None,
                "raw_output": raw_neg,
            })
        else:
            if pred_neg is False:
                negative_correct += 1
            else:
                failures.append({
                    "type": "negative",
                    "image": neg_image_path,
                    "class_id": label,
                    "class_name": class_name,
                    "expected": "no",
                    "prediction": "yes",
                    "raw_output": raw_neg,
                })

        answered = total_questions - parse_failures
        running_acc = 100.0 * (positive_correct + negative_correct) / total_questions
        progress.set_description(f"Acc {running_acc:.2f}% | Parsed {answered}/{total_questions}")

    total_correct = positive_correct + negative_correct
    overall_accuracy = 100.0 * total_correct / total_questions

    positive_accuracy = 100.0 * positive_correct / positive_total if positive_total > 0 else 0.0
    negative_accuracy = 100.0 * negative_correct / negative_total if negative_total > 0 else 0.0
    balanced_accuracy = 0.5 * (positive_accuracy + negative_accuracy)

    print()
    print(f"Images: {len(samples)}")
    print(f"Total questions: {total_questions}")
    print(f"Parse failures: {parse_failures}")
    print(f"Positive questions: {positive_total}")
    print(f"Negative questions: {negative_total}")
    print(f"Positive accuracy (should answer yes): {positive_accuracy:.2f}")
    print(f"Negative accuracy (should answer no): {negative_accuracy:.2f}")
    print(f"Overall accuracy: {overall_accuracy:.2f}")
    print(f"Balanced accuracy: {balanced_accuracy:.2f}")

    if args.save_failures:
        with open(args.save_failures, "w") as f:
            json.dump(failures, f, indent=2)

        print(f"Saved {len(failures)} failure cases to {args.save_failures}")


if __name__ == "__main__":
    main()