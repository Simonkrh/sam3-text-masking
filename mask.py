from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


COLORS = [(0, 210, 100), (30, 150, 255), (255, 120, 30), (210, 70, 220), (240, 210, 20)]


def save_result(
    image: Image.Image,
    masks: np.ndarray,
    scores: list,
    boxes: list,
    output: Path,
    metadata: dict,
) -> dict:
    """Save every matching instance, their union, an overlay, and scores."""
    image = image.convert("RGB")
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.size == 0:
        masks = np.zeros((0, image.height, image.width), dtype=bool)
    if masks.shape != (len(scores), image.height, image.width) or len(boxes) != len(
        scores
    ):
        raise ValueError(
            "Mask dimensions and detection counts must match the image and scores."
        )
    output.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    union = np.zeros((image.height, image.width), dtype=bool)
    detections = []
    for index, (mask, score, box) in enumerate(zip(masks, scores, boxes)):
        filename = f"mask_{index:03d}.png"
        binary = Image.fromarray(mask.astype(np.uint8) * 255)
        binary.save(output / filename)
        union |= mask
        color = COLORS[index % len(COLORS)]
        tint = Image.blend(overlay, Image.new("RGB", image.size, color), 0.45)
        overlay = Image.composite(tint, overlay, binary)
        detections.append(
            {
                "mask": filename,
                "score": float(score),
                "box_xyxy": [float(value) for value in box],
                "area_pixels": int(mask.sum()),
            }
        )
    draw = ImageDraw.Draw(overlay)
    for index, detection in enumerate(detections):
        box = detection["box_xyxy"]
        color = COLORS[index % len(COLORS)]
        draw.rectangle(box, outline=color, width=3)
        draw.text(
            (max(0, box[0]), max(0, box[1] - 14)),
            f"{index:03d}: {detection['score']:.2f}",
            fill=color,
            stroke_width=1,
            stroke_fill="black",
        )
    Image.fromarray(union.astype(np.uint8) * 255).save(output / "combined_mask.png")
    overlay.save(output / "overlay.jpg", quality=95)
    result = {
        **metadata,
        "image_size": [image.width, image.height],
        "instance_count": len(detections),
        "detections": detections,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        "--images",
        nargs="+",
        required=True,
        type=Path,
        help="One or more selected images (shell globs work).",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        required=True,
        help="Text describing what to segment; repeat to compare prompts.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum instance confidence (default: 0.5).",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument(
        "--model",
        default="facebook/sam3",
        help="Hugging Face model ID or a local Transformers model directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or not 0 <= args.mask_threshold <= 1:
        parser.error("Thresholds must be between 0 and 1.")
    if any(not prompt.strip() for prompt in args.prompt):
        parser.error("Text prompts must not be empty.")
    for path in args.image:
        if not path.is_file():
            parser.error(f"Image not found: {path}")

    import torch
    from transformers import Sam3Model, Sam3Processor

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable. Check your GPU environment or use --device cpu."
        )
    dtype_name = args.dtype
    if dtype_name == "auto":
        dtype_name = (
            ("bfloat16" if torch.cuda.is_bf16_supported() else "float16")
            if device == "cuda"
            else "float32"
        )
    dtype = getattr(torch, dtype_name)
    started = time.perf_counter()
    print(f"Loading {args.model} on {device} ({dtype_name})...", flush=True)
    try:
        processor = Sam3Processor.from_pretrained(args.model)
        model = Sam3Model.from_pretrained(args.model, dtype=dtype).to(device).eval()
    except OSError as error:
        parser.exit(
            1,
            f"Could not load SAM 3: {error}\n\n"
            "For facebook/sam3, request model access at https://huggingface.co/facebook/sam3 "
            "and log in with: uv run hf auth login\n",
        )
    if device == "cuda":
        torch.cuda.synchronize()
    load_seconds = round(time.perf_counter() - started, 3)
    run_dir = args.output_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "model": args.model,
        "device": device,
        "dtype": dtype_name,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "torch_version": torch.__version__,
        "model_load_seconds": load_seconds,
        "results": [],
    }
    for image_index, path in enumerate(args.image):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        for prompt_index, prompt in enumerate(args.prompt):
            tick = time.perf_counter()
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                device
            )

            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
            with torch.inference_mode():
                outputs = model(**inputs)
                result = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=args.threshold,
                    mask_threshold=args.mask_threshold,
                    target_sizes=inputs["original_sizes"].tolist(),
                )[0]
            if device == "cuda":
                torch.cuda.synchronize()
            inference_seconds = round(time.perf_counter() - tick, 3)
            slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", prompt).strip("_")[:60] or "prompt"
            folder = (
                run_dir
                / f"{image_index:02d}_{path.stem}"
                / f"{prompt_index:02d}_{slug}"
            )
            saved = save_result(
                image,
                result["masks"].detach().cpu().numpy(),
                result["scores"].detach().float().cpu().tolist(),
                result["boxes"].detach().float().cpu().tolist(),
                folder,
                {
                    "image": str(path.resolve()),
                    "prompt": prompt,
                    "model": args.model,
                    "device": device,
                    "dtype": dtype_name,
                    "threshold": args.threshold,
                    "mask_threshold": args.mask_threshold,
                    "inference_seconds": inference_seconds,
                },
            )
            report["results"].append(
                {
                    "directory": str(folder.relative_to(run_dir)),
                    "image": str(path),
                    "prompt": prompt,
                    "instance_count": saved["instance_count"],
                    "inference_seconds": inference_seconds,
                }
            )
            report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            (run_dir / "run.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"{path.name} / {prompt!r}: {saved['instance_count']} instances, "
                f"{inference_seconds}s -> {folder}",
                flush=True,
            )
    print(f"Results: {run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
