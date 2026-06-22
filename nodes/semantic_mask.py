"""Text-guided masks via CLIPSeg (same CLIP family as SDXL prompts — size/shape agnostic)."""
from __future__ import annotations

import logging
import os
from typing import Iterable

import cv2
import folder_paths
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(folder_paths.models_dir, "zit_upscale", "clipseg-rd64-refined")
CLIPSEG_REPO = "CIDAS/clipseg-rd64-refined"

PRESETS = [
    "breasts",
    "waist",
    "belly",
    "hips",
    "thighs",
    "torso",
    "face",
    "hair",
    "clothes",
    "background",
    "custom",
]

PRESET_PROMPTS = {
    "breasts": "woman's breasts, chest, bust, cleavage, boobs",
    "waist": "woman's waist, midsection, narrow waist, love handles",
    "belly": "woman's stomach, belly, abdomen, navel",
    "hips": "woman's hips, buttocks, rear, ass",
    "thighs": "woman's thighs, upper legs",
    "torso": "woman's torso, upper body, chest and stomach",
    "face": "woman's face, head",
    "hair": "woman's hair, hairstyle",
    "clothes": "woman's clothes, dress, skirt, fabric, outfit, wrap",
    "background": "background, wall, room, posters behind the woman",
    "custom": "",
}

LOCK_PROMPTS = {
    "face": "woman's face, head, eyes, mouth",
    "hair": "woman's hair",
    "clothes": "clothes, dress, skirt, fabric, outfit, wrap",
    "background": "background, wall, posters, room",
}

_clipseg_processor = None
_clipseg_model = None


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    arr = (255.0 * image.detach().cpu().numpy()).clip(0, 255).astype(np.uint8)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return Image.fromarray(arr)


def _mask_to_tensor(mask: np.ndarray, batch: int) -> torch.Tensor:
    m = mask.astype(np.float32)
    if m.max() > 1.0:
        m = m / 255.0
    t = torch.from_numpy(m)[None, ...]
    if batch > 1:
        t = t.repeat(batch, 1, 1)
    return t


def _feather(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    k = radius * 2 + 1
    return cv2.GaussianBlur(mask, (k, k), 0)


def _grow(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    k = pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel, iterations=1)


def _combine_or(masks: Iterable[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    for m in masks:
        out = np.maximum(out, m.astype(np.float32))
    return out


def _load_clipseg():
    global _clipseg_processor, _clipseg_model
    if _clipseg_model is not None:
        return _clipseg_processor, _clipseg_model

    try:
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    except ImportError as exc:
        raise RuntimeError(
            "ZiTSemanticMask needs transformers. Install: pip install transformers"
        ) from exc

    import comfy.model_management as mm

    if os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
        logger.info("ZiT-Upscale: loading CLIPSeg from %s", MODEL_DIR)
        _clipseg_processor = CLIPSegProcessor.from_pretrained(MODEL_DIR)
        _clipseg_model = CLIPSegForImageSegmentation.from_pretrained(MODEL_DIR)
    else:
        logger.info("ZiT-Upscale: downloading CLIPSeg %s", CLIPSEG_REPO)
        os.makedirs(MODEL_DIR, exist_ok=True)
        _clipseg_processor = CLIPSegProcessor.from_pretrained(CLIPSEG_REPO)
        _clipseg_model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_REPO)
        _clipseg_processor.save_pretrained(MODEL_DIR)
        _clipseg_model.save_pretrained(MODEL_DIR)

    device = mm.get_torch_device()
    _clipseg_model.to(device)
    _clipseg_model.eval()
    return _clipseg_processor, _clipseg_model


def _clipseg_mask(pil: Image.Image, prompts: list[str], threshold: float) -> np.ndarray:
    if not prompts:
        return np.zeros((pil.height, pil.width), dtype=np.float32)

    processor, model = _load_clipseg()
    import comfy.model_management as mm

    device = mm.get_torch_device()
    w, h = pil.size

    inputs = processor(text=prompts, images=[pil] * len(prompts), return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    logits = outputs.logits
    if logits.ndim == 4:
        logits = logits.squeeze(1)
    logits = F.interpolate(
        logits.unsqueeze(1),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    prob = torch.sigmoid(logits).amax(dim=0).clamp(0, 1)
    mask = (prob > threshold).float().cpu().numpy()
    return mask


def _resolve_edit_prompt(preset: str, prompt_extra: str) -> list[str]:
    base = PRESET_PROMPTS.get(preset, "").strip()
    extra = (prompt_extra or "").strip()
    if preset == "custom":
        text = extra or "woman's breasts, chest"
    elif extra:
        text = f"{base}, {extra}"
    else:
        text = base
    # CLIPSeg works better with 2–3 short phrases max
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return parts[:4] if parts else ["woman's breasts, chest"]


def build_semantic_mask(
    pil: Image.Image,
    preset: str,
    prompt_extra: str,
    threshold: float,
    expand: int,
    feather: int,
    lock_face: bool,
    lock_hair: bool,
    lock_clothes: bool,
    lock_background: bool,
) -> np.ndarray:
    h, w = pil.height, pil.width

    edit_prompts = _resolve_edit_prompt(preset, prompt_extra)
    edit = _clipseg_mask(pil, edit_prompts, threshold)

    locks = []
    if lock_face:
        locks.append(_clipseg_mask(pil, [LOCK_PROMPTS["face"]], max(0.25, threshold - 0.05)))
    if lock_hair:
        locks.append(_clipseg_mask(pil, [LOCK_PROMPTS["hair"]], max(0.25, threshold - 0.05)))
    if lock_clothes:
        locks.append(_clipseg_mask(pil, [LOCK_PROMPTS["clothes"]], max(0.28, threshold - 0.02)))
    if lock_background:
        locks.append(_clipseg_mask(pil, [LOCK_PROMPTS["background"]], max(0.25, threshold - 0.05)))

    if locks:
        locked = _combine_or(locks, (h, w))
        edit = np.clip(edit - locked, 0.0, 1.0)

    edit = (edit > 0.2).astype(np.float32)
    edit = _grow(edit, expand)
    edit = _feather(edit, feather)
    return np.clip(edit, 0.0, 1.0)


class ZiTSemanticMask:
    """CLIP text → mask. Works on any breast size / pose (not geometry hacks)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (PRESETS, {"default": "breasts"}),
                "prompt_extra": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "optional: pink bikini top, side view…",
                    },
                ),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.32,
                        "min": 0.10,
                        "max": 0.90,
                        "step": 0.01,
                        "tooltip": "Lower = bigger mask. Raise if zone too large.",
                    },
                ),
                "expand": ("INT", {"default": 12, "min": 0, "max": 64, "step": 1}),
                "feather": ("INT", {"default": 16, "min": 0, "max": 64, "step": 1}),
                "lock_face": ("BOOLEAN", {"default": True}),
                "lock_hair": ("BOOLEAN", {"default": True}),
                "lock_clothes": ("BOOLEAN", {"default": True}),
                "lock_background": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("edit_mask", "preview", "resolved_prompt")
    FUNCTION = "segment"
    CATEGORY = "ZiT-Upscale/Edit"

    def segment(
        self,
        image,
        preset,
        prompt_extra,
        threshold,
        expand,
        feather,
        lock_face,
        lock_hair,
        lock_clothes,
        lock_background,
    ):
        batch = image.shape[0]
        masks = []
        previews = []
        resolved = ", ".join(_resolve_edit_prompt(preset, prompt_extra))

        for i in range(batch):
            pil = _tensor_to_pil(image[i : i + 1])
            mask = build_semantic_mask(
                pil,
                preset,
                prompt_extra,
                threshold,
                expand,
                feather,
                lock_face,
                lock_hair,
                lock_clothes,
                lock_background,
            )
            if float(mask.max()) < 0.02:
                raise RuntimeError(
                    f"CLIPSeg mask empty for preset={preset!r}. "
                    "Lower threshold, change prompt_extra, or pick another preset."
                )
            masks.append(mask)
            rgb = np.array(pil.convert("RGB")).astype(np.float32) / 255.0
            red = np.zeros_like(rgb)
            red[..., 0] = 1.0
            alpha = mask[..., None] * 0.55
            previews.append(rgb * (1.0 - alpha) + red * alpha)

        return (
            _mask_to_tensor(np.stack(masks, axis=0), batch),
            torch.from_numpy(np.stack(previews, axis=0)).float(),
            resolved,
        )


NODE_CLASS_MAPPINGS = {
    "ZiTSemanticMask": ZiTSemanticMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZiTSemanticMask": "ZiT Semantic Mask (CLIP text → zone)",
}
