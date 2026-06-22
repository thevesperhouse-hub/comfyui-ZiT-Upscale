"""Lock the FACE / HAIR / CLOTHES of an i2i edit back to the original (pixel-faithful).

Plain SDXL img2img reliably changes body shape (bigger breasts, etc.) but it
drifts the whole frame. This node pastes the ORIGINAL face / hair / (optionally)
clothes back over the edited result, using auto CLIPSeg masks — no manual
painting. So the body change is kept, but identity (head, hair) is byte-identical.

Put it right after VAEDecode and feed it (original image, edited image).

Credits: CLIPSeg by CIDAS.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

try:
    from morpho_core import composite_preserve
except ImportError:
    from .morpho_core import composite_preserve

# semantic_mask is registered under its bare name in sys.modules by __init__.py
import semantic_mask as _sm


def _img_to_rgb_u8(img_t: torch.Tensor) -> np.ndarray:
    arr = img_t.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def _region_mask(ref_rgb: np.ndarray, keys: list[str], threshold: float, expand: int, feather: int) -> np.ndarray:
    from PIL import Image

    if not keys:
        return np.zeros(ref_rgb.shape[:2], dtype=np.float32)
    pil = Image.fromarray(ref_rgb)
    masks = [_sm._clipseg_mask(pil, [_sm.LOCK_PROMPTS[k]], threshold) for k in keys]
    m = _sm._combine_or(masks, ref_rgb.shape[:2])
    m = (m > 0.2).astype(np.float32)
    m = _sm._grow(m, expand)
    m = _sm._feather(m, feather)
    return np.clip(m, 0.0, 1.0)


class ZiTPreserveRegions:
    """Paste original face/hair/clothes back over an i2i edit (hard identity lock)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE",),
                "edited": ("IMAGE",),
                "preserve_face": ("BOOLEAN", {"default": True}),
                "preserve_hair": ("BOOLEAN", {"default": True}),
                "preserve_clothes": ("BOOLEAN", {"default": False}),
                "threshold": (
                    "FLOAT",
                    {"default": 0.30, "min": 0.10, "max": 0.90, "step": 0.01,
                     "tooltip": "Lower = bigger preserve mask."},
                ),
                "expand": ("INT", {"default": 6, "min": 0, "max": 128, "step": 1}),
                "feather": (
                    "INT",
                    {"default": 16, "min": 0, "max": 128, "step": 1,
                     "tooltip": "Soft edge so the locked region blends in."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("image", "preview", "preserve_mask")
    FUNCTION = "run"
    CATEGORY = "ZiT-Upscale/Edit"

    def run(self, original, edited, preserve_face, preserve_hair, preserve_clothes, threshold, expand, feather):
        keys = []
        if preserve_face:
            keys.append("face")
        if preserve_hair:
            keys.append("hair")
        if preserve_clothes:
            keys.append("clothes")

        batch = edited.shape[0]
        ref_batch = original.shape[0]
        outs, prevs, masks = [], [], []

        for i in range(batch):
            ref_rgb = _img_to_rgb_u8(original[min(i, ref_batch - 1): min(i, ref_batch - 1) + 1])
            ed = edited[i: i + 1]
            eh, ew = ed.shape[1], ed.shape[2]

            mask = _region_mask(ref_rgb, keys, float(threshold), int(expand), int(feather))

            if float(mask.max()) < 0.02:
                out = ed
                mask_resized = np.zeros((eh, ew), dtype=np.float32)
            else:
                mask_t = torch.from_numpy(mask)[None, ...]
                ref_t = torch.from_numpy(ref_rgb.astype(np.float32) / 255.0)[None, ...]
                out = composite_preserve(ref_t, ed, mask_t)
                mask_resized = cv2.resize(mask, (ew, eh), interpolation=cv2.INTER_LINEAR)

            out_rgb = _img_to_rgb_u8(out).astype(np.float32) / 255.0
            green = np.zeros_like(out_rgb)
            green[..., 1] = 1.0
            alpha = mask_resized[..., None] * 0.5
            prevs.append(torch.from_numpy(out_rgb * (1.0 - alpha) + green * alpha).float())
            outs.append(out[0])
            masks.append(torch.from_numpy(mask_resized).float())

        return (torch.stack(outs, 0), torch.stack(prevs, 0), torch.stack(masks, 0))


NODE_CLASS_MAPPINGS = {
    "ZiTPreserveRegions": ZiTPreserveRegions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZiTPreserveRegions": "ZiT Preserve Regions (garde visage/cheveux/vetements)",
}
