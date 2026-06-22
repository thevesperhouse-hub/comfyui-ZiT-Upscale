"""Restore the original face + match colors AFTER generative edits (Qwen / ZiT).

Both Qwen-Image-Edit and Z-Image regenerate the whole frame, so they drift the
face identity and push a warm/orange skin cast. Trying to "hold" the face during
diffusion is unreliable on non-inpaint models. Instead this node fixes both
AFTER the edit, so the result is guaranteed regardless of sampler/steps:

  1. color-match the edited image back to the original (LAB mean/std transfer),
     'chroma_only' fixes the orange/beige hue without touching brightness;
  2. paste the ORIGINAL face back over the edit (CLIPSeg face mask, feathered),
     so identity is pixel-faithful.

Put it at the very END of the chain (after the ZiT upscale) and feed it the
clean original image + the edited result.

Credits: CLIPSeg by CIDAS. Z-Image / ZiT by Tongyi-MAI (Alibaba).
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
    """(H,W,3) uint8 RGB from a single-image float tensor (0..1)."""
    arr = img_t.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def _color_match(edited_rgb: np.ndarray, ref_rgb: np.ndarray, strength: float, chroma_only: bool) -> np.ndarray:
    """Reinhard LAB transfer: push edited color statistics towards the original."""
    if strength <= 0.0:
        return edited_rgb
    h, w = edited_rgb.shape[:2]
    ref = cv2.resize(ref_rgb, (w, h), interpolation=cv2.INTER_AREA)

    src_lab = cv2.cvtColor(edited_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)

    out = src_lab.copy()
    channels = (1, 2) if chroma_only else (0, 1, 2)
    for c in channels:
        s_mean, s_std = float(src_lab[..., c].mean()), float(src_lab[..., c].std()) + 1e-5
        r_mean, r_std = float(ref_lab[..., c].mean()), float(ref_lab[..., c].std()) + 1e-5
        out[..., c] = (src_lab[..., c] - s_mean) * (r_std / s_std) + r_mean

    out = np.clip(out, 0, 255).astype(np.uint8)
    matched = cv2.cvtColor(out, cv2.COLOR_LAB2RGB).astype(np.float32)
    blended = edited_rgb.astype(np.float32) * (1.0 - strength) + matched * strength
    return np.clip(blended, 0, 255).astype(np.uint8)


def _face_mask(ref_rgb: np.ndarray, threshold: float, expand: int, feather: int) -> np.ndarray:
    """CLIPSeg face mask at the original resolution (float 0..1)."""
    from PIL import Image

    pil = Image.fromarray(ref_rgb)
    mask = _sm._clipseg_mask(pil, [_sm.LOCK_PROMPTS["face"]], threshold)
    mask = (mask > 0.2).astype(np.float32)
    mask = _sm._grow(mask, expand)
    mask = _sm._feather(mask, feather)
    return np.clip(mask, 0.0, 1.0)


class ZiTFaceColorRestore:
    """Paste the original face back + match colors after Qwen/ZiT edits."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE",),
                "edited": ("IMAGE",),
                "preserve_face": ("BOOLEAN", {"default": True}),
                "face_threshold": (
                    "FLOAT",
                    {"default": 0.30, "min": 0.10, "max": 0.90, "step": 0.01,
                     "tooltip": "Lower = bigger face mask."},
                ),
                "face_expand": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1}),
                "face_feather": (
                    "INT",
                    {"default": 24, "min": 0, "max": 128, "step": 1,
                     "tooltip": "Soft edge so the pasted face blends in."},
                ),
                "color_match": ("BOOLEAN", {"default": True}),
                "color_strength": (
                    "FLOAT",
                    {"default": 0.70, "min": 0.0, "max": 1.0, "step": 0.05,
                     "tooltip": "How strongly to recolor towards the original."},
                ),
                "color_mode": (
                    ["chroma_only", "full"],
                    {"default": "chroma_only",
                     "tooltip": "chroma_only = fix orange/beige hue, keep brightness. "
                                "full = also match contrast/brightness."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("image", "preview", "face_mask")
    FUNCTION = "restore"
    CATEGORY = "ZiT-Upscale/Edit"

    def restore(
        self,
        original,
        edited,
        preserve_face,
        face_threshold,
        face_expand,
        face_feather,
        color_match,
        color_strength,
        color_mode,
    ):
        chroma_only = color_mode == "chroma_only"
        batch = edited.shape[0]
        ref_batch = original.shape[0]

        out_imgs = []
        previews = []
        masks_out = []

        for i in range(batch):
            ref_rgb = _img_to_rgb_u8(original[min(i, ref_batch - 1): min(i, ref_batch - 1) + 1])
            edit_rgb = _img_to_rgb_u8(edited[i: i + 1])

            if color_match:
                edit_rgb = _color_match(edit_rgb, ref_rgb, float(color_strength), chroma_only)

            matched_t = torch.from_numpy(edit_rgb.astype(np.float32) / 255.0)[None, ...]
            eh, ew = edit_rgb.shape[:2]

            if preserve_face:
                fmask = _face_mask(ref_rgb, float(face_threshold), int(face_expand), int(face_feather))
                if float(fmask.max()) < 0.02:
                    # No face found — skip the paste rather than erroring out.
                    out_t = matched_t
                    mask_resized = np.zeros((eh, ew), dtype=np.float32)
                else:
                    mask_t = torch.from_numpy(fmask)[None, ...]
                    ref_t = torch.from_numpy(ref_rgb.astype(np.float32) / 255.0)[None, ...]
                    out_t = composite_preserve(ref_t, matched_t, mask_t)
                    mask_resized = cv2.resize(fmask, (ew, eh), interpolation=cv2.INTER_LINEAR)
            else:
                out_t = matched_t
                mask_resized = np.zeros((eh, ew), dtype=np.float32)

            out_rgb = _img_to_rgb_u8(out_t).astype(np.float32) / 255.0
            red = np.zeros_like(out_rgb)
            red[..., 0] = 1.0
            alpha = mask_resized[..., None] * 0.5
            prev = out_rgb * (1.0 - alpha) + red * alpha

            out_imgs.append(out_t[0])
            previews.append(torch.from_numpy(prev).float())
            masks_out.append(torch.from_numpy(mask_resized).float())

        return (
            torch.stack(out_imgs, axis=0),
            torch.stack(previews, axis=0),
            torch.stack(masks_out, axis=0),
        )


NODE_CLASS_MAPPINGS = {
    "ZiTFaceColorRestore": ZiTFaceColorRestore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZiTFaceColorRestore": "ZiT Face + Color Restore (anti-orange, garde le visage)",
}
