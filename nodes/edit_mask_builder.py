"""Build edit + preserve masks per body region (ComfyUI node)."""
from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    from .morpho_core import (
        CLOTHED_VOLUME_REGIONS,
        EDIT_REGIONS,
        MORPHO_STRATEGIES,
        build_morpho_masks,
        intersect_manual_mask,
        preserve_preview,
        _tensor_to_rgb_uint8,
    )
    from .edit_mask import mask_preview
except ImportError:
    from morpho_core import (
        CLOTHED_VOLUME_REGIONS,
        EDIT_REGIONS,
        MORPHO_STRATEGIES,
        build_morpho_masks,
        intersect_manual_mask,
        preserve_preview,
        _tensor_to_rgb_uint8,
    )
    from edit_mask import mask_preview


class ZiTEditMaskBuilder:
    """Detect body region → edit mask (volume) + preserve mask (clothes lock)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "edit_region": (EDIT_REGIONS, {"default": "breasts"}),
                "strategy": (
                    MORPHO_STRATEGIES,
                    {
                        "default": "volume_restore",
                        "tooltip": "volume_restore = seins entiers sous vetements + restore tissu apres inpaint.",
                    },
                ),
                "breast_size": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.65, "max": 1.55, "step": 0.05},
                ),
                "expand": ("INT", {"default": 12, "min": 0, "max": 64, "step": 1}),
                "feather": ("INT", {"default": 16, "min": 0, "max": 64, "step": 1}),
                "restore_clothes": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "preserve_mask = pixels vetements (bikini) recolles apres morpho.",
                    },
                ),
                "lock_face": ("BOOLEAN", {"default": True}),
                "lock_hair": ("BOOLEAN", {"default": True}),
                "lock_background": ("BOOLEAN", {"default": True}),
                "lock_arms": ("BOOLEAN", {"default": True}),
                "lock_legs": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prompt_extra": ("STRING", {"default": "", "multiline": False}),
                "threshold": ("FLOAT", {"default": 0.30, "min": 0.10, "max": 0.90, "step": 0.01}),
                "mask_override": ("MASK", {"tooltip": "Intersect with auto mask (peinture manuelle)."}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("edit_mask", "preserve_mask", "edit_preview", "preserve_preview")
    FUNCTION = "build"
    CATEGORY = "ZiT-Upscale/Edit"

    def build(
        self,
        image,
        edit_region,
        strategy,
        breast_size,
        expand,
        feather,
        restore_clothes,
        lock_face,
        lock_hair,
        lock_background,
        lock_arms,
        lock_legs,
        prompt_extra="",
        threshold=0.30,
        mask_override=None,
    ):
        try:
            import mediapipe  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("ZiTEditMaskBuilder needs mediapipe. pip install mediapipe") from exc

        if edit_region in CLOTHED_VOLUME_REGIONS and strategy == "volume_restore":
            restore_clothes = True

        batch = image.shape[0]
        edits, preserves, edit_prevs, preserve_prevs = [], [], [], []

        for i in range(batch):
            bgr = _tensor_to_rgb_uint8(image[i : i + 1])
            edit, preserve = build_morpho_masks(
                bgr,
                edit_region,
                strategy,
                prompt_extra=prompt_extra,
                threshold=threshold,
                breast_size=breast_size,
                expand=expand,
                feather=feather,
                lock_face=lock_face,
                lock_hair=lock_hair,
                lock_background=lock_background,
                lock_arms=lock_arms,
                lock_legs=lock_legs,
                restore_clothes=restore_clothes,
            )

            if mask_override is not None:
                m = mask_override[i] if mask_override.shape[0] > i else mask_override[0]
                manual = m.detach().cpu().numpy().astype(np.float32)
                edit = intersect_manual_mask(edit, manual)

            if float(edit.max()) < 0.02:
                raise RuntimeError(
                    f"Masque edit vide (region={edit_region}, strategy={strategy}). "
                    "Monte breast_size/expand ou peins mask_override."
                )

            edits.append(edit)
            preserves.append(preserve)
            edit_prevs.append(mask_preview(bgr, edit))
            preserve_prevs.append(preserve_preview(bgr, preserve))

        edit_t = torch.from_numpy(np.stack(edits, axis=0)).float()
        preserve_t = torch.from_numpy(np.stack(preserves, axis=0)).float()
        edit_prev_t = torch.from_numpy(np.stack(edit_prevs, axis=0)).float()
        preserve_prev_t = torch.from_numpy(np.stack(preserve_prevs, axis=0)).float()
        return (edit_t, preserve_t, edit_prev_t, preserve_prev_t)


NODE_CLASS_MAPPINGS = {
    "ZiTEditMaskBuilder": ZiTEditMaskBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZiTEditMaskBuilder": "ZiT Edit Mask Builder (region + preserve)",
}
