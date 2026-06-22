"""Composite original clothing pixels over an edited image."""
from __future__ import annotations

try:
    from .morpho_core import composite_preserve, preserve_preview, _tensor_to_rgb_uint8
except ImportError:
    from morpho_core import composite_preserve, preserve_preview, _tensor_to_rgb_uint8

import torch


class ZiTCompositePreserve:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE",),
                "edited": ("IMAGE",),
                "preserve_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "preview")
    FUNCTION = "composite"
    CATEGORY = "ZiT-Upscale/Edit"

    def composite(self, original, edited, preserve_mask):
        out = composite_preserve(original, edited, preserve_mask)
        bgr = _tensor_to_rgb_uint8(original[0:1])
        pm = preserve_mask[0].detach().cpu().numpy()
        prev = torch.from_numpy(preserve_preview(bgr, pm)[None, ...]).float()
        return (out, prev)


NODE_CLASS_MAPPINGS = {
    "ZiTCompositePreserve": ZiTCompositePreserve,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZiTCompositePreserve": "ZiT Composite Preserve (recolle vetements)",
}
