"""All-in-one regional SDXL edit: auto mask + inpaint + preview."""
from __future__ import annotations

import logging

import node_helpers
import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    from .edit_mask import (
        EDIT_REGIONS,
        MASK_MODES,
        build_edit_mask,
        default_denoise,
        mask_preview,
        _tensor_to_rgb_uint8,
    )
    from .morpho_core import align_mask_np
except ImportError:
    from edit_mask import (
        EDIT_REGIONS,
        MASK_MODES,
        build_edit_mask,
        default_denoise,
        mask_preview,
        _tensor_to_rgb_uint8,
    )
    from morpho_core import align_mask_np


class ZiTRegionalEdit:
    """Pick a body zone, preview the mask, inpaint only that area."""

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "edit_region": (EDIT_REGIONS, {"default": "breasts"}),
                "mask_mode": (
                    MASK_MODES,
                    {
                        "default": "pose",
                        "tooltip": "pose = ellipses MediaPipe (fiable avec texte sur vetements). hybrid = pose + CLIPSeg si coherent.",
                    },
                ),
                "prompt_extra": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "large breasts, side view, pink top…",
                    },
                ),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.30,
                        "min": 0.10,
                        "max": 0.90,
                        "step": 0.01,
                        "tooltip": "CLIPSeg sensitivity. Lower = bigger zone.",
                    },
                ),
                "breast_size": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.65,
                        "max": 1.50,
                        "step": 0.05,
                        "tooltip": "Scale chest geometry if hybrid preview misses volume.",
                    },
                ),
                "expand": ("INT", {"default": 10, "min": 0, "max": 64, "step": 1}),
                "feather": ("INT", {"default": 14, "min": 0, "max": 64, "step": 1}),
                "lock_face": ("BOOLEAN", {"default": True}),
                "lock_hair": ("BOOLEAN", {"default": True}),
                "lock_clothes": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Keep OFF for breasts/waist — locking clothes removes the fabric you want to edit.",
                    },
                ),
                "lock_background": ("BOOLEAN", {"default": True}),
                "lock_arms": ("BOOLEAN", {"default": True}),
                "lock_legs": ("BOOLEAN", {"default": False}),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "steps": ("INT", {"default": 22, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 0.68,
                        "min": 0.05,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "0.65–0.85 for morpho. Too low = no visible change.",
                    },
                ),
            },
            "optional": {
                "mask_override": (
                    "MASK",
                    {"tooltip": "Skip auto mask — paint in LoadImage or plug a custom mask here."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "edit_mask", "mask_preview")
    FUNCTION = "edit"
    CATEGORY = "ZiT-Upscale/Edit"

    def edit(
        self,
        image,
        model,
        clip,
        vae,
        positive,
        negative,
        edit_region,
        mask_mode,
        prompt_extra,
        threshold,
        breast_size,
        expand,
        feather,
        lock_face,
        lock_hair,
        lock_clothes,
        lock_background,
        lock_arms,
        lock_legs,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        mask_override=None,
    ):
        del clip

        try:
            import mediapipe  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "ZiTRegionalEdit needs mediapipe for regional masks. Install: pip install mediapipe"
            ) from exc

        batch = image.shape[0]
        mask_tensors = []
        preview_tensors = []

        for i in range(batch):
            bgr = _tensor_to_rgb_uint8(image[i : i + 1])
            h, w = bgr.shape[:2]

            if mask_override is not None:
                m = mask_override[i] if mask_override.shape[0] > i else mask_override[0]
                mask_np = align_mask_np(m.detach().cpu().numpy().astype(np.float32), h, w)
                if float(mask_np.max()) < 0.02:
                    mask_np = build_edit_mask(
                        bgr,
                        edit_region,
                        mask_mode,
                        prompt_extra=prompt_extra,
                        threshold=threshold,
                        confidence=0.40,
                        expand=expand,
                        feather=feather,
                        breast_size=breast_size,
                        lock_face=lock_face,
                        lock_hair=lock_hair,
                        lock_clothes=lock_clothes,
                        lock_background=lock_background,
                        lock_arms=lock_arms,
                        lock_legs=lock_legs,
                    )
            else:
                mask_np = build_edit_mask(
                    bgr,
                    edit_region,
                    mask_mode,
                    prompt_extra=prompt_extra,
                    threshold=threshold,
                    confidence=0.40,
                    expand=expand,
                    feather=feather,
                    breast_size=breast_size,
                    lock_face=lock_face,
                    lock_hair=lock_hair,
                    lock_clothes=lock_clothes,
                    lock_background=lock_background,
                    lock_arms=lock_arms,
                    lock_legs=lock_legs,
                )

            if float(mask_np.max()) < 0.02:
                raise RuntimeError(
                    f"Edit mask empty for region={edit_region!r} mode={mask_mode!r}. "
                    "Lower threshold, raise breast_size, add prompt_extra, or paint mask_override."
                )

            mask_tensors.append(mask_np)
            preview_tensors.append(mask_preview(bgr, mask_np))

        edit_mask = torch.from_numpy(np.stack(mask_tensors, axis=0)).float()
        mask_preview_t = torch.from_numpy(np.stack(preview_tensors, axis=0)).float()

        if denoise < 0.40 and edit_region in ("breasts", "bust", "waist", "belly", "hips", "full_body_no_face"):
            suggested = default_denoise(edit_region)
            logger.warning(
                "ZiTRegionalEdit: denoise %.2f is low for %s — try %.2f+.",
                denoise,
                edit_region,
                suggested,
            )

        out_images = []
        for i in range(batch):
            img = image[i : i + 1]
            m = edit_mask[i : i + 1]
            out = self._inpaint(
                img,
                m,
                model,
                vae,
                positive,
                negative,
                seed + i,
                steps,
                cfg,
                sampler_name,
                scheduler,
                denoise,
            )
            out_images.append(out)

        if batch == 1:
            result = out_images[0]
        else:
            result = torch.cat(out_images, dim=0)

        return (result, edit_mask, mask_preview_t)

    @staticmethod
    def _inpaint(
        image,
        edit_mask,
        model,
        vae,
        positive,
        negative,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
    ):
        from nodes import common_ksampler

        pixels = image
        mask = edit_mask

        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        mask_b = torch.nn.functional.interpolate(
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
            size=(pixels.shape[1], pixels.shape[2]),
            mode="bilinear",
        )

        orig_pixels = pixels
        pixels = orig_pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset : x + x_offset, y_offset : y + y_offset, :]
            mask_b = mask_b[:, :, x_offset : x + x_offset, y_offset : y + y_offset]

        m = (1.0 - mask_b.round()).squeeze(1)
        for ch in range(3):
            pixels[:, :, :, ch] -= 0.5
            pixels[:, :, :, ch] *= m
            pixels[:, :, :, ch] += 0.5

        concat_latent = vae.encode(pixels)
        orig_latent = vae.encode(orig_pixels)
        latent = {"samples": orig_latent, "noise_mask": mask_b}

        positive = node_helpers.conditioning_set_values(
            positive,
            {"concat_latent_image": concat_latent, "concat_mask": mask_b},
        )
        negative = node_helpers.conditioning_set_values(
            negative,
            {"concat_latent_image": concat_latent, "concat_mask": mask_b},
        )

        samples = common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent,
            denoise=denoise,
        )[0]

        return vae.decode(samples["samples"])


NODE_CLASS_MAPPINGS = {
    "ZiTRegionalEdit": ZiTRegionalEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZiTRegionalEdit": "ZiT Regional Edit (mask + inpaint)",
}
