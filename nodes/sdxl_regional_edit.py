"""SDXL inpaint pass driven by ZiT region masks — no empty-mask img2img dead ends."""
from __future__ import annotations

import logging

import node_helpers
import torch

logger = logging.getLogger(__name__)


class ZiTSDXLRegionalEdit:
    """Inpaint only the masked region. White = change, black = frozen."""

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        return {
            "required": {
                "image": ("IMAGE",),
                "edit_mask": ("MASK",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
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
                        "default": 0.72,
                        "min": 0.05,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Too low = no visible change. Bust/waist/belly: try 0.6–0.8.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "edit"
    CATEGORY = "ZiT-Upscale/Edit"

    def edit(
        self,
        image,
        edit_mask,
        model,
        clip,
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

        del clip
        if edit_mask.shape[0] != image.shape[0]:
            if edit_mask.shape[0] == 1:
                edit_mask = edit_mask.repeat(image.shape[0], 1, 1)
            else:
                raise ValueError("edit_mask batch size must match image")

        mask_max = float(edit_mask.max().item())
        if mask_max < 0.02:
            raise RuntimeError(
                "Edit mask is empty (all black). Change edit_region or disable locks in ZiT Region Mask."
            )

        if denoise < 0.35:
            logger.warning(
                "ZiTSDXLRegionalEdit: denoise %.2f is low for body morph — try 0.65+.",
                denoise,
            )

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
        for i in range(3):
            pixels[:, :, :, i] -= 0.5
            pixels[:, :, :, i] *= m
            pixels[:, :, :, i] += 0.5

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

        out = vae.decode(samples["samples"])
        return (out,)


NODE_CLASS_MAPPINGS = {
    "ZiTSDXLRegionalEdit": ZiTSDXLRegionalEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZiTSDXLRegionalEdit": "ZiT SDXL Regional Edit (inpaint)",
}
