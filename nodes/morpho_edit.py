"""All-in-one morpho edit: volume mask → inpaint → clothing restore."""
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
        composite_inside_mask,
        composite_preserve,
        default_morpho_denoise,
        run_sdxl_inpaint,
        _tensor_to_rgb_uint8,
    )
    from .edit_mask import mask_preview
except ImportError:
    from morpho_core import (
        CLOTHED_VOLUME_REGIONS,
        EDIT_REGIONS,
        MORPHO_STRATEGIES,
        build_morpho_masks,
        composite_inside_mask,
        composite_preserve,
        default_morpho_denoise,
        run_sdxl_inpaint,
        _tensor_to_rgb_uint8,
    )
    from edit_mask import mask_preview


class ZiTMorphoEdit:
    """Regional morpho with optional clothing pixel restore (bikini-safe)."""

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "edit_region": (EDIT_REGIONS, {"default": "breasts"}),
                "strategy": (MORPHO_STRATEGIES, {"default": "volume_restore"}),
                "breast_size": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.65, "max": 1.55, "step": 0.05},
                ),
                "expand": ("INT", {"default": 12, "min": 0, "max": 64, "step": 1}),
                "feather": ("INT", {"default": 16, "min": 0, "max": 64, "step": 1}),
                "restore_clothes": ("BOOLEAN", {"default": True}),
                "hard_lock_outside": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Hors zone = original au pixel pres (visage, fond, bas du corps)."},
                ),
                "lock_face": ("BOOLEAN", {"default": True}),
                "lock_hair": ("BOOLEAN", {"default": True}),
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
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 0.68,
                        "min": 0.05,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "0.62–0.75 volume_restore. Re-queue pour cumuler.",
                    },
                ),
                "refine_denoise": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 0.85,
                        "step": 0.01,
                        "tooltip": "0 = off. 0.35–0.45 = 2e passe renforce morpho.",
                    },
                ),
            },
            "optional": {
                "edit_mask": ("MASK", {"tooltip": "Skip auto — branche ZiT Edit Mask Builder."}),
                "preserve_mask": ("MASK", {"tooltip": "Masque vetements a recoller (cyan preview)."}),
                "prompt_extra": ("STRING", {"default": "", "multiline": False}),
                "threshold": ("FLOAT", {"default": 0.30, "min": 0.10, "max": 0.90, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "edit_mask", "preserve_mask", "mask_preview")
    FUNCTION = "edit"
    CATEGORY = "ZiT-Upscale/Edit"

    def edit(
        self,
        image,
        model,
        vae,
        positive,
        negative,
        edit_region,
        strategy,
        breast_size,
        expand,
        feather,
        restore_clothes,
        hard_lock_outside,
        lock_face,
        lock_hair,
        lock_background,
        lock_arms,
        lock_legs,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        refine_denoise,
        edit_mask=None,
        preserve_mask=None,
        prompt_extra="",
        threshold=0.30,
    ):
        try:
            import mediapipe  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("ZiTMorphoEdit needs mediapipe. pip install mediapipe") from exc

        batch = image.shape[0]
        edit_masks = []
        preserve_masks = []
        previews = []

        for i in range(batch):
            bgr = _tensor_to_rgb_uint8(image[i : i + 1])
            eh, ew = bgr.shape[:2]

            painted = None
            if edit_mask is not None:
                m = edit_mask[i] if edit_mask.shape[0] > i else edit_mask[0]
                painted = m.detach().cpu().numpy().astype(np.float32)
                if painted.shape[:2] != (eh, ew):
                    import cv2

                    painted = cv2.resize(painted, (ew, eh), interpolation=cv2.INTER_LINEAR)
                if float(painted.max()) < 0.02:
                    # Rien peint dans le MaskEditor -> on retombe sur l'auto-detection.
                    painted = None

            if painted is not None:
                edit_np = painted
                if preserve_mask is not None:
                    p = preserve_mask[i] if preserve_mask.shape[0] > i else preserve_mask[0]
                    preserve_np = p.detach().cpu().numpy().astype(np.float32)
                    if preserve_np.shape[:2] != (eh, ew):
                        import cv2

                        preserve_np = cv2.resize(preserve_np, (ew, eh), interpolation=cv2.INTER_LINEAR)
                elif restore_clothes:
                    # Auto-lock the bikini fabric INSIDE the painted region (cups stay pixel-perfect).
                    try:
                        from morpho_core import _segment_masks
                    except ImportError:
                        from .morpho_core import _segment_masks
                    seg = _segment_masks(bgr, 0.40)
                    preserve_np = np.minimum(seg["clothes"].astype(np.float32), (edit_np > 0.2).astype(np.float32))
                else:
                    preserve_np = np.zeros(edit_np.shape, dtype=np.float32)
            else:
                edit_np, preserve_np = build_morpho_masks(
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

            if float(edit_np.max()) < 0.02:
                raise RuntimeError(
                    f"Masque edit vide ({edit_region}/{strategy}). "
                    "Peins la zone des seins dans le MaskEditor (clic droit sur Load Image), "
                    "ou monte breast_size, ou baisse threshold pour l'auto-detection."
                )

            cov = float((edit_np > 0.5).mean()) * 100.0
            pcov = float((preserve_np > 0.5).mean()) * 100.0
            src_kind = "PEINT" if painted is not None else "AUTO"
            logger.info(
                "ZiTMorphoEdit[%d]: masque=%s couverture=%.2f%% preserve=%.2f%% (image %dx%d)",
                i, src_kind, cov, pcov, ew, eh,
            )

            edit_masks.append(edit_np)
            preserve_masks.append(preserve_np)
            previews.append(mask_preview(bgr, edit_np))

        edit_mask_t = torch.from_numpy(np.stack(edit_masks, axis=0)).float()
        preserve_mask_t = torch.from_numpy(np.stack(preserve_masks, axis=0)).float()
        preview_t = torch.from_numpy(np.stack(previews, axis=0)).float()

        if denoise < 0.45 and edit_region in CLOTHED_VOLUME_REGIONS and strategy == "volume_restore":
            suggested = default_morpho_denoise(edit_region, strategy)
            logger.warning("ZiTMorphoEdit: denoise %.2f faible pour volume — essaie %.2f+", denoise, suggested)

        results = []
        for i in range(batch):
            src = image[i : i + 1]
            em = edit_mask_t[i : i + 1]
            pm = preserve_mask_t[i : i + 1]

            out = run_sdxl_inpaint(
                src,
                em,
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
            logger.info(
                "ZiTMorphoEdit[%d]: inpaint delta moyen=%.5f (0.0 = AUCUN changement)",
                i, float((out - src).abs().mean()),
            )

            if refine_denoise > 0.02:
                out = run_sdxl_inpaint(
                    out,
                    em,
                    model,
                    vae,
                    positive,
                    negative,
                    seed + i + 1000,
                    max(steps // 2, 12),
                    cfg,
                    sampler_name,
                    scheduler,
                    refine_denoise,
                )

            if hard_lock_outside:
                out = composite_inside_mask(src, out, em, feather=max(2, feather // 3))
            out = composite_preserve(src, out, pm)
            results.append(out)

        result = results[0] if batch == 1 else torch.cat(results, dim=0)
        return (result, edit_mask_t, preserve_mask_t, preview_t)


NODE_CLASS_MAPPINGS = {
    "ZiTMorphoEdit": ZiTMorphoEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZiTMorphoEdit": "ZiT Morpho Edit (volume + restore vetements)",
}
