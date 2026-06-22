"""ZiT Morph Dataset — generate N labeled training pairs from ONE base image.

Goal: build clean (input, instruction) -> (target) pairs for fine-tuning a maskless,
prompt-driven body-morph editor. From a single base image + a painted zone, this node:

  1. runs N prompt-driven plain SDXL img2img edits (NO noise_mask),
  2. localizes each change to the painted zone via composite_inside_mask, so everything
     outside the zone is restored to the original pixel-perfect (face, clothes, bg),
  3. saves each target + a caption .txt (subject context + edit instruction),
  4. appends a metadata.jsonl manifest (input / target / instruction) ready for training.

Why plain img2img + composite instead of inpaint conditioning: ComfyUI's noise_mask
re-composites the original latent every step, which pins non-inpaint checkpoints back to
the base image ('generates then reverts'). Plain img2img genuinely changes; we localize after.
The generation method is a DATA-PREP tool — the shipped model is maskless.
Wire Florence2 (caption output) into `subject_caption` so each instruction carries the base
subject context, exactly as the user asked (base image -> transfo 1 -> transfo 2 ...).

Credits: Z-Image / ZiT by Tongyi-MAI (Alibaba). Text encoder: Qwen (Alibaba). Florence2 by Microsoft.
"""
from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    from .morpho_core import (
        align_mask_tensor,
        build_morpho_masks,
        composite_inside_mask,
        _combine_or,
        _segment_masks,
        _tensor_to_rgb_uint8,
    )
except ImportError:
    from morpho_core import (
        align_mask_tensor,
        build_morpho_masks,
        composite_inside_mask,
        _combine_or,
        _segment_masks,
        _tensor_to_rgb_uint8,
    )


def _img2img(src, model, vae, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise):
    """Plain SDXL img2img — NO noise_mask. The masked region is localized afterwards via
    composite_inside_mask. Avoids ComfyUI's inpaint mask-compositing that pins the latent
    back to the original on non-inpaint checkpoints (the 'reverts to base' bug)."""
    from nodes import common_ksampler

    latent = {"samples": vae.encode(src[:, :, :, :3])}
    samples = common_ksampler(
        model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=denoise
    )[0]
    return vae.decode(samples["samples"])


# 15 transformations, slimmest -> biggest. One line each: "label | edit instruction | denoise".
# Edit ONLY the instruction text to morph any zone you painted (breasts, butt, waist, thighs...).
DEFAULT_TRANSFORMS = """\
01_flat        | completely flat chest, no breasts                       | 0.70
02_xs          | very small breasts, almost flat                          | 0.62
03_small       | small breasts                                            | 0.55
04_modest      | modest natural breasts                                   | 0.50
05_natural     | natural medium breasts                                   | 0.45
06_full        | full rounded breasts                                     | 0.50
07_large       | large breasts                                            | 0.55
08_xl          | very large heavy breasts                                 | 0.60
09_huge        | huge breasts                                             | 0.65
10_enormous    | enormous breasts                                         | 0.70
11_gigantic    | gigantic oversized breasts                               | 0.74
12_extreme     | extremely huge breasts, far beyond natural               | 0.78
13_hyper       | hyper huge breasts, unrealistic massive volume           | 0.82
14_colossal    | colossal breasts dominating the torso                    | 0.86
15_max         | maximum impossible breast volume, extreme expansion      | 0.90
"""


def _save_png(rgb_uint8: np.ndarray, path: str) -> None:
    from PIL import Image

    Image.fromarray(rgb_uint8).save(path, compress_level=4)


def _parse_transforms(text: str, default_denoise: float) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        label = parts[0] if parts else f"t{len(rows) + 1:02d}"
        instruction = parts[1] if len(parts) > 1 and parts[1] else label
        denoise = default_denoise
        if len(parts) > 2 and parts[2]:
            try:
                denoise = float(parts[2])
            except ValueError:
                pass
        rows.append((label, instruction, max(0.05, min(1.0, denoise))))
    return rows


class ZiTMorphDataset:
    """Generate N labeled morph pairs (input + targets + captions) from one painted base."""

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "transforms": ("STRING", {"default": DEFAULT_TRANSFORMS, "multiline": True}),
                "negative_text": (
                    "STRING",
                    {"default": "blurry, deformed, extra limbs, bad anatomy, lowres, watermark", "multiline": True},
                ),
                "dataset_dir": ("STRING", {"default": "ZiT-Morph-Dataset"}),
                "pair_tag": ("STRING", {"default": "pair"}),
                "expand": ("INT", {"default": 24, "min": 0, "max": 128, "step": 1}),
                "feather": ("INT", {"default": 24, "min": 0, "max": 128, "step": 1}),
                "hard_lock_outside": ("BOOLEAN", {"default": True}),
                "lock_face": ("BOOLEAN", {"default": True, "tooltip": "Visage + cheveux JAMAIS modifies (force l'original)."}),
                "lock_clothes": ("BOOLEAN", {"default": True, "tooltip": "Vetements JAMAIS modifies. OFF pour photos peau nue."}),
                "save_original": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "default_denoise": ("FLOAT", {"default": 0.60, "min": 0.05, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "edit_mask": ("MASK", {"tooltip": "Painted zone to morph (MaskEditor). Empty -> auto breast detect."}),
                "subject_caption": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "Wire Florence2 caption of the BASE image here."},
                ),
                "prompt_prefix": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("transformations", "original", "summary")
    FUNCTION = "generate"
    CATEGORY = "ZiT-Upscale/Dataset"

    def generate(
        self,
        image,
        model,
        clip,
        vae,
        transforms,
        negative_text,
        dataset_dir,
        pair_tag,
        expand,
        feather,
        hard_lock_outside,
        lock_face,
        lock_clothes,
        save_original,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        default_denoise,
        edit_mask=None,
        subject_caption="",
        prompt_prefix="",
    ):
        import folder_paths
        from nodes import CLIPTextEncode

        rows = _parse_transforms(transforms, default_denoise)
        if not rows:
            raise RuntimeError("ZiTMorphDataset: aucune transformation. Remplis le champ 'transforms'.")

        src = image[:1]  # one base per run
        eh, ew = int(src.shape[1]), int(src.shape[2])

        # --- Resolve the edit zone (painted mask, else auto-detect breasts) ---
        em = None
        if edit_mask is not None:
            m = align_mask_tensor(edit_mask, eh, ew)
            if float(m.max()) >= 0.02:
                mraw = m[:1]
                # Polarity guard: MaskEditor/LoadImage can output an INVERTED mask.
                # A hand-painted zone is small; >55% coverage means it's inverted -> flip it.
                cov_raw = float((mraw > 0.5).float().mean())
                if cov_raw > 0.55:
                    logger.warning(
                        "ZiTMorphDataset: masque peint couvre %.0f%% (>55%%) -> polarite INVERSEE, je le retourne.",
                        cov_raw * 100.0,
                    )
                    mraw = 1.0 - mraw
                em = mraw
        if em is None:
            bgr = _tensor_to_rgb_uint8(src)
            edit_np, _ = build_morpho_masks(
                bgr, "breasts", "volume_restore", expand=expand, feather=feather, restore_clothes=False
            )
            if float(edit_np.max()) < 0.02:
                raise RuntimeError(
                    "ZiTMorphDataset: zone vide. Peins la zone a morpher dans le MaskEditor "
                    "(clic droit sur Load Image), ou la detection auto a echoue."
                )
            em = torch.from_numpy(edit_np[None, ...]).float()

        # --- Feather the edit zone, then HARD-LOCK face/hair/clothes out of it ---
        # Even if the paint or the feather spills onto the face or clothes, those pixels
        # are forced to 0 in the edit mask, so they can never be regenerated.
        import cv2

        bgr = _tensor_to_rgb_uint8(src)
        em_np = em[0].detach().cpu().numpy().astype(np.float32)
        if feather > 0:
            k = int(feather) * 2 + 1
            em_np = cv2.GaussianBlur(em_np, (k, k), 0)

        lock_np = np.zeros((eh, ew), dtype=np.float32)
        if lock_face or lock_clothes:
            try:
                seg = _segment_masks(bgr, 0.40)
                locks = []
                if lock_face:
                    locks.append(seg["face"])
                    locks.append(seg["hair"])
                if lock_clothes:
                    locks.append(seg["clothes"])
                lock_np = _combine_or(locks, (eh, ew))
            except Exception as exc:  # noqa: BLE001
                logger.warning("ZiTMorphDataset: lock visage/vetements echoue (%s).", exc)

        em_np = np.clip(em_np * (1.0 - lock_np), 0.0, 1.0)
        em = torch.from_numpy(em_np[None, ...]).float()

        coverage = float((em_np > 0.5).mean()) * 100.0
        logger.info(
            "ZiTMorphDataset: zone editable=%.2f%% (lock visage=%s vetements=%s, image %dx%d)",
            coverage, lock_face, lock_clothes, ew, eh,
        )
        if coverage < 0.2:
            raise RuntimeError(
                f"ZiTMorphDataset: zone editable trop petite ({coverage:.2f}%) apres lock. "
                "Peins une zone de PEAU plus large, ou desactive lock_clothes si la zone est sous le vetement."
            )

        # --- Output folder under ComfyUI/output ---
        out_root = os.path.join(folder_paths.get_output_directory(), dataset_dir)
        pair_id = f"{pair_tag}_{int(time.time())}"
        pair_dir = os.path.join(out_root, pair_id)
        os.makedirs(pair_dir, exist_ok=True)
        manifest_path = os.path.join(out_root, "metadata.jsonl")

        subject = subject_caption.strip()
        prefix = prompt_prefix.strip()
        # _tensor_to_rgb_uint8 returns BGR (for cv2); convert to RGB for saving.
        src_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        input_rel = f"{pair_id}/input.png"
        if save_original:
            _save_png(src_rgb, os.path.join(pair_dir, "input.png"))
            if subject:
                with open(os.path.join(pair_dir, "input.txt"), "w", encoding="utf-8") as fh:
                    fh.write(subject)

        neg_cond = CLIPTextEncode().encode(clip, negative_text)[0]

        # Red overlay of the FINAL editable zone (after polarity guard + face/clothes lock),
        # prepended to the preview so the exact zone that will change is visible at a glance.
        prev = src_rgb.astype(np.float32) / 255.0
        a = np.clip(em_np, 0.0, 1.0)[..., None] * 0.55
        red = np.zeros_like(prev)
        red[..., 0] = 1.0
        mask_overlay = torch.from_numpy((prev * (1.0 - a) + red * a)[None, ...]).float()

        results = []
        manifest_lines = []
        for idx, (label, instruction, denoise) in enumerate(rows):
            edit_instruction = ", ".join(p for p in [prefix, instruction] if p)
            positive_text = ", ".join(p for p in [subject, edit_instruction] if p) or edit_instruction
            pos_cond = CLIPTextEncode().encode(clip, positive_text)[0]

            edited = _img2img(
                src, model, vae, pos_cond, neg_cond,
                seed + idx, steps, cfg, sampler_name, scheduler, denoise,
            )
            raw_delta = float((edited - src).abs().mean())
            if hard_lock_outside:
                # em is already feathered + face/clothes-locked; no extra blur (would re-bleed).
                out = composite_inside_mask(src, edited, em, feather=0)
            else:
                out = edited

            delta = float((out - src).abs().mean())
            logger.info(
                "ZiTMorphDataset[%02d] %s denoise=%.2f img2img_delta=%.5f final_delta=%.5f",
                idx + 1, label, denoise, raw_delta, delta,
            )

            target_name = f"t{idx + 1:02d}_{label}.png"
            target_rgb = cv2.cvtColor(_tensor_to_rgb_uint8(out), cv2.COLOR_BGR2RGB)
            _save_png(target_rgb, os.path.join(pair_dir, target_name))
            with open(os.path.join(pair_dir, target_name.replace(".png", ".txt")), "w", encoding="utf-8") as fh:
                fh.write(positive_text)

            manifest_lines.append(json.dumps({
                "input": input_rel,
                "edited": f"{pair_id}/{target_name}",
                "edit_prompt": edit_instruction,
                "full_caption": positive_text,
                "subject": subject,
                "label": label,
                "denoise": denoise,
            }, ensure_ascii=False))
            results.append(out)

        with open(manifest_path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(manifest_lines) + "\n")

        montage = torch.cat([mask_overlay] + results, dim=0) if results else mask_overlay
        summary = (
            f"{len(rows)} transformations -> {pair_dir}\n"
            f"manifest: {manifest_path}\n"
            f"subject: {subject or '(none — wire Florence2)'}"
        )
        logger.info("ZiTMorphDataset: %s", summary.replace("\n", " | "))
        return (montage, src, summary)


NODE_CLASS_MAPPINGS = {
    "ZiTMorphDataset": ZiTMorphDataset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZiTMorphDataset": "ZiT Morph Dataset (N pairs + captions)",
}
