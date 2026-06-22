"""Morpho masks + SDXL inpaint + clothing preserve composite."""
from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    from .edit_mask import MASK_MODES, build_edit_mask, mask_preview
    from .region_mask import (
        EDIT_REGIONS,
        REGION_DENOISE_DEFAULTS,
        _arms_mask,
        _breasts_mask,
        _combine_and,
        _combine_or,
        _feather,
        _grow,
        _legs_mask,
        _pose_landmarks,
        _segment_masks,
        _tensor_to_rgb_uint8,
        _torso_band_mask,
        build_breast_skin_mask,
    )
except ImportError:
    from edit_mask import MASK_MODES, build_edit_mask, mask_preview
    from region_mask import (
        EDIT_REGIONS,
        REGION_DENOISE_DEFAULTS,
        _arms_mask,
        _breasts_mask,
        _combine_and,
        _combine_or,
        _feather,
        _grow,
        _legs_mask,
        _pose_landmarks,
        _segment_masks,
        _tensor_to_rgb_uint8,
        _torso_band_mask,
        build_breast_skin_mask,
    )

MORPHO_STRATEGIES = [
    "volume_restore",
    "hybrid",
    "pose",
    "skin_only",
]

CLOTHED_VOLUME_REGIONS = frozenset({"breasts", "bust"})


def default_morpho_denoise(edit_region: str, strategy: str) -> float:
    base = float(REGION_DENOISE_DEFAULTS.get(edit_region, 0.68))
    if strategy == "volume_restore" and edit_region in CLOTHED_VOLUME_REGIONS:
        return min(0.72, base)
    if strategy == "skin_only":
        return 0.58
    return base


def _subtract_locks(
    edit: np.ndarray,
    seg: dict,
    lm,
    *,
    lock_face: bool,
    lock_hair: bool,
    lock_background: bool,
    lock_arms: bool,
    lock_legs: bool,
) -> np.ndarray:
    h, w = seg["shape"]
    body = seg["body"]
    locks = []
    if lock_face:
        locks.append(seg["face"])
    if lock_hair:
        locks.append(seg["hair"])
    if lock_background:
        locks.append(seg["background"])
    if lock_arms:
        locks.append(_arms_mask(h, w, lm, body))
    if lock_legs:
        locks.append(_legs_mask(h, w, lm, body))
    if locks:
        locked = _combine_or(locks, (h, w))
        edit = _combine_and(edit, 1.0 - locked)
    return edit


def align_mask_np(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a manual mask to match image dimensions."""
    import cv2

    m = mask.astype(np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape[0] == height and m.shape[1] == width:
        return m
    return cv2.resize(m, (width, height), interpolation=cv2.INTER_LINEAR)


def intersect_manual_mask(
    edit: np.ndarray,
    manual: np.ndarray,
    *,
    min_coverage: float = 0.02,
) -> np.ndarray:
    """Bitwise AND with a painted mask; ignore empty / default LoadImage masks."""
    h, w = edit.shape[:2]
    manual = align_mask_np(manual, h, w)
    if float(manual.max()) < min_coverage:
        return edit
    return np.minimum(edit, manual)


def align_mask_tensor(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.shape[-2] == height and mask.shape[-1] == width:
        return mask
    m = mask.reshape(-1, 1, mask.shape[-2], mask.shape[-1])
    m = torch.nn.functional.interpolate(m, size=(height, width), mode="bilinear", align_corners=False)
    return m.reshape(mask.shape[0], height, width)


def manual_mask_is_active(mask: torch.Tensor | np.ndarray, *, min_coverage: float = 0.02) -> bool:
    if isinstance(mask, torch.Tensor):
        return float(mask.max()) >= min_coverage
    return float(np.max(mask)) >= min_coverage


def build_morpho_masks(
    bgr: np.ndarray,
    edit_region: str,
    strategy: str = "volume_restore",
    *,
    prompt_extra: str = "",
    threshold: float = 0.30,
    confidence: float = 0.40,
    expand: int = 12,
    feather: int = 16,
    breast_size: float = 1.0,
    lock_face: bool = True,
    lock_hair: bool = True,
    lock_background: bool = True,
    lock_arms: bool = True,
    lock_legs: bool = False,
    restore_clothes: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (edit_mask, preserve_mask).
    edit_mask = zone to morph (full volume for breasts when volume_restore).
    preserve_mask = clothing pixels pasted back after inpaint (skull/print intact).
    """
    seg = _segment_masks(bgr, confidence)
    h, w = seg["shape"]
    lm = _pose_landmarks(bgr)
    body = seg["body"]
    clothes = seg["clothes"]
    tissue = np.clip(body + clothes, 0.0, 1.0)
    preserve = np.zeros((h, w), dtype=np.float32)

    if strategy == "skin_only" and edit_region in CLOTHED_VOLUME_REGIONS:
        edit = build_breast_skin_mask(
            bgr,
            confidence=confidence,
            expand=expand,
            feather=feather,
            breast_size=breast_size,
        )
        return edit, preserve

    if strategy == "volume_restore" and edit_region in CLOTHED_VOLUME_REGIONS:
        edit = _breasts_mask(h, w, lm, body, clothes, size_scale=breast_size)
        if float(edit.max()) < 0.05:
            edit = _breasts_mask(h, w, None, body, clothes, size_scale=breast_size)

        if edit_region == "bust":
            wide = _torso_band_mask(h, w, lm, -0.03, 0.40, tissue)
            edit = _combine_or([edit, _combine_and(wide, tissue)], (h, w))

        edit = _subtract_locks(
            edit,
            seg,
            lm,
            lock_face=lock_face,
            lock_hair=lock_hair,
            lock_background=lock_background,
            lock_arms=lock_arms,
            lock_legs=lock_legs,
        )

        if restore_clothes:
            chest_band = _torso_band_mask(h, w, lm, -0.04, 0.44, tissue)
            if float(chest_band.max()) < 0.05:
                ys, xs = np.where(tissue > 0.25)
                if ys.size > 32:
                    chest_band = np.zeros((h, w), dtype=np.float32)
                    y_top = int(ys.min())
                    y_bot = int(ys.min() + (ys.max() - ys.min()) * 0.44)
                    chest_band[y_top:y_bot, :] = 1.0
                    chest_band = _combine_and(chest_band, tissue)
            preserve = _combine_and(clothes, chest_band)
            preserve = _combine_and(preserve, _grow(edit, max(6, expand // 2)))

    elif strategy in MASK_MODES:
        edit = build_edit_mask(
            bgr,
            edit_region,
            strategy,
            prompt_extra=prompt_extra,
            threshold=threshold,
            confidence=confidence,
            expand=0,
            feather=0,
            breast_size=breast_size,
            lock_face=lock_face,
            lock_hair=lock_hair,
            lock_clothes=False,
            lock_background=lock_background,
            lock_arms=lock_arms,
            lock_legs=lock_legs,
        )
        if restore_clothes and edit_region in CLOTHED_VOLUME_REGIONS:
            chest_band = _torso_band_mask(h, w, lm, -0.04, 0.44, tissue)
            preserve = _combine_and(clothes, chest_band)
            preserve = _combine_and(preserve, _grow(edit, max(4, expand // 2)))
    else:
        raise ValueError(f"Unknown morpho strategy: {strategy!r}")

    edit = (edit > 0.20).astype(np.float32)
    edit = _grow(edit, expand)
    edit = _feather(edit, feather)
    edit = np.clip(edit, 0.0, 1.0)

    if float(preserve.max()) > 0.02:
        preserve = (preserve > 0.22).astype(np.float32)
        preserve = _feather(preserve, max(3, feather // 4))
        preserve = np.clip(preserve, 0.0, 1.0)

    return edit, preserve


def preserve_preview(bgr: np.ndarray, preserve: np.ndarray) -> np.ndarray:
    import cv2

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tint = np.zeros_like(rgb)
    tint[..., 2] = 1.0
    alpha = preserve[..., None] * 0.55
    return rgb * (1.0 - alpha) + tint * alpha


def _feather_mask_tensor(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    import cv2

    out = []
    arr = mask.detach().cpu().numpy().astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    k = radius * 2 + 1
    for i in range(arr.shape[0]):
        out.append(cv2.GaussianBlur(arr[i], (k, k), 0))
    return torch.from_numpy(np.stack(out, axis=0)).float()


def composite_inside_mask(
    original: torch.Tensor,
    edited: torch.Tensor,
    edit_mask: torch.Tensor,
    feather: int = 6,
) -> torch.Tensor:
    """Keep edited pixels ONLY inside edit_mask; everything else = original (pixel-perfect)."""
    em = edit_mask
    if em.shape[0] != edited.shape[0]:
        em = em[0:1].repeat(edited.shape[0], 1, 1)
    em = _feather_mask_tensor(em, feather).to(edited.device)
    if em.shape[-2] != edited.shape[1] or em.shape[-1] != edited.shape[2]:
        em = torch.nn.functional.interpolate(
            em.unsqueeze(1), size=(edited.shape[1], edited.shape[2]), mode="bilinear", align_corners=False
        ).squeeze(1)
    p = em.unsqueeze(-1).clamp(0.0, 1.0)
    return original * (1.0 - p) + edited * p


def composite_preserve(
    original: torch.Tensor,
    edited: torch.Tensor,
    preserve_mask: torch.Tensor,
) -> torch.Tensor:
    """Paste original pixels where preserve_mask is white (clothing/face lock).

    Robust to size mismatch: the original image and the mask are both resized to
    the edited image's resolution (so it works after a 2x upscale)."""
    if float(preserve_mask.max()) < 0.02:
        return edited

    if original.shape[1] != edited.shape[1] or original.shape[2] != edited.shape[2]:
        original = torch.nn.functional.interpolate(
            original.permute(0, 3, 1, 2),
            size=(edited.shape[1], edited.shape[2]),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
    if original.shape[0] != edited.shape[0] and original.shape[0] == 1:
        original = original.repeat(edited.shape[0], 1, 1, 1)

    if preserve_mask.shape[0] != edited.shape[0]:
        if preserve_mask.shape[0] == 1:
            preserve_mask = preserve_mask.repeat(edited.shape[0], 1, 1)
        else:
            raise ValueError("preserve_mask batch must match image")

    p = preserve_mask.reshape(preserve_mask.shape[0], preserve_mask.shape[1], preserve_mask.shape[2], 1)
    if p.shape[1] != edited.shape[1] or p.shape[2] != edited.shape[2]:
        p = torch.nn.functional.interpolate(
            p.permute(0, 3, 1, 2),
            size=(edited.shape[1], edited.shape[2]),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)

    return edited * (1.0 - p) + original * p


def run_sdxl_inpaint(
    image: torch.Tensor,
    edit_mask: torch.Tensor,
    model,
    vae,
    positive,
    negative,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    denoise: float,
) -> torch.Tensor:
    from nodes import InpaintModelConditioning, common_ksampler

    positive, negative, latent = InpaintModelConditioning().encode(
        positive,
        negative,
        image,
        vae,
        edit_mask,
        noise_mask=True,
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
