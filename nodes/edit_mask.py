"""Hybrid region masks: pose ROI + CLIPSeg + geometry fallback."""
from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
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
        build_region_mask,
    )
    from .semantic_mask import PRESET_PROMPTS, _clipseg_mask, _resolve_edit_prompt, _tensor_to_pil
except ImportError:
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
        build_region_mask,
    )
    from semantic_mask import PRESET_PROMPTS, _clipseg_mask, _resolve_edit_prompt, _tensor_to_pil

MASK_MODES = ["pose", "hybrid", "clipseg"]

BODY_SURFACE = {
    "breasts",
    "bust",
    "waist",
    "belly",
    "hips",
    "thighs",
    "full_torso",
    "arms",
    "legs",
    "full_body_no_face",
}

REGION_TO_CLIP_PRESET = {
    "breasts": "breasts",
    "bust": "breasts",
    "waist": "waist",
    "belly": "belly",
    "hips": "hips",
    "thighs": "thighs",
    "full_torso": "torso",
    "arms": "custom",
    "legs": "custom",
    "face": "face",
    "hair": "hair",
    "clothes": "clothes",
    "background": "background",
    "full_body_no_face": "torso",
}


def default_denoise(edit_region: str) -> float:
    return float(REGION_DENOISE_DEFAULTS.get(edit_region, 0.68))


def _chest_tissue_fallback(h: int, w: int, body: np.ndarray, clothes: np.ndarray) -> np.ndarray:
    """When pose fails: upper torso from person segment (no CLIPSeg)."""
    tissue = np.clip(body + clothes, 0.0, 1.0)
    ys, xs = np.where(tissue > 0.25)
    if ys.size < 32:
        return np.zeros((h, w), dtype=np.float32)
    y_top = int(ys.min())
    y_bot = int(ys.min() + (ys.max() - ys.min()) * 0.38)
    x_left = max(0, int(xs.min() + (xs.max() - xs.min()) * 0.08))
    x_right = min(w, int(xs.max() - (xs.max() - xs.min()) * 0.08))
    band = np.zeros((h, w), dtype=np.float32)
    band[y_top:y_bot, x_left:x_right] = 1.0
    return _combine_and(band, tissue)


def _clip_agrees_with_geo(clip: np.ndarray, geo: np.ndarray) -> bool:
    """Reject CLIPSeg when it chases text/logos instead of anatomy."""
    clip_b = (clip > 0.25).astype(np.float32)
    geo_b = (geo > 0.25).astype(np.float32)
    geo_a = float(geo_b.sum())
    if geo_a < 16:
        return False
    inter = float((clip_b * geo_b).sum())
    clip_a = float(clip_b.sum())
    if clip_a < 1:
        return False
    overlap = inter / (geo_a + 1e-6)
    ratio = clip_a / (geo_a + 1e-6)
    # Text on clothes: high CLIP response, low overlap with chest ellipses.
    return overlap >= 0.12 and 0.25 <= ratio <= 1.8


def _breasts_mask_hybrid(
    bgr: np.ndarray,
    edit_region: str,
    seg: dict,
    lm,
    *,
    prompt_extra: str,
    threshold: float,
    breast_size: float,
) -> np.ndarray:
    h, w = seg["shape"]
    tissue = np.clip(seg["body"] + seg["clothes"], 0.0, 1.0)
    roi = _pose_roi(bgr, edit_region, seg, lm)

    geo = _breasts_mask(h, w, lm, seg["body"], seg["clothes"], size_scale=breast_size)
    if edit_region == "bust":
        wide = _torso_band_mask(h, w, lm, -0.02, 0.34, tissue)
        geo = _combine_or([geo, _combine_and(wide, seg["clothes"])], (h, w))

    if float(geo.max()) < 0.05:
        geo = _breasts_mask(h, w, None, seg["body"], seg["clothes"], size_scale=breast_size)
        if roi is not None:
            geo = _combine_and(geo, roi)

    clip = _clipseg_region_mask(bgr, edit_region, prompt_extra, threshold, roi)
    if _clip_agrees_with_geo(clip, geo):
        edit = np.maximum(geo, _combine_and(clip, geo))
    else:
        edit = geo

    edit = _combine_and(edit, tissue)
    ref_roi = roi if roi is not None and float(roi.max()) > 0.05 else geo
    edit = _filter_breast_blobs(edit, ref_roi, h, w)
    return edit


def _filter_breast_blobs(mask: np.ndarray, roi: np.ndarray, h: int, w: int) -> np.ndarray:
    """Keep the 1–2 largest blobs in the chest ROI (left/right breast)."""
    binary = ((mask > 0.22) & (roi > 0.15)).astype(np.uint8)
    if binary.max() == 0:
        return mask

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return mask

    roi_ys = np.where(roi > 0.15)[0]
    y_min = int(roi_ys.min()) if roi_ys.size else 0
    y_max = int(roi_ys.max()) if roi_ys.size else h
    min_area = max(48, (h * w) // 8000)

    candidates = []
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cy = centroids[idx][1]
        if cy < y_min - 4 or cy > y_max + 8:
            continue
        candidates.append((area, idx))

    if not candidates:
        return mask

    candidates.sort(reverse=True)
    keep = {idx for _, idx in candidates[:2]}
    filtered = np.zeros((h, w), dtype=np.float32)
    for idx in keep:
        filtered[labels == idx] = 1.0
    return np.maximum(mask * 0.35, filtered)


def _clipseg_region_mask(
    bgr: np.ndarray,
    edit_region: str,
    prompt_extra: str,
    threshold: float,
    roi: np.ndarray | None,
) -> np.ndarray:
    preset = REGION_TO_CLIP_PRESET.get(edit_region, "custom")
    extra = prompt_extra
    if edit_region == "arms" and not extra.strip():
        extra = "woman's arms, upper arms"
    elif edit_region == "legs" and not extra.strip():
        extra = "woman's legs, thighs, calves"

    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    prompts = _resolve_edit_prompt(preset, extra)
    if not prompts:
        base = PRESET_PROMPTS.get(preset, "woman's body")
        prompts = [p.strip() for p in base.split(",") if p.strip()][:4]

    clip = _clipseg_mask(pil, prompts, threshold)
    if roi is not None:
        clip = _combine_and(clip, roi)
    return clip


def _pose_roi(bgr: np.ndarray, edit_region: str, seg: dict, lm) -> np.ndarray | None:
    h, w = seg["shape"]
    tissue = np.clip(seg["body"] + seg["clothes"], 0.0, 1.0)

    bands = {
        "breasts": (0.02, 0.36),
        "bust": (0.00, 0.42),
        "waist": (0.26, 0.48),
        "belly": (0.40, 0.66),
        "hips": (0.58, 1.02),
        "thighs": (0.55, 0.92),
        "full_torso": (-0.03, 1.02),
    }
    if edit_region in bands:
        y0, y1 = bands[edit_region]
        return _torso_band_mask(h, w, lm, y0, y1, tissue)
    return None


def _hybrid_mask(
    bgr: np.ndarray,
    edit_region: str,
    *,
    prompt_extra: str,
    threshold: float,
    confidence: float,
    breast_size: float,
) -> np.ndarray:
    seg = _segment_masks(bgr, confidence)
    h, w = seg["shape"]
    lm = _pose_landmarks(bgr)
    tissue = np.clip(seg["body"] + seg["clothes"], 0.0, 1.0)
    roi = _pose_roi(bgr, edit_region, seg, lm)

    if edit_region in ("breasts", "bust"):
        return _breasts_mask_hybrid(
            bgr,
            edit_region,
            seg,
            lm,
            prompt_extra=prompt_extra,
            threshold=threshold,
            breast_size=breast_size,
        )

    if edit_region in BODY_SURFACE and roi is not None:
        clip = _clipseg_region_mask(bgr, edit_region, prompt_extra, threshold, roi)
        pose = build_region_mask(
            bgr,
            edit_region,
            confidence=confidence,
            lock_face=False,
            lock_hair=False,
            lock_clothes=False,
            lock_background=False,
            lock_arms=False,
            lock_legs=False,
            expand=0,
            feather=0,
            breast_size=breast_size,
        )
        if _clip_agrees_with_geo(clip, pose):
            edit = np.maximum(pose, _combine_and(clip, pose))
        else:
            edit = pose
        edit = _combine_and(edit, tissue if edit_region != "background" else 1.0)
        return edit

    if edit_region in ("face", "hair", "clothes", "background"):
        key = edit_region
        return seg[key].astype(np.float32)

    clip = _clipseg_region_mask(bgr, edit_region, prompt_extra, threshold, roi)
    if float(clip.max()) > 0.05:
        return clip
    return build_region_mask(
        bgr,
        edit_region,
        confidence=confidence,
        lock_face=False,
        lock_hair=False,
        lock_clothes=False,
        lock_background=False,
        expand=0,
        feather=0,
        breast_size=breast_size,
    )


def _apply_locks(
    edit: np.ndarray,
    bgr: np.ndarray,
    edit_region: str,
    *,
    confidence: float,
    lock_face: bool,
    lock_hair: bool,
    lock_clothes: bool,
    lock_background: bool,
    lock_arms: bool,
    lock_legs: bool,
) -> np.ndarray:
    seg = _segment_masks(bgr, confidence)
    h, w = seg["shape"]
    lm = _pose_landmarks(bgr)
    body = seg["body"]

    locks = []
    if lock_face and edit_region != "face":
        locks.append(seg["face"])
    if lock_hair and edit_region != "hair":
        locks.append(seg["hair"])
    if lock_clothes and edit_region != "clothes":
        locks.append(seg["clothes"])
    if lock_background and edit_region != "background":
        locks.append(seg["background"])
    if lock_arms and edit_region not in ("arms", "full_body_no_face"):
        locks.append(_arms_mask(h, w, lm, body))
    if lock_legs and edit_region not in ("legs", "thighs", "full_body_no_face"):
        locks.append(_legs_mask(h, w, lm, body))

    if locks:
        locked = _combine_or(locks, (h, w))
        edit = _combine_and(edit, 1.0 - locked)
    return edit


def build_edit_mask(
    bgr: np.ndarray,
    edit_region: str,
    mask_mode: str,
    *,
    prompt_extra: str = "",
    threshold: float = 0.30,
    confidence: float = 0.40,
    expand: int = 10,
    feather: int = 14,
    breast_size: float = 1.0,
    lock_face: bool = True,
    lock_hair: bool = True,
    lock_clothes: bool = False,
    lock_background: bool = True,
    lock_arms: bool = True,
    lock_legs: bool = False,
) -> np.ndarray:
    if mask_mode == "pose":
        edit = build_region_mask(
            bgr,
            edit_region,
            confidence=confidence,
            lock_face=lock_face,
            lock_hair=lock_hair,
            lock_clothes=lock_clothes,
            lock_background=lock_background,
            lock_arms=lock_arms,
            lock_legs=lock_legs,
            expand=expand,
            feather=feather,
            breast_size=breast_size,
        )
        return edit

    if mask_mode == "clipseg":
        roi = None
        if edit_region in BODY_SURFACE:
            seg = _segment_masks(bgr, confidence)
            lm = _pose_landmarks(bgr)
            roi = _pose_roi(bgr, edit_region, seg, lm)
        edit = _clipseg_region_mask(bgr, edit_region, prompt_extra, threshold, roi)
        edit = _apply_locks(
            edit,
            bgr,
            edit_region,
            confidence=confidence,
            lock_face=lock_face,
            lock_hair=lock_hair,
            lock_clothes=lock_clothes,
            lock_background=lock_background,
            lock_arms=lock_arms,
            lock_legs=lock_legs,
        )
    else:
        edit = _hybrid_mask(
            bgr,
            edit_region,
            prompt_extra=prompt_extra,
            threshold=threshold,
            confidence=confidence,
            breast_size=breast_size,
        )
        edit = _apply_locks(
            edit,
            bgr,
            edit_region,
            confidence=confidence,
            lock_face=lock_face,
            lock_hair=lock_hair,
            lock_clothes=lock_clothes,
            lock_background=lock_background,
            lock_arms=lock_arms,
            lock_legs=lock_legs,
        )

    edit = (edit > 0.22).astype(np.float32)
    edit = _grow(edit, expand)
    edit = _feather(edit, feather)
    return np.clip(edit, 0.0, 1.0)


def mask_preview(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    red = np.zeros_like(rgb)
    red[..., 0] = 1.0
    alpha = mask[..., None] * 0.55
    return rgb * (1.0 - alpha) + red * alpha
