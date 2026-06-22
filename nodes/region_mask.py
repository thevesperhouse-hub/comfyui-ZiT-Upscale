"""Anatomical region masks for localized SDXL edits (MediaPipe segment + pose geometry)."""
from __future__ import annotations

import logging
import os
import urllib.request
from functools import lru_cache
from typing import Iterable

import cv2
import folder_paths
import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(folder_paths.models_dir, "zit_upscale")
SEGMENTER_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
POSE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

# MediaPipe selfie_multiclass indices
MP_BG, MP_HAIR, MP_BODY, MP_FACE, MP_CLOTHES = 0, 1, 2, 3, 4

EDIT_REGIONS = [
    "breasts",
    "bust",
    "waist",
    "belly",
    "hips",
    "thighs",
    "full_torso",
    "arms",
    "legs",
    "face",
    "hair",
    "clothes",
    "background",
    "full_body_no_face",
]

REGION_DENOISE_DEFAULTS = {
    "breasts": 0.68,
    "bust": 0.70,
    "waist": 0.62,
    "belly": 0.65,
    "hips": 0.68,
    "thighs": 0.65,
    "full_torso": 0.72,
    "arms": 0.58,
    "legs": 0.62,
    "face": 0.45,
    "hair": 0.50,
    "clothes": 0.55,
    "background": 0.60,
    "full_body_no_face": 0.75,
}


def _download(url: str, dest: str) -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.isfile(dest):
        logger.info("ZiT-Upscale: downloading %s", os.path.basename(dest))
        urllib.request.urlretrieve(url, dest)
    return dest


def _tensor_to_rgb_uint8(image: torch.Tensor) -> np.ndarray:
    arr = (255.0 * image.detach().cpu().numpy()).clip(0, 255).astype(np.uint8)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _mask_to_tensor(mask: np.ndarray, batch: int) -> torch.Tensor:
    m = mask.astype(np.float32)
    if m.max() > 1.0:
        m = m / 255.0
    t = torch.from_numpy(m)[None, ...]
    if batch > 1:
        t = t.repeat(batch, 1, 1)
    return t


def _feather(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    k = radius * 2 + 1
    return cv2.GaussianBlur(mask, (k, k), 0)


def _grow(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    k = pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel, iterations=1)


def _combine_or(masks: Iterable[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    for m in masks:
        if m is None:
            continue
        out = np.maximum(out, m.astype(np.float32))
    return out


def _combine_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(a.astype(np.float32), b.astype(np.float32))


def _y_band(h: int, y0: float, y1: float) -> np.ndarray:
    band = np.zeros((h, 1), dtype=np.float32)
    top = int(max(0, min(h - 1, y0 * h)))
    bot = int(max(0, min(h, y1 * h)))
    if bot <= top:
        bot = min(h, top + 1)
    band[top:bot, 0] = 1.0
    return np.repeat(band, 1, axis=1)


def _landmark_y(lm, idx: int) -> float | None:
    if lm is None or len(lm) <= idx:
        return None
    return float(lm[idx].y)


def _landmark_x(lm, idx: int) -> float | None:
    if lm is None or len(lm) <= idx:
        return None
    return float(lm[idx].x)


def _landmark_vis(lm, idx: int) -> float:
    if lm is None or len(lm) <= idx or not hasattr(lm[idx], "visibility"):
        return 1.0
    return float(lm[idx].visibility)


def _pose_frame(h: int, w: int, lm):
    """Shoulder/hip frame in pixel coords. Returns None if pose is unusable."""
    if lm is None:
        return None
    needed = (11, 12, 23, 24)
    if any(_landmark_y(lm, i) is None or _landmark_x(lm, i) is None for i in needed):
        return None
    if any(_landmark_vis(lm, i) < 0.35 for i in needed):
        return None

    ls = np.array([lm[11].x * w, lm[11].y * h], dtype=np.float32)
    rs = np.array([lm[12].x * w, lm[12].y * h], dtype=np.float32)
    lh = np.array([lm[23].x * w, lm[23].y * h], dtype=np.float32)
    rh = np.array([lm[24].x * w, lm[24].y * h], dtype=np.float32)

    shoulder_mid = (ls + rs) / 2.0
    hip_mid = (lh + rh) / 2.0
    shoulder_w = max(float(np.linalg.norm(rs - ls)), w * 0.12)
    torso_h = max(float(hip_mid[1] - shoulder_mid[1]), h * 0.12)
    return {
        "shoulder_mid": shoulder_mid,
        "shoulder_w": shoulder_w,
        "torso_h": torso_h,
    }


def _ellipse_mask(h: int, w: int, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    if rx < 2 or ry < 2:
        return mask
    cv2.ellipse(
        mask,
        (int(round(cx)), int(round(cy))),
        (int(round(rx)), int(round(ry))),
        0,
        0,
        360,
        1.0,
        -1,
    )
    return mask


def _breasts_mask_from_segment(
    h: int,
    w: int,
    body: np.ndarray,
    clothes: np.ndarray,
    *,
    size_scale: float = 1.0,
) -> np.ndarray:
    """Close-up / no pose: dual ellipses from person bbox (body + clothes)."""
    tissue = np.clip(body + clothes, 0.0, 1.0)
    ys, xs = np.where(tissue > 0.25)
    if ys.size < 48:
        return np.zeros((h, w), dtype=np.float32)

    y_top = float(ys.min())
    y_bot = float(ys.min() + (ys.max() - ys.min()) * 0.46)
    x_left = float(xs.min())
    x_right = float(xs.max())
    cx = (x_left + x_right) * 0.5
    sw = max(x_right - x_left, w * 0.18)
    th = max(y_bot - y_top, h * 0.10)

    cy = y_top + th * 0.42
    rx = sw * 0.21 * size_scale
    ry = th * 0.38 * size_scale
    offset = sw * 0.20

    left = _ellipse_mask(h, w, cx - offset, cy, rx, ry)
    right = _ellipse_mask(h, w, cx + offset, cy, rx, ry)
    geo = np.clip(left + right, 0.0, 1.0)
    sternum = _ellipse_mask(h, w, cx, cy + th * 0.04, sw * 0.08, th * 0.22)
    geo = np.clip(geo - sternum * 0.85, 0.0, 1.0)
    return _combine_and(geo, tissue)


def _breasts_mask(
    h: int,
    w: int,
    lm,
    body: np.ndarray,
    clothes: np.ndarray,
    *,
    size_scale: float = 1.0,
) -> np.ndarray:
    """Dual-ellipse chest mask: skin + fabric over each breast."""
    frame = _pose_frame(h, w, lm)
    if frame is None:
        return _breasts_mask_from_segment(h, w, body, clothes, size_scale=size_scale)

    sm = frame["shoulder_mid"]
    sw = frame["shoulder_w"]
    th = frame["torso_h"]

    cy = sm[1] + th * 0.17
    rx = sw * 0.22 * size_scale
    ry = th * 0.17 * size_scale
    offset = sw * 0.19

    left = _ellipse_mask(h, w, sm[0] - offset, cy, rx, ry)
    right = _ellipse_mask(h, w, sm[0] + offset, cy, rx, ry)
    geo = np.clip(left + right, 0.0, 1.0)

    # Sternum strip: avoid editing the flat center chest.
    sternum = _ellipse_mask(h, w, sm[0], cy + th * 0.02, sw * 0.07, th * 0.16)
    geo = np.clip(geo - sternum * 0.85, 0.0, 1.0)

    # Skin where visible + fabric over breasts when covered.
    tissue = np.clip(body + clothes, 0.0, 1.0)
    edit = _combine_and(geo, tissue)

    # Drop arm overlap using elbow-shoulder lines.
    for shoulder_idx, elbow_idx in ((11, 13), (12, 14)):
        if _landmark_y(lm, elbow_idx) is None:
            continue
        p0 = (int(lm[shoulder_idx].x * w), int(lm[shoulder_idx].y * h))
        p1 = (int(lm[elbow_idx].x * w), int(lm[elbow_idx].y * h))
        arm = np.zeros((h, w), dtype=np.float32)
        cv2.line(arm, p0, p1, 1.0, thickness=max(10, int(sw * 0.14)))
        arm = _grow(arm, max(4, w // 120))
        edit = np.clip(edit - arm, 0.0, 1.0)

    return edit


@lru_cache(maxsize=1)
def _segmenter_model_buffer() -> bytes:
    path = _download(SEGMENTER_URL, os.path.join(MODEL_DIR, "selfie_multiclass_256x256.tflite"))
    with open(path, "rb") as f:
        return f.read()


@lru_cache(maxsize=1)
def _pose_model_path() -> str:
    return _download(POSE_URL, os.path.join(MODEL_DIR, "pose_landmarker_full.task"))


def _mp_image(bgr: np.ndarray, mp):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _segment_masks(bgr: np.ndarray, confidence: float) -> dict[str, np.ndarray]:
    import mediapipe as mp

    h, w = bgr.shape[:2]
    opts = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=_segmenter_model_buffer()),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_category_mask=True,
    )
    mp_img = _mp_image(bgr, mp)
    with mp.tasks.vision.ImageSegmenter.create_from_options(opts) as segmenter:
        result = segmenter.segment(mp_img)

    def _cat(idx: int) -> np.ndarray:
        m = result.confidence_masks[idx].numpy_view()
        if m.ndim == 3:
            m = m.squeeze(-1)
        return (m > confidence).astype(np.float32)

    return {
        "background": _cat(MP_BG),
        "hair": _cat(MP_HAIR),
        "body": _cat(MP_BODY),
        "face": _cat(MP_FACE),
        "clothes": _cat(MP_CLOTHES),
        "shape": (h, w),
    }


def _pose_landmarks(bgr: np.ndarray):
    import mediapipe as mp

    opts = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=_pose_model_path()),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
    )
    mp_img = _mp_image(bgr, mp)
    with mp.tasks.vision.PoseLandmarker.create_from_options(opts) as landmarker:
        result = landmarker.detect(mp_img)
        if not result.pose_landmarks:
            return None
        return result.pose_landmarks[0]


def _torso_band_mask(h: int, w: int, lm, y_start: float, y_end: float, body: np.ndarray) -> np.ndarray:
    if lm is None:
        return np.zeros((h, w), dtype=np.float32)

    ls, rs = _landmark_y(lm, 11), _landmark_y(lm, 12)
    lh, rh = _landmark_y(lm, 23), _landmark_y(lm, 24)
    if any(v is None for v in (ls, rs, lh, rh)):
        return np.zeros((h, w), dtype=np.float32)

    shoulder_y = (ls + rs) / 2.0
    hip_y = (lh + rh) / 2.0
    torso = max(0.08, hip_y - shoulder_y)

    y0 = shoulder_y + y_start * torso
    y1 = shoulder_y + y_end * torso
    band = _y_band(h, y0, y1)
    band = np.repeat(band, w, axis=1)
    return _combine_and(band, body)


def _arms_mask(h: int, w: int, lm, body: np.ndarray) -> np.ndarray:
    if lm is None:
        return np.zeros((h, w), dtype=np.float32)
    out = np.zeros((h, w), dtype=np.float32)
    for elbow, wrist in ((13, 15), (14, 16)):
        pts = []
        for idx in (11 if elbow == 13 else 12, elbow, wrist):
            y, x = _landmark_y(lm, idx), None
            if y is None:
                continue
            x = float(lm[idx].x)
            pts.append((int(x * w), int(y * h)))
        if len(pts) >= 2:
            cv2.line(out, pts[0], pts[-1], 1.0, thickness=max(8, w // 40))
    out = _grow(out, max(4, w // 80))
    return _combine_and(out, body)


def _legs_mask(h: int, w: int, lm, body: np.ndarray) -> np.ndarray:
    if lm is None:
        return np.zeros((h, w), dtype=np.float32)
    lh, rh = _landmark_y(lm, 23), _landmark_y(lm, 24)
    lk, rk = _landmark_y(lm, 25), _landmark_y(lm, 26)
    if lh is None or rh is None:
        return np.zeros((h, w), dtype=np.float32)
    hip_y = (lh + rh) / 2.0
    knee_y = (lk + rk) / 2.0 if lk is not None and rk is not None else min(1.0, hip_y + 0.25)
    band = _y_band(h, hip_y - 0.02, knee_y + 0.15)
    band = np.repeat(band, w, axis=1)
    leg_body = _combine_and(band, body)
    if lk is None:
        return leg_body
    out = np.zeros((h, w), dtype=np.float32)
    for hip, knee, ankle in ((23, 25, 27), (24, 26, 28)):
        pts = []
        for idx in (hip, knee, ankle):
            if _landmark_y(lm, idx) is None:
                continue
            pts.append((int(lm[idx].x * w), int(lm[idx].y * h)))
        if len(pts) >= 2:
            cv2.line(out, pts[0], pts[-1], 1.0, thickness=max(10, w // 35))
    out = _grow(out, max(6, w // 60))
    return _combine_and(np.maximum(out, leg_body * 0.5), np.maximum(body, band))


def _thighs_mask(h: int, w: int, lm, body: np.ndarray) -> np.ndarray:
    if lm is None:
        return np.zeros((h, w), dtype=np.float32)
    lh, rh = _landmark_y(lm, 23), _landmark_y(lm, 24)
    lk, rk = _landmark_y(lm, 25), _landmark_y(lm, 26)
    if any(v is None for v in (lh, rh, lk, rk)):
        return np.zeros((h, w), dtype=np.float32)
    hip_y = (lh + rh) / 2.0
    knee_y = (lk + rk) / 2.0
    band = _y_band(h, hip_y - 0.01, knee_y + 0.02)
    band = np.repeat(band, w, axis=1)
    return _combine_and(band, body)


def build_region_mask(
    bgr: np.ndarray,
    edit_region: str,
    *,
    confidence: float = 0.4,
    lock_face: bool = True,
    lock_hair: bool = True,
    lock_clothes: bool = True,
    lock_background: bool = True,
    lock_arms: bool = False,
    lock_legs: bool = False,
    expand: int = 6,
    feather: int = 16,
    breast_size: float = 1.0,
) -> np.ndarray:
    seg = _segment_masks(bgr, confidence)
    h, w = seg["shape"]
    lm = _pose_landmarks(bgr)
    body = seg["body"]

    if edit_region == "face":
        edit = seg["face"]
    elif edit_region == "hair":
        edit = seg["hair"]
    elif edit_region == "clothes":
        edit = seg["clothes"]
    elif edit_region == "background":
        edit = seg["background"]
    elif edit_region == "breasts":
        edit = _breasts_mask(h, w, lm, body, seg["clothes"], size_scale=breast_size)
    elif edit_region == "bust":
        edit = _breasts_mask(h, w, lm, body, seg["clothes"], size_scale=breast_size * 1.15)
        wide = _torso_band_mask(h, w, lm, -0.02, 0.34, body)
        edit = _combine_or([edit, _combine_and(wide, seg["clothes"])], (h, w))
    elif edit_region == "waist":
        edit = _torso_band_mask(h, w, lm, 0.28, 0.46, body)
    elif edit_region == "belly":
        edit = _torso_band_mask(h, w, lm, 0.44, 0.66, body)
    elif edit_region == "hips":
        edit = _torso_band_mask(h, w, lm, 0.62, 1.02, body)
    elif edit_region == "thighs":
        edit = _thighs_mask(h, w, lm, body)
    elif edit_region == "full_torso":
        edit = _torso_band_mask(h, w, lm, -0.03, 1.02, body)
    elif edit_region == "arms":
        edit = _arms_mask(h, w, lm, body)
    elif edit_region == "legs":
        edit = _legs_mask(h, w, lm, body)
    elif edit_region == "full_body_no_face":
        edit = _combine_or([body, seg["clothes"]], (h, w))
        edit = _combine_and(edit, 1.0 - seg["face"])
    else:
        edit = body

    locks = []
    if lock_face:
        locks.append(seg["face"])
    if lock_hair:
        locks.append(seg["hair"])
    if lock_clothes:
        locks.append(seg["clothes"])
    if lock_background:
        locks.append(seg["background"])
    if lock_arms:
        locks.append(_arms_mask(h, w, lm, body))
    if lock_legs:
        locks.append(_legs_mask(h, w, lm, body))

    if locks:
        locked = _combine_or(locks, (h, w))
        edit = _combine_and(edit, 1.0 - locked)

    edit = (edit > 0.25).astype(np.float32)
    edit = _grow(edit, expand)
    edit = _feather(edit, feather)
    return np.clip(edit, 0.0, 1.0)


def build_breast_skin_mask(
    bgr: np.ndarray,
    *,
    confidence: float = 0.40,
    expand: int = 8,
    feather: int = 14,
    breast_size: float = 1.0,
) -> np.ndarray:
    """Chest SKIN only — bikini/clothes pixels are never in the mask."""
    seg = _segment_masks(bgr, confidence)
    h, w = seg["shape"]
    lm = _pose_landmarks(bgr)
    body = seg["body"]
    clothes = seg["clothes"]

    geo = _breasts_mask(h, w, lm, body, np.zeros_like(clothes), size_scale=breast_size)
    band = _torso_band_mask(h, w, lm, 0.02, 0.44, body)
    edit = _combine_or([geo, band], (h, w))
    edit = _combine_and(edit, body)
    edit = _combine_and(edit, 1.0 - clothes)
    edit = _combine_and(edit, 1.0 - seg["face"])
    edit = _combine_and(edit, 1.0 - seg["hair"])
    edit = _combine_and(edit, 1.0 - seg["background"])

    if float(edit.max()) < 0.02:
        ys, xs = np.where(body > 0.25)
        if ys.size > 48:
            y_top = int(ys.min())
            y_bot = int(ys.min() + (ys.max() - ys.min()) * 0.42)
            x_left = int(xs.min() + (xs.max() - xs.min()) * 0.06)
            x_right = int(xs.max() - (xs.max() - xs.min()) * 0.06)
            blob = np.zeros((h, w), dtype=np.float32)
            blob[y_top:y_bot, x_left:x_right] = 1.0
            edit = _combine_and(blob, body)
            edit = _combine_and(edit, 1.0 - clothes)
            edit = _combine_and(edit, 1.0 - seg["face"])

    edit = (edit > 0.22).astype(np.float32)
    edit = _grow(edit, expand)
    edit = _feather(edit, feather)
    return np.clip(edit, 0.0, 1.0)


class ZiTBreastSkinMask:
    """Morpho seins SDXL: masque = PEAU poitrine, le bikini reste figé."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "confidence": ("FLOAT", {"default": 0.40, "min": 0.05, "max": 0.95, "step": 0.01}),
                "expand": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
                "feather": ("INT", {"default": 14, "min": 0, "max": 64, "step": 1}),
                "breast_size": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.65,
                        "max": 1.50,
                        "step": 0.05,
                        "tooltip": "Agrandit la zone peau si le preview rate du volume.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("edit_mask", "preview")
    FUNCTION = "build"
    CATEGORY = "ZiT-Upscale/Edit"

    def build(self, image, confidence, expand, feather, breast_size):
        try:
            import mediapipe  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("ZiTBreastSkinMask needs mediapipe. Install: pip install mediapipe") from exc

        batch = image.shape[0]
        masks, previews = [], []
        for i in range(batch):
            bgr = _tensor_to_rgb_uint8(image[i : i + 1])
            mask = build_breast_skin_mask(
                bgr,
                confidence=confidence,
                expand=expand,
                feather=feather,
                breast_size=breast_size,
            )
            if float(mask.max()) < 0.02:
                raise RuntimeError(
                    "Breast skin mask empty. Peins le masque LoadImage (peau sous/sur les seins) "
                    "et branche-le via BitwiseAnd sur le workflow manuel."
                )
            masks.append(mask)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            red = np.zeros_like(rgb)
            red[..., 0] = 1.0
            alpha = mask[..., None] * 0.55
            previews.append(rgb * (1.0 - alpha) + red * alpha)

        return (_mask_to_tensor(np.stack(masks, axis=0), batch), torch.from_numpy(np.stack(previews, axis=0)).float())


class ZiTRegionMask:
    """Pick a body/background region; optional locks for everything else."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "edit_region": (EDIT_REGIONS, {"default": "breasts"}),
                "confidence": ("FLOAT", {"default": 0.40, "min": 0.05, "max": 0.95, "step": 0.01}),
                "expand": ("INT", {"default": 10, "min": 0, "max": 64, "step": 1}),
                "feather": ("INT", {"default": 14, "min": 0, "max": 64, "step": 1}),
                "breast_size": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.65,
                        "max": 1.45,
                        "step": 0.05,
                        "tooltip": "Scale breast ellipses if preview misses (only for breasts/bust).",
                    },
                ),
            },
            "optional": {
                "lock_face": ("BOOLEAN", {"default": True}),
                "lock_hair": ("BOOLEAN", {"default": True}),
                "lock_clothes": ("BOOLEAN", {"default": True}),
                "lock_background": ("BOOLEAN", {"default": True}),
                "lock_arms": ("BOOLEAN", {"default": False}),
                "lock_legs": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("edit_mask", "preview")
    FUNCTION = "build"
    CATEGORY = "ZiT-Upscale/Edit"

    def build(
        self,
        image,
        edit_region,
        confidence,
        expand,
        feather,
        lock_face=True,
        lock_hair=True,
        lock_clothes=True,
        lock_background=True,
        lock_arms=False,
        lock_legs=False,
        breast_size=1.0,
    ):
        try:
            import mediapipe  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "ZiTRegionMask needs mediapipe. Install: pip install mediapipe"
            ) from exc

        batch = image.shape[0]
        masks = []
        previews = []
        for i in range(batch):
            bgr = _tensor_to_rgb_uint8(image[i : i + 1])
            mask = build_region_mask(
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
            masks.append(mask)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            overlay = rgb.copy()
            red = np.zeros_like(rgb)
            red[..., 0] = 1.0
            alpha = mask[..., None] * 0.55
            blend = rgb * (1.0 - alpha) + red * alpha
            previews.append(blend)

        mask_t = _mask_to_tensor(np.stack(masks, axis=0), batch)
        preview_t = torch.from_numpy(np.stack(previews, axis=0)).float()
        return (mask_t, preview_t)


class ZiTRegionDenoiseHint:
    """Suggested denoise for the selected edit region."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"edit_region": (EDIT_REGIONS, {"default": "full_body_no_face"})}}

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("denoise",)
    FUNCTION = "hint"
    CATEGORY = "ZiT-Upscale/Edit"

    def hint(self, edit_region):
        return (REGION_DENOISE_DEFAULTS.get(edit_region, 0.65),)


NODE_CLASS_MAPPINGS = {
    "ZiTRegionMask": ZiTRegionMask,
    "ZiTRegionDenoiseHint": ZiTRegionDenoiseHint,
    "ZiTBreastSkinMask": ZiTBreastSkinMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZiTRegionMask": "ZiT Region Mask (pose + segment)",
    "ZiTRegionDenoiseHint": "ZiT Region Denoise Hint",
    "ZiTBreastSkinMask": "ZiT Breast Skin Mask (peau — pas le bikini)",
}
