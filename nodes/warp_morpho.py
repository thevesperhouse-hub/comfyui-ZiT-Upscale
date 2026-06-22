"""Geometric morpho via radial pixel warp (Liquify-style). No regeneration → keeps clothes/print/face intact."""
from __future__ import annotations

import logging

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    from .region_mask import _pose_frame, _pose_landmarks, _segment_masks, _tensor_to_rgb_uint8
except ImportError:
    from region_mask import _pose_frame, _pose_landmarks, _segment_masks, _tensor_to_rgb_uint8


WARP_MODES = ["from_mask", "auto_breasts", "manual_two_points", "manual_one_point"]


def _bulge_remap(
    h: int,
    w: int,
    centers_radii: list[tuple[float, float, float]],
    strength: float,
    aspect: float = 1.15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build cv2.remap maps for a radial bulge/pinch at each (cx, cy, radius).
    strength > 0 = enlarge (magnify outward), < 0 = shrink.
    aspect > 1 stretches the effect vertically (breasts hang/project down).
    """
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = xs.copy()
    map_y = ys.copy()

    exponent = 1.0 + float(np.clip(strength, -0.95, 1.5))

    for cx, cy, radius in centers_radii:
        radius = max(float(radius), 4.0)
        dx = xs - cx
        dy = (ys - cy) / max(aspect, 1e-3)
        r = np.sqrt(dx * dx + dy * dy)
        inside = r < radius

        norm = np.clip(r / radius, 1e-6, 1.0)
        new_norm = norm ** exponent
        # Smooth falloff so the boundary at r=radius stays continuous (identity).
        blend = (np.cos(norm * np.pi) * 0.5 + 0.5)  # 1 at center → 0 at radius
        scale = (1.0 - blend) + blend * (new_norm / norm)

        src_dx = dx * scale
        src_dy = dy * scale * max(aspect, 1e-3)
        cand_x = cx + src_dx
        cand_y = cy + src_dy

        map_x = np.where(inside, cand_x, map_x)
        map_y = np.where(inside, cand_y, map_y)

    return map_x, map_y


def _centers_from_mask(mask: np.ndarray, radius_scale: float) -> list[tuple[float, float, float]]:
    """Painted mask → one (cx, cy, radius) per blob. Radius from blob area."""
    h, w = mask.shape[:2]
    binary = (mask > 0.5).astype(np.uint8)
    if binary.max() == 0:
        return []
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = max(64, (h * w) // 4000)
    out = []
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx, cy = centroids[idx]
        # Effective radius of the blob, extended a bit for smooth falloff.
        eff = float(np.sqrt(area / np.pi)) * 1.25 * radius_scale
        out.append((float(cx), float(cy), eff))
    return out


def _auto_breast_centers(bgr: np.ndarray, confidence: float, radius_scale: float):
    h, w = bgr.shape[:2]
    lm = _pose_landmarks(bgr)
    frame = _pose_frame(h, w, lm)
    if frame is not None:
        sm = frame["shoulder_mid"]
        sw = frame["shoulder_w"]
        th = frame["torso_h"]
        cy = sm[1] + th * 0.18
        offset = sw * 0.19
        radius = sw * 0.30 * radius_scale
        return [(sm[0] - offset, cy), (sm[0] + offset, cy)], radius

    seg = _segment_masks(bgr, confidence)
    tissue = np.clip(seg["body"] + seg["clothes"], 0.0, 1.0)
    ys, xs = np.where(tissue > 0.25)
    if ys.size < 48:
        cx, cy = w * 0.5, h * 0.42
        return [(cx - w * 0.12, cy), (cx + w * 0.12, cy)], w * 0.18 * radius_scale

    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    span = x1 - x0
    cx = (x0 + x1) * 0.5
    cy = y0 + (y1 - y0) * 0.40
    offset = span * 0.20
    radius = span * 0.26 * radius_scale
    return [(cx - offset, cy), (cx + offset, cy)], radius


class ZiTWarpMorpho:
    """Liquify-style morpho. Pushes pixels to grow/shrink a region — keeps clothes, print, face pixel-exact."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (WARP_MODES, {"default": "from_mask"}),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": -0.90,
                        "max": 1.50,
                        "step": 0.01,
                        "tooltip": "> 0 agrandit, < 0 reduit. 0.25–0.5 = naturel.",
                    },
                ),
                "radius_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.3, "max": 3.0, "step": 0.05, "tooltip": "Taille de la zone affectee."},
                ),
                "vertical_stretch": (
                    "FLOAT",
                    {"default": 1.15, "min": 0.6, "max": 2.0, "step": 0.05, "tooltip": "Projette le volume vers le bas."},
                ),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Peins sur les zones a gonfler (MaskEditor). Mode from_mask."}),
                "left_x": ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.005}),
                "left_y": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.005}),
                "right_x": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.005}),
                "right_y": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.005}),
                "confidence": ("FLOAT", {"default": 0.40, "min": 0.05, "max": 0.95, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "preview_points")
    FUNCTION = "warp"
    CATEGORY = "ZiT-Upscale/Edit"

    def warp(
        self,
        image,
        mode,
        strength,
        radius_scale,
        vertical_stretch,
        mask=None,
        left_x=0.40,
        left_y=0.45,
        right_x=0.60,
        right_y=0.45,
        confidence=0.40,
    ):
        batch = image.shape[0]
        out_imgs = []
        previews = []

        for i in range(batch):
            rgb = (255.0 * image[i].detach().cpu().numpy()).clip(0, 255).astype(np.uint8)
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            h, w = rgb.shape[:2]

            centers_radii: list[tuple[float, float, float]] = []

            if mode == "from_mask":
                if mask is None:
                    raise RuntimeError(
                        "Mode from_mask : connecte un masque (peins les seins dans MaskEditor sur Load Image)."
                    )
                m = mask[i] if mask.shape[0] > i else mask[0]
                m_np = m.detach().cpu().numpy().astype(np.float32)
                if m_np.shape[:2] != (h, w):
                    m_np = cv2.resize(m_np, (w, h), interpolation=cv2.INTER_LINEAR)
                centers_radii = _centers_from_mask(m_np, radius_scale)
                if not centers_radii:
                    raise RuntimeError(
                        "Masque vide : peins sur les zones a gonfler (clic droit Load Image > Open in MaskEditor)."
                    )
            elif mode == "auto_breasts":
                try:
                    import mediapipe  # noqa: F401

                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    centers, radius = _auto_breast_centers(bgr, confidence, radius_scale)
                except Exception as exc:
                    logger.warning("ZiTWarpMorpho auto detect failed (%s) — using manual points.", exc)
                    centers = [(left_x * w, left_y * h), (right_x * w, right_y * h)]
                    radius = min(w, h) * 0.32 * radius_scale
                centers_radii = [(cx, cy, radius) for cx, cy in centers]
            elif mode == "manual_one_point":
                radius = min(w, h) * 0.40 * radius_scale
                centers_radii = [(left_x * w, left_y * h, radius)]
            else:
                radius = min(w, h) * 0.32 * radius_scale
                centers_radii = [(left_x * w, left_y * h, radius), (right_x * w, right_y * h, radius)]

            map_x, map_y = _bulge_remap(h, w, centers_radii, strength, aspect=vertical_stretch)
            warped = cv2.remap(
                rgb, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

            prev = rgb.copy()
            for cx, cy, radius in centers_radii:
                cv2.circle(prev, (int(cx), int(cy)), int(radius), (255, 0, 0), max(2, w // 400))
                cv2.circle(prev, (int(cx), int(cy)), max(3, w // 250), (0, 255, 0), -1)

            out_imgs.append(warped.astype(np.float32) / 255.0)
            previews.append(prev.astype(np.float32) / 255.0)

        out_t = torch.from_numpy(np.stack(out_imgs, axis=0)).float()
        prev_t = torch.from_numpy(np.stack(previews, axis=0)).float()
        return (out_t, prev_t)


NODE_CLASS_MAPPINGS = {
    "ZiTWarpMorpho": ZiTWarpMorpho,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZiTWarpMorpho": "ZiT Warp Morpho (Liquify — garde tout)",
}
