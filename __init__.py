"""ComfyUI-ZiT-Upscale custom node entrypoint."""
from __future__ import annotations

import importlib.util
import logging
import os
import sys

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.realpath(__file__))
NODES_DIR = os.path.join(ROOT, "nodes")

# Sibling imports (edit_mask → region_mask) need nodes/ on sys.path when loaded via importlib.
if NODES_DIR not in sys.path:
    sys.path.insert(0, NODES_DIR)

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

# edit_mask must load before regional_edit (regional_edit imports edit_mask).
NODE_LOAD_ORDER = (
    "zit_upscale_refine",
    "region_mask",
    "semantic_mask",
    "sdxl_regional_edit",
    "edit_mask",
    "edit_mask_builder",
    "composite_preserve",
    "face_color_restore",
    "preserve_regions",
    "morpho_edit",
    "morph_dataset",
    "warp_morpho",
    "regional_edit",
    "inpaint_edit",
)


def _load_node_module(name: str):
    path = os.path.join(NODES_DIR, f"{name}.py")
    mod_name = f"comfyui_zit_upscale_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    if name in ("region_mask", "semantic_mask", "edit_mask"):
        sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


for _node_file in NODE_LOAD_ORDER:
    try:
        _mod = _load_node_module(_node_file)
        NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))
        logger.info("ZiT-Upscale: loaded %s", _node_file)
    except Exception as exc:
        logger.warning("ZiT-Upscale: failed to load %s: %s", _node_file, exc)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
