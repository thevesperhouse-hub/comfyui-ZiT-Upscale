import logging
import math

import comfy.model_management
import comfy.model_sampling
import comfy.samplers
import comfy.utils
import node_helpers
import torch

from nodes import MAX_RESOLUTION, common_ksampler

logger = logging.getLogger(__name__)

LATENT_UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
PIXEL_UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

# Z-Image Turbo on ~16 GB GPUs is comfortable around 2048 px output after a x2 latent pass.
DEFAULT_MAX_OUTPUT_SIDE = 2048
TILED_VAE_AUTO_THRESHOLD = 768
# ComfyUI encode_tiled_ uses tile_y // 2; tiny images produce invalid conv tiles.
MIN_TILED_VAE_SIDE = 512
MIN_VAE_SIDE = 64
# Z-Image patch_size=2 on latents; VAE is 8x → pixel sides must be multiples of 16.
ZIMAGE_PIXEL_ALIGN = 16

UPSCALE_FACTOR_OPTIONS = ["1x refine", "1.5x", "2x", "4x"]
PROMPT_GOAL_OPTIONS = [
    "enhance quality",
    "balanced",
    "follow prompt",
    "strong edit",
    "sculpt edit",
    "instruct edit",
]
# Backward compatibility with older workflows/widgets.
LEGACY_DETAIL_TO_GOAL = {
    "subtle": "enhance quality",
    "balanced": "balanced",
    "strong": "follow prompt",
    "sculpt": "sculpt edit",
}

UPSCALE_FACTOR_PRESETS = {
    "1x refine": {"scale_by": 1.0, "mode": "refine", "passes": 1},
    "1.5x": {"scale_by": 1.5, "mode": "latent_upscale", "passes": 1},
    "2x": {"scale_by": 2.0, "mode": "latent_upscale", "passes": 1},
    "4x": {"scale_by": 2.0, "mode": "progressive", "passes": 2},
}

PROMPT_GOAL_PRESETS = {
    # ComfyUI blueprint "Image Upscale (Z-image-Turbo)":
    # 1MP normalize → RealESRGAN x4 → scale 0.5 → ae VAE → AuraFlow 3 → KSampler denoise 0.33
    "enhance quality": {
        "denoise": 0.33,
        "reference_cond": False,
        "edit_mode": False,
        "min_steps": 5,
        "preserve_pixels": False,
        "official_upscale": True,
        "target_megapixels": 1.0,
        "post_esrgan_scale": 0.5,
        "aura_shift": 3.0,
        "default_sampler": "dpmpp_2m_sde",
        "default_scheduler": "beta",
    },
    "balanced": {"denoise": 0.40, "reference_cond": False, "edit_mode": False, "min_steps": 5},
    # Omni img2img: anchors identity via reference_latents + encoded source latent.
    "follow prompt": {"denoise": 0.38, "reference_cond": False, "edit_mode": True, "min_steps": 8, "controlnet_denoise": 0.32},
    "strong edit": {"denoise": 0.48, "reference_cond": False, "edit_mode": True, "min_steps": 12, "controlnet_denoise": 0.40},
    # Hybrid Omni img2img + optional ControlNet on source RGB — body/pose reshaping while anchoring identity.
    "sculpt edit": {"denoise": 0.56, "reference_cond": False, "edit_mode": True, "min_steps": 10, "controlnet_denoise": 0.44},
    # Max instruction following on Turbo: Omni + high denoise + 2-pass consolidate.
    "instruct edit": {
        "denoise": 0.62,
        "denoise_pass2": 0.38,
        "reference_cond": False,
        "edit_mode": True,
        "min_steps": 16,
        "pass2_steps": 10,
        "two_pass": True,
        "controlnet_denoise": 0.50,
        "instruct_prompt": True,
    },
}


def _resolve_upscale_factor(upscale_factor: str):
    preset = UPSCALE_FACTOR_PRESETS.get(upscale_factor)
    if preset is None:
        logger.warning("ZiT-Upscale: unknown upscale_factor %r, using 2x.", upscale_factor)
        return UPSCALE_FACTOR_PRESETS["2x"]
    return preset


def _resolve_prompt_goal(prompt_goal: str):
    prompt_goal = LEGACY_DETAIL_TO_GOAL.get(prompt_goal, prompt_goal)
    preset = PROMPT_GOAL_PRESETS.get(prompt_goal)
    if preset is None:
        logger.warning("ZiT-Upscale: unknown prompt_goal %r, using balanced.", prompt_goal)
        return PROMPT_GOAL_PRESETS["balanced"]
    return preset


def _resolve_sampling_params(prompt_goal: str, denoise: float):
    preset = _resolve_prompt_goal(prompt_goal)
    use_denoise = denoise if denoise > 0.0 else preset["denoise"]
    return (
        use_denoise,
        preset["reference_cond"],
        preset.get("edit_mode", False),
        preset.get("min_steps", 5),
    )


def _apply_reference_conditioning(positive, negative, reference_latent, enabled: bool):
    if not enabled or reference_latent is None:
        return positive, negative
    positive = node_helpers.conditioning_set_values(
        positive,
        {"reference_latents": [reference_latent]},
        append=True,
    )
    return positive, negative


def _gonza_split_sample(
    model,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent,
    split_at: int = 3,
    denoise: float = 1.0,
):
    """Gonza Z-Image refiner: KSamplerAdvanced 0→split then split→steps (not img2img denoise 0.20)."""
    split_at = max(1, min(int(split_at), steps - 1))
    (latent,) = common_ksampler(
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
        disable_noise=False,
        start_step=0,
        last_step=split_at,
        force_full_denoise=False,
    )
    (latent,) = common_ksampler(
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
        disable_noise=True,
        start_step=split_at,
        last_step=steps,
        force_full_denoise=True,
    )
    return latent


OMNI_DEFAULT_PROMPTS = {
    "enhance quality": "masterpiece, best quality, ultra detailed, sharp focus",
    "balanced": "masterpiece, best quality, detailed, natural lighting",
}


def _format_instruct_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    if not prompt:
        return prompt
    lower = prompt.lower()
    if lower.startswith("edit:") or "keep unchanged" in lower or "while keeping" in lower:
        return prompt
    return (
        f"Edit the image: {prompt}. "
        "Keep the same person, face, pose, clothing, and background unless the edit explicitly changes them."
    )


def _resolve_omni_prompt(positive_prompt: str, prompt_goal: str) -> str:
    prompt = positive_prompt.strip()
    if prompt:
        preset = _resolve_prompt_goal(prompt_goal)
        if preset.get("instruct_prompt"):
            return _format_instruct_prompt(prompt)
        return prompt
    default = OMNI_DEFAULT_PROMPTS.get(prompt_goal)
    if default:
        return default
    raise RuntimeError(
        "ZiT-Upscale: fill positive_prompt on the node for guided edits "
        "(prompt_goal = follow prompt / strong edit / sculpt edit)."
    )


def _sanitize_float(value, default: float, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        logger.warning("ZiT-Upscale: invalid %s=%r, using %s.", label, value, default)
        return default
    if number != number or number in (float("inf"), float("-inf")):
        logger.warning("ZiT-Upscale: invalid %s=%r, using %s.", label, value, default)
        return default
    return number


def _sanitize_int(value, default: int, label: str, minimum: int = 0) -> int:
    number = int(_sanitize_float(value, default, label))
    if number < minimum:
        logger.warning("ZiT-Upscale: invalid %s=%r, using %s.", label, value, default)
        return default
    return number


def _sanitize_combo(value, options, default: str, label: str) -> str:
    if value not in options:
        logger.warning("ZiT-Upscale: invalid %s=%r, using %r.", label, value, default)
        return default
    return value


def _sanitize_upscale_refine_params(
    denoise,
    aura_shift,
    control_strength,
    max_output_side,
    target_width,
    target_height,
    latent_upscale_method,
    pixel_scale_by,
    pixel_upscale_method,
    tile_size,
    overlap,
):
    return {
        "denoise": _sanitize_float(denoise, 0.0, "denoise"),
        "aura_shift": _sanitize_float(aura_shift, 0.0, "aura_shift"),
        "control_strength": _sanitize_float(control_strength, 1.0, "control_strength"),
        "max_output_side": _sanitize_int(max_output_side, 0, "max_output_side"),
        "target_width": _sanitize_int(target_width, 0, "target_width"),
        "target_height": _sanitize_int(target_height, 0, "target_height"),
        "latent_upscale_method": _sanitize_combo(
            latent_upscale_method, LATENT_UPSCALE_METHODS, "nearest-exact", "latent_upscale_method"
        ),
        "pixel_scale_by": _sanitize_float(pixel_scale_by, 1.0, "pixel_scale_by"),
        "pixel_upscale_method": _sanitize_combo(
            pixel_upscale_method, PIXEL_UPSCALE_METHODS, "lanczos", "pixel_upscale_method"
        ),
        "tile_size": _sanitize_int(tile_size, 512, "tile_size", minimum=256),
        "overlap": _sanitize_int(overlap, 64, "overlap", minimum=32),
    }


def _prepare_source_pixels(
    image,
    prompt_goal: str,
    pixel_scale_by: float,
    pixel_upscale_method: str,
    target_width: int,
    target_height: int,
    max_output_side: int,
    mode: str,
    scale_by: float,
    compression: int,
):
    """Resize source before upscale prep. Official path defers to RealESRGAN megapixel normalize."""
    goal = _resolve_prompt_goal(prompt_goal)
    preserve = goal.get("preserve_pixels", False)

    pixels = _scale_image_pixels(image, pixel_upscale_method, pixel_scale_by)
    if goal.get("official_upscale") and not (target_width or target_height or max_output_side):
        return _ensure_minimum_vae_size(pixels, compression, pixel_upscale_method)
    if preserve and not (target_width or target_height or max_output_side):
        pixels = _ensure_minimum_vae_size(pixels, compression, pixel_upscale_method)
        return pixels

    pixels = _fit_image(
        pixels,
        target_width,
        target_height,
        0,
        compression,
        pixel_upscale_method,
    )
    pixels = _cap_for_output_side(
        pixels,
        mode,
        scale_by,
        max_output_side,
        compression,
        pixel_upscale_method,
    )
    pixels = _ensure_minimum_vae_size(pixels, compression, pixel_upscale_method)
    return pixels


def _pixel_align(compression: int) -> int:
    return max(compression * 2, ZIMAGE_PIXEL_ALIGN)


def _build_plain_text_conditioning(clip, prompt: str):
    tokens = clip.tokenize(prompt.strip())
    return clip.encode_from_tokens_scheduled(tokens)


def _zero_out_conditioning(conditioning):
    """Same as ComfyUI ConditioningZeroOut — required for Z-Image Fun ControlNet."""
    zeroed = []
    for tensor, metadata in conditioning:
        meta = metadata.copy()
        pooled = meta.get("pooled_output")
        if pooled is not None:
            meta["pooled_output"] = torch.zeros_like(pooled)
        lyrics = meta.get("conditioning_lyrics")
        if lyrics is not None:
            meta["conditioning_lyrics"] = torch.zeros_like(lyrics)
        zeroed.append([torch.zeros_like(tensor), meta])
    return zeroed


def _resolve_edit_denoise(manual_denoise: float, preset_denoise: float) -> float:
    if manual_denoise > 0.0 and manual_denoise < 0.99:
        return manual_denoise
    return preset_denoise


def _encode_omni_reference(vae, pixels, use_tiled_vae: bool, tile_size: int, overlap: int):
    """Reference image for Omni edit mode (~1 MP, 16px aligned). Separate from the sampling latent."""
    samples = pixels.movedim(-1, 1)
    total = int(1024 * 1024)
    width_in = samples.shape[3]
    height_in = samples.shape[2]
    scale_by = math.sqrt(total / (width_in * height_in))
    align = ZIMAGE_PIXEL_ALIGN
    width = max(align, round(width_in * scale_by / align) * align)
    height = max(align, round(height_in * scale_by / align) * align)
    resized = comfy.utils.common_upscale(samples, width, height, "area", "disabled").movedim(1, -1)
    encoded = _encode_pixels(vae, resized, use_tiled_vae, tile_size, overlap)
    return encoded["samples"]


def _empty_latent_for_pixels(pixels, compression: int, channels: int):
    _, height, width, _ = pixels.shape
    latent_h = max(1, height // compression)
    latent_w = max(1, width // compression)
    return {"samples": torch.zeros(1, channels, latent_h, latent_w)}


def _build_omni_conditioning(clip, reference_latent, prompt: str):
    """
    Z-Image Omni text+image conditioning for img2img.

    Uses the same encoded latent as the img2img pass (not a separate 1024px resize)
    so reference spatial dims stay patch-compatible with Lumina2 (patch_size=2).
    """
    prompt = prompt.strip()
    if not prompt:
        raise RuntimeError("ZiT-Upscale: positive_prompt is empty but Omni conditioning was requested.")

    ref = reference_latent
    if ref.ndim == 3:
        ref = ref.unsqueeze(0)
    _, _, ref_h, ref_w = ref.shape
    if ref_h % 2 != 0 or ref_w % 2 != 0:
        raise RuntimeError(
            f"ZiT-Upscale: Omni reference latent size {ref_w}x{ref_h} is not divisible by 2. "
            "Use image dimensions that are multiples of 16 pixels."
        )

    vision_prefixes = [
        "<|im_start|>user\n<|vision_start|>",
        "<|vision_end|><|im_end|>",
    ]
    llama_template = "<|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n<|vision_start|>"

    tokens = clip.tokenize(prompt, llama_template=llama_template)
    conditioning = clip.encode_from_tokens_scheduled(tokens)

    extra_text_embeds = []
    for prefix in vision_prefixes:
        prefix_tokens = clip.tokenize(prefix, llama_template="{}")
        extra_text_embeds.append(clip.encode_from_tokens_scheduled(prefix_tokens)[0][0])

    conditioning = node_helpers.conditioning_set_values(
        conditioning,
        {"reference_latents": [ref]},
        append=True,
    )
    conditioning = node_helpers.conditioning_set_values(
        conditioning,
        {"reference_latents_text_embeds": extra_text_embeds},
        append=True,
    )
    return conditioning


def _apply_zimage_controlnet(model, model_patch, vae, control_image, strength, mask=None, inpaint_image=None):
    if model_patch is None:
        return model

    try:
        import comfy.ldm.lumina.controlnet  # noqa: F401
        from comfy_extras.nodes_model_patch import ZImageControlPatch
    except ImportError as exc:
        raise RuntimeError(
            "ZiT-Upscale: Z-Image ControlNet requires ComfyUI comfy_extras.nodes_model_patch."
        ) from exc

    if not isinstance(model_patch.model, comfy.ldm.lumina.controlnet.ZImage_Control):
        raise RuntimeError(
            "ZiT-Upscale: model_patch is not a Z-Image ControlNet. "
            "Load it with ModelPatchLoader (Z-Image Fun controlnet .safetensors)."
        )

    model_patched = model.clone()
    if control_image is not None:
        control_image = control_image[:, :, :, :3]
    if inpaint_image is not None:
        inpaint_image = inpaint_image[:, :, :, :3]

    mask_tensor = None
    if mask is not None:
        mask_tensor = mask
        if mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.unsqueeze(1)
        if mask_tensor.ndim == 4:
            mask_tensor = mask_tensor.unsqueeze(2)
        mask_tensor = 1.0 - mask_tensor

    patch = ZImageControlPatch(
        model_patch,
        vae,
        control_image,
        strength,
        inpaint_image=inpaint_image,
        mask=mask_tensor,
    )
    model_patched.set_model_noise_refiner_patch(patch)
    model_patched.set_model_double_block_patch(patch)
    return model_patched


def _unload_for_sampling(vae, model, unload_all_models=False):
    _unload_vae(vae)
    if unload_all_models:
        # Drop SDXL / other workflow models before loading Z-Image (fixes vbar Fault failed: 2).
        _free_all_loaded_models()
    else:
        _unload_model(model)
    _release_vram()


def _unload_for_decode(model):
    _unload_model(model)
    _release_vram()


def _snap(value: int, multiple: int) -> int:
    multiple = max(1, multiple)
    value = max(multiple, value)
    return (value // multiple) * multiple


def _release_vram():
    comfy.model_management.soft_empty_cache(force=True)


def _free_all_loaded_models():
    comfy.model_management.unload_all_models()
    _release_vram()


def _unload_patcher(patcher):
    if patcher is None:
        return
    try:
        comfy.model_management.unload_model_and_clones(patcher)
    except Exception:
        pass


def _unload_vae(vae):
    if vae is not None and hasattr(vae, "patcher"):
        _unload_patcher(vae.patcher)
    _release_vram()


def _unload_model(model):
    if model is not None and hasattr(model, "patcher"):
        _unload_patcher(model.patcher)
    _release_vram()


def _get_model_latent_channels(model) -> int:
    try:
        latent_format = model.get_model_object("latent_format")
        if latent_format is not None and hasattr(latent_format, "latent_channels"):
            return int(latent_format.latent_channels)
    except Exception:
        pass
    try:
        return int(model.model.model_config.unet_config.get("in_channels", 16))
    except Exception:
        return 16


def _validate_vae_latent_compatibility(vae, model, latent):
    samples = latent["samples"] if isinstance(latent, dict) else latent
    latent_channels = int(samples.shape[1])
    expected_channels = _get_model_latent_channels(model)

    if latent_channels == expected_channels:
        return

    vae_name = getattr(vae, "vae_name", None) or "unknown VAE"
    hint = "ae.safetensors"
    if latent_channels == 128 or "flux2" in str(vae_name).lower():
        hint = "ae.safetensors (16 channels). flux2-vae.safetensors is for Flux2 only (128 channels)."

    raise RuntimeError(
        f"ZiT-Upscale: incompatible VAE latent ({latent_channels} channels from {vae_name}). "
        f"Z-Image Turbo expects {expected_channels} channels — use {hint}"
    )

def _ensure_minimum_vae_size(image, compression: int, upscale_method: str):
    _, height, width, _ = image.shape
    min_required = max(compression, MIN_VAE_SIDE)
    if height >= min_required and width >= min_required:
        return image

    scale = min_required / float(min(height, width))
    align = _pixel_align(compression)
    new_width = _snap(max(compression, round(width * scale)), align)
    new_height = _snap(max(compression, round(height * scale)), align)
    logger.info(
        "ZiT-Upscale: input too small for VAE (%sx%s), upscaling to %sx%s.",
        width,
        height,
        new_width,
        new_height,
    )
    return _resize_image_to(image, new_width, new_height, upscale_method)


def _should_use_tiled_vae(pixels, use_tiled_vae: bool, tile_size: int) -> bool:
    if not use_tiled_vae:
        return False
    height, width = pixels.shape[1], pixels.shape[2]
    if min(height, width) < MIN_TILED_VAE_SIDE:
        return False
    if min(height, width, tile_size) < MIN_TILED_VAE_SIDE:
        return False
    return True


def _resolve_vae_tile_params(pixels, tile_size: int, overlap: int):
    height, width = pixels.shape[1], pixels.shape[2]
    tile = min(tile_size, height, width)
    tile = max(MIN_TILED_VAE_SIDE, tile)
    tile = min(tile, height, width)
    overlap = min(overlap, max(32, tile // 8))
    if tile < MIN_TILED_VAE_SIDE or min(height, width) < MIN_TILED_VAE_SIDE:
        return None
    return tile, overlap


def _scale_image_pixels(image, upscale_method: str, scale_by: float):
    if abs(scale_by - 1.0) < 1e-6:
        return image

    samples = image.movedim(-1, 1)
    width = max(1, round(samples.shape[3] * scale_by))
    height = max(1, round(samples.shape[2] * scale_by))
    scaled = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
    return scaled.movedim(1, -1)


def _scale_to_total_megapixels(image, megapixels: float, upscale_method: str, align: int = 8):
    """Match ComfyUI ImageScaleToTotalPixels (blueprint uses 1 MP before RealESRGAN)."""
    if megapixels <= 0.0:
        return image
    samples = image.movedim(-1, 1)
    target_pixels = megapixels * 1024.0 * 1024.0
    current_pixels = float(samples.shape[2] * samples.shape[3])
    if current_pixels <= 0.0:
        return image
    scale_by = math.sqrt(target_pixels / current_pixels)
    width = max(align, round(samples.shape[3] * scale_by / align) * align)
    height = max(align, round(samples.shape[2] * scale_by / align) * align)
    if width == samples.shape[3] and height == samples.shape[2]:
        return image
    scaled = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
    return scaled.movedim(1, -1)


def _upscale_with_model(upscale_model, image):
    """Same tiled path as ComfyUI ImageUpscaleWithModel."""
    device = comfy.model_management.get_torch_device()
    memory_required = comfy.model_management.module_size(upscale_model.model)
    memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
    memory_required += image.nelement() * image.element_size()
    comfy.model_management.free_memory(memory_required, device)

    upscale_model.to(device)
    in_img = image.movedim(-1, -3).to(device)
    tile = 512
    overlap = 32
    output_device = comfy.model_management.intermediate_device()

    try:
        while True:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                    in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
                )
                pbar = comfy.utils.ProgressBar(steps)
                scaled = comfy.utils.tiled_scale(
                    in_img,
                    lambda batch: upscale_model(batch.float()),
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                    upscale_amount=upscale_model.scale,
                    pbar=pbar,
                    output_device=output_device,
                )
                break
            except Exception as exc:
                comfy.model_management.raise_non_oom(exc)
                tile //= 2
                if tile < 128:
                    raise
    finally:
        upscale_model.to("cpu")

    return torch.clamp(scaled.movedim(-3, -1), min=0, max=1.0).to(comfy.model_management.intermediate_dtype())


def _apply_official_upscale_prep(image, upscale_model, scale_by: float, pixel_upscale_method: str, goal_preset: dict):
    """
    Official ComfyUI Z-Image Turbo upscale pixel prep.
    Requires RealESRGAN (or compatible x4 model) for 2x; falls back to Lanczos with a warning.
    """
    megapixels = goal_preset.get("target_megapixels", 1.0)
    pixels = _scale_to_total_megapixels(image, megapixels, pixel_upscale_method)

    if scale_by <= 1.01:
        return pixels, "refine", 1.0

    if upscale_model is None:
        logger.warning(
            "ZiT-Upscale: connect upscale_model (RealESRGAN_x4plus) for official sharp 2x. "
            "Falling back to pixel Lanczos — softer result."
        )
        return _scale_image_pixels(pixels, pixel_upscale_method, scale_by), "refine", 1.0

    pixels = _upscale_with_model(upscale_model, pixels)
    post_scale = goal_preset.get("post_esrgan_scale", 0.5)
    if scale_by >= 1.75:
        pixels = _scale_image_pixels(pixels, pixel_upscale_method, post_scale)
    elif scale_by > 1.01:
        target_scale = scale_by / max(upscale_model.scale, 1.0)
        pixels = _scale_image_pixels(pixels, pixel_upscale_method, target_scale)
    logger.info(
        "ZiT-Upscale: official prep — %.1f MP → ESRGAN x%s → scale %.2f",
        megapixels,
        upscale_model.scale,
        post_scale if scale_by >= 1.75 else scale_by,
    )
    return pixels, "refine", 1.0


def _apply_goal_sampling_defaults(prompt_goal: str, aura_shift: float, sampler_name: str, scheduler: str):
    preset = _resolve_prompt_goal(prompt_goal)
    if preset.get("aura_shift") and aura_shift <= 0.0:
        aura_shift = preset["aura_shift"]
    if preset.get("default_sampler"):
        sampler_name = preset["default_sampler"]
    if preset.get("default_scheduler"):
        scheduler = preset["default_scheduler"]
    return aura_shift, sampler_name, scheduler


def _resize_image_to(image, width: int, height: int, upscale_method: str):
    samples = image.movedim(-1, 1)
    scaled = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
    return scaled.movedim(1, -1)


def _cap_for_output_side(
    image,
    mode: str,
    scale_by: float,
    max_output_side: int,
    compression: int,
    upscale_method: str,
):
    if max_output_side <= 0:
        return image

    _, height, width, _ = image.shape
    upscale_factor = 1.0 if mode == "refine" else max(1.0, scale_by)
    projected_output = max(height, width) * upscale_factor
    if projected_output <= max_output_side:
        return image

    allowed_input_longest = max_output_side / upscale_factor
    scale = allowed_input_longest / max(height, width)
    min_side = max(compression, MIN_VAE_SIDE)
    align = _pixel_align(compression)
    new_width = _snap(max(min_side, round(width * scale)), align)
    new_height = _snap(max(min_side, round(height * scale)), align)
    logger.info(
        "ZiT-Upscale: limiting input from %sx%s to %sx%s (max output side %s).",
        width,
        height,
        new_width,
        new_height,
        max_output_side,
    )
    return _resize_image_to(image, new_width, new_height, upscale_method)


def _fit_image(image, target_width: int, target_height: int, max_side: int, compression: int, upscale_method: str):
    _, height, width, _ = image.shape
    align = _pixel_align(compression)
    min_side = max(compression, MIN_VAE_SIDE)

    if target_width > 0 and target_height > 0:
        width = _snap(target_width, align)
        height = _snap(target_height, align)
        return _resize_image_to(image, width, height, upscale_method)

    if target_width > 0:
        width = _snap(target_width, align)
        height = max(min_side, round(height * (width / max(1, image.shape[2]))))
        height = _snap(height, align)
        return _resize_image_to(image, width, height, upscale_method)

    if target_height > 0:
        height = _snap(target_height, align)
        width = max(min_side, round(width * (height / max(1, image.shape[1]))))
        width = _snap(width, align)
        return _resize_image_to(image, width, height, upscale_method)

    if max_side > 0:
        longest = max(width, height)
        if longest > max_side:
            scale = max_side / float(longest)
            width = _snap(max(compression, round(width * scale)), align)
            height = _snap(max(compression, round(height * scale)), align)
            return _resize_image_to(image, width, height, upscale_method)

    width = _snap(width, align)
    height = _snap(height, align)
    if width != image.shape[2] or height != image.shape[1]:
        return _resize_image_to(image, width, height, upscale_method)
    return image


def _upscale_latent(samples, upscale_method: str, scale_by: float):
    if abs(scale_by - 1.0) < 1e-6:
        return samples

    latent = samples.copy()
    width = max(1, round(samples["samples"].shape[-1] * scale_by))
    height = max(1, round(samples["samples"].shape[-2] * scale_by))
    latent["samples"] = comfy.utils.common_upscale(
        samples["samples"], width, height, upscale_method, "disabled"
    )
    return latent


def _apply_quality_upscale_pixels(pixels, mode: str, scale_by: float, pixel_upscale_method: str, prompt_goal: str = "", upscale_model=None):
    goal = _resolve_prompt_goal(prompt_goal)
    if goal.get("official_upscale"):
        return _apply_official_upscale_prep(pixels, upscale_model, scale_by, pixel_upscale_method, goal)
    if mode == "latent_upscale" and scale_by > 1.0:
        scaled = _scale_image_pixels(pixels, pixel_upscale_method, scale_by)
        logger.info("ZiT-Upscale: pixel upscale x%.2f (%s).", scale_by, pixel_upscale_method)
        return scaled, "refine", 1.0
    return pixels, mode, scale_by


def _apply_aura_flow(model, shift: float):
    if shift <= 0.0:
        return model

    m = model.clone()

    sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
    sampling_type = comfy.model_sampling.CONST

    class ModelSamplingAdvanced(sampling_base, sampling_type):
        pass

    original = m.get_model_object("model_sampling")
    model_sampling = ModelSamplingAdvanced(model.model.model_config)
    model_sampling.set_parameters(shift=shift, multiplier=1.0)
    if hasattr(original, "noise_scale"):
        model_sampling.set_noise_scale(original.noise_scale)
    m.add_object_patch("model_sampling", model_sampling)
    return m


def _encode_pixels(vae, pixels, use_tiled_vae: bool, tile_size: int, overlap: int):
    tile_params = _resolve_vae_tile_params(pixels, tile_size, overlap)
    tiled = tile_params is not None and _should_use_tiled_vae(pixels, use_tiled_vae, tile_size)

    if tiled:
        tile, tile_overlap = tile_params
        try:
            latent = vae.encode_tiled(
                pixels,
                tile_x=tile,
                tile_y=tile,
                overlap=tile_overlap,
            )
        except RuntimeError as exc:
            if "Kernel size can't be greater than actual input size" not in str(exc):
                raise
            logger.warning("ZiT-Upscale: tiled VAE encode failed on small tiles, using regular encode.")
            _unload_vae(vae)
            latent = vae.encode(pixels)
    else:
        try:
            latent = vae.encode(pixels)
        except Exception as exc:
            comfy.model_management.raise_non_oom(exc)
            if tile_params is None:
                raise
            logger.warning("ZiT-Upscale: VAE encode OOM, retrying with tiled VAE.")
            _unload_vae(vae)
            tile, tile_overlap = tile_params
            latent = vae.encode_tiled(
                pixels,
                tile_x=tile,
                tile_y=tile,
                overlap=tile_overlap,
            )

    return {"samples": latent}


def _decode_latent(vae, latent, use_tiled_vae: bool, tile_size: int, overlap: int):
    samples = latent["samples"]
    compression = vae.spacial_compression_decode()
    pixel_height = samples.shape[-2] * compression
    pixel_width = samples.shape[-1] * compression
    fake_pixels = torch.empty((samples.shape[0], pixel_height, pixel_width, 3), device="cpu")
    tile_params = _resolve_vae_tile_params(fake_pixels, tile_size, overlap)
    tiled = tile_params is not None and _should_use_tiled_vae(
        fake_pixels,
        use_tiled_vae,
        tile_size,
    )

    if tiled:
        tile, tile_overlap = tile_params
        try:
            return vae.decode_tiled(
                samples,
                tile_x=tile,
                tile_y=tile,
                overlap=tile_overlap,
            )
        except Exception as exc:
            comfy.model_management.raise_non_oom(exc)
            if isinstance(exc, RuntimeError) and "Kernel size can't be greater than actual input size" in str(exc):
                logger.warning("ZiT-Upscale: tiled VAE decode failed on small tiles, using regular decode.")
                _unload_vae(vae)
                return vae.decode(samples)
            logger.warning("ZiT-Upscale: VAE decode OOM, retrying with smaller tiles.")
            _unload_vae(vae)
            small_tile = max(MIN_TILED_VAE_SIDE, tile // 2)
            small_overlap = max(32, tile_overlap // 2)
            return vae.decode_tiled(
                samples,
                tile_x=small_tile,
                tile_y=small_tile,
                overlap=small_overlap,
            )

    try:
        return vae.decode(samples)
    except Exception as exc:
        comfy.model_management.raise_non_oom(exc)
        if tile_params is None:
            raise
        logger.warning("ZiT-Upscale: VAE decode OOM, retrying with tiled VAE.")
        _unload_vae(vae)
        tile, tile_overlap = tile_params
        return vae.decode_tiled(
            samples,
            tile_x=tile,
            tile_y=tile,
            overlap=tile_overlap,
        )


def _attach_mask(latent, mask, pixel_height: int, pixel_width: int):
    if mask is None:
        return latent

    mask_tensor = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
    if mask_tensor.shape[-2] != pixel_height or mask_tensor.shape[-1] != pixel_width:
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor,
            size=(pixel_height, pixel_width),
            mode="bilinear",
        )

    out = latent.copy()
    out["noise_mask"] = mask_tensor
    return out


class ZiTUpscaleRefine:
    """
    Upscale and refine an existing image with Z-Image Turbo.

    Connect MODEL / CONDITIONING / VAE from your usual loader chain so LoRAs,
    ControlNet hooks, and AuraFlow patches applied upstream are preserved.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "positive_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "dynamicPrompts": True,
                        "tooltip": "Describe what to change or enhance. Used with the CLIP input for Z-Image Omni image+text conditioning.",
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "steps": ("INT", {"default": 5, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "upscale_factor": (
                    UPSCALE_FACTOR_OPTIONS,
                    {
                        "default": "2x",
                        "tooltip": "Output resolution vs source. 1x refine = same size, better details. 4x = two latent x2 passes.",
                    },
                ),
                "prompt_goal": (
                    PROMPT_GOAL_OPTIONS,
                    {
                        "default": "enhance quality",
                        "tooltip": "instruct edit = max Turbo instruction following (Omni + 2-pass). enhance quality = upscale only.",
                    },
                ),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Manual denoise override. 0 = use prompt_goal preset.",
                    },
                ),
                "aura_shift": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "AuraFlow shift. enhance quality auto-uses 3.0 (ComfyUI official upscale). 0 = preset default.",
                    },
                ),
                "use_tiled_vae": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Tiled VAE encode/decode. Softens fine texture — leave OFF for quality (Gonza-style). Auto-fallback on OOM only.",
                    },
                ),
                "free_vram_between_steps": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Unload models between encode/sample/decode. OFF = sharper (Gonza-style). ON = saves VRAM after SDXL chains.",
                    },
                ),
                "control_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "ControlNet strength when model_patch is connected. Ignored otherwise.",
                    },
                ),
            },
            "optional": {
                "clip": (
                    "CLIP",
                    {
                        "tooltip": "Required for prompt-guided edits. Wire CLIP from your loader/LoRA chain. Enables Z-Image Omni image+text conditioning.",
                    },
                ),
                "positive": (
                    "CONDITIONING",
                    {
                        "tooltip": "Legacy quality-only path when CLIP is not connected. Uses plain CLIPTextEncode (prompt has little effect on img2img).",
                    },
                ),
                "model_patch": (
                    "MODEL_PATCH",
                    {
                        "tooltip": "Optional Z-Image Fun ControlNet from ModelPatchLoader. Enables structural edits (pose, depth, canny…).",
                    },
                ),
                "control_image": (
                    "IMAGE",
                    {
                        "tooltip": "Control map (canny, depth, pose…). Defaults to the source image when empty.",
                    },
                ),
                "mask": ("MASK",),
                "upscale_model": (
                    "UPSCALE_MODEL",
                    {
                        "tooltip": "RealESRGAN_x4plus for sharp 2x (ComfyUI official Z-Image upscale blueprint).",
                    },
                ),
                "max_output_side": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_RESOLUTION,
                        "step": 8,
                        "tooltip": "Optional safety cap on longest output side. 0 = no limit (keeps source proportions).",
                    },
                ),
                "target_width": (
                    "INT",
                    {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Force output width. 0 = keep aspect ratio."},
                ),
                "target_height": (
                    "INT",
                    {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Force output height. 0 = keep aspect ratio."},
                ),
                "latent_upscale_method": (LATENT_UPSCALE_METHODS,),
                "pixel_scale_by": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.25,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": "Pre-encode pixel resize. 1.0 = keep source resolution.",
                    },
                ),
                "pixel_upscale_method": (PIXEL_UPSCALE_METHODS,),
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 32, "max": 4096, "step": 32}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "upscale_refine"
    CATEGORY = "ZiT-Upscale"

    def upscale_refine(
        self,
        model,
        negative,
        vae,
        image,
        positive_prompt,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        upscale_factor,
        prompt_goal,
        denoise,
        aura_shift,
        use_tiled_vae,
        free_vram_between_steps,
        control_strength,
        clip=None,
        positive=None,
        model_patch=None,
        control_image=None,
        mask=None,
        upscale_model=None,
        max_output_side=0,
        target_width=0,
        target_height=0,
        latent_upscale_method="bislerp",
        pixel_scale_by=1.0,
        pixel_upscale_method="lanczos",
        tile_size=512,
        overlap=64,
    ):
        params = _sanitize_upscale_refine_params(
            denoise,
            aura_shift,
            control_strength,
            max_output_side,
            target_width,
            target_height,
            latent_upscale_method,
            pixel_scale_by,
            pixel_upscale_method,
            tile_size,
            overlap,
        )
        denoise = params["denoise"]
        aura_shift = params["aura_shift"]
        control_strength = params["control_strength"]
        max_output_side = params["max_output_side"]
        target_width = params["target_width"]
        target_height = params["target_height"]
        latent_upscale_method = params["latent_upscale_method"]
        pixel_scale_by = params["pixel_scale_by"]
        pixel_upscale_method = params["pixel_upscale_method"]
        tile_size = params["tile_size"]
        overlap = params["overlap"]

        preset = _resolve_upscale_factor(upscale_factor)
        mode = preset["mode"]
        scale_by = preset["scale_by"]
        upscale_passes = preset["passes"]
        goal_preset = _resolve_prompt_goal(prompt_goal)
        denoise, use_reference_cond, use_edit_mode, min_steps = _resolve_sampling_params(prompt_goal, denoise)
        aura_shift, sampler_name, scheduler = _apply_goal_sampling_defaults(
            prompt_goal, aura_shift, sampler_name, scheduler
        )
        sample_steps = max(steps, min_steps)
        sample_cfg = 1.0 if model_patch is not None else cfg
        if model_patch is not None and cfg != 1.0:
            logger.warning("ZiT-Upscale: Z-Image ControlNet requires cfg=1.0, overriding cfg=%.2f.", cfg)

        omni_prompt = None
        if use_edit_mode:
            use_reference_cond = False
            if model_patch is not None:
                preset_denoise = goal_preset.get("controlnet_denoise", 0.55)
            else:
                preset_denoise = goal_preset["denoise"]
            denoise = _resolve_edit_denoise(denoise, preset_denoise)
            if clip is None:
                raise RuntimeError(
                    "ZiT-Upscale: follow prompt / strong edit require CLIP connected "
                    "and a positive_prompt describing the edit."
                )
            omni_prompt = _resolve_omni_prompt(positive_prompt, prompt_goal)
            if not positive_prompt.strip() and positive is not None:
                raise RuntimeError(
                    "ZiT-Upscale instruct edit: the connected 'positive' socket is ignored. "
                    "Connect CLIP to this node and write your edit instruction in positive_prompt "
                    "(Omni image+text conditioning). Do not rely on an external CLIPTextEncode alone."
                )
            if model_patch is not None and prompt_goal not in ("sculpt edit", "strong edit", "instruct edit"):
                logger.warning(
                    "ZiT-Upscale: ControlNet with follow prompt is conservative. "
                    "Use sculpt edit for body/pose reshaping, or disconnect model_patch."
                )
            elif model_patch is not None:
                logger.info(
                    "ZiT-Upscale: ControlNet active — plain CLIP + img2img (Omni disabled to avoid batch crash)."
                )
        elif positive is None and not (clip is not None and positive_prompt.strip()):
            raise RuntimeError(
                "ZiT-Upscale: connect CLIP + positive_prompt, or connect a positive CONDITIONING input."
            )

        logger.info(
            "ZiT-Upscale: factor=%s mode=%s scale=%s goal=%s denoise=%.2f edit=%s ref_cond=%s controlnet=%s steps=%s prompt=%r",
            upscale_factor,
            mode,
            scale_by,
            prompt_goal,
            denoise,
            use_edit_mode,
            use_reference_cond,
            model_patch is not None,
            sample_steps,
            (omni_prompt or positive_prompt)[:80] if omni_prompt or positive_prompt else "",
        )

        if not use_edit_mode and goal_preset.get("official_upscale"):
            logger.info("ZiT-Upscale: official ComfyUI upscale path (ESRGAN + denoise 0.33 + AuraFlow 3)")

        work_model = _apply_aura_flow(model, aura_shift)
        compression = vae.spacial_compression_encode()
        channels = _get_model_latent_channels(work_model)

        pixels = _prepare_source_pixels(
            image,
            prompt_goal,
            pixel_scale_by,
            pixel_upscale_method,
            target_width,
            target_height,
            max_output_side,
            mode,
            scale_by,
            compression,
        )

        quality_preserve = _resolve_prompt_goal(prompt_goal).get("preserve_pixels", False)
        unload_all = free_vram_between_steps and not quality_preserve

        if use_edit_mode:
            gen_pixels = pixels
            if mode != "refine" and scale_by > 1.0:
                gen_pixels = _scale_image_pixels(pixels, pixel_upscale_method, scale_by)
                gen_pixels = _ensure_minimum_vae_size(gen_pixels, compression, pixel_upscale_method)

            latent = _encode_pixels(vae, gen_pixels, use_tiled_vae, tile_size, overlap)
            if model_patch is not None:
                # Omni reference_latents doubles the batch in Lumina2; Z-Image ControlNet
                # patches expect batch=1 → RuntimeError [1] vs [2]. Use plain CLIP + img2img.
                pass_positive = _build_plain_text_conditioning(clip, omni_prompt)
                pass_negative = _zero_out_conditioning(pass_positive)
            else:
                pass_positive = _build_omni_conditioning(clip, latent["samples"], omni_prompt)
                pass_negative = negative
            latent = _attach_mask(latent, mask, gen_pixels.shape[1], gen_pixels.shape[2])
            _validate_vae_latent_compatibility(vae, work_model, latent)

            pass_model = work_model
            if model_patch is not None:
                control_src = control_image if control_image is not None else gen_pixels
                inpaint_src = gen_pixels if mask is not None else None
                pass_model = _apply_zimage_controlnet(
                    work_model,
                    model_patch,
                    vae,
                    control_src,
                    control_strength,
                    mask=mask,
                    inpaint_image=inpaint_src,
                )

            if free_vram_between_steps:
                _unload_for_sampling(vae, work_model, unload_all_models=unload_all)
                if clip is not None and hasattr(clip, "patcher"):
                    _unload_patcher(clip.patcher)

            (latent,) = common_ksampler(
                pass_model,
                seed,
                sample_steps,
                sample_cfg,
                sampler_name,
                scheduler,
                pass_positive,
                pass_negative,
                latent,
                denoise=denoise,
            )

            if goal_preset.get("two_pass"):
                pass2_denoise = float(goal_preset.get("denoise_pass2", 0.35))
                pass2_steps = int(goal_preset.get("pass2_steps", max(8, sample_steps // 2)))
                logger.info(
                    "ZiT-Upscale: instruct two-pass pass2 denoise=%.2f steps=%s",
                    pass2_denoise,
                    pass2_steps,
                )
                (latent,) = common_ksampler(
                    pass_model,
                    seed + 1,
                    pass2_steps,
                    sample_cfg,
                    sampler_name,
                    scheduler,
                    pass_positive,
                    pass_negative,
                    latent,
                    denoise=pass2_denoise,
                )

            if free_vram_between_steps:
                _unload_for_decode(work_model)

            output_image = _decode_latent(
                vae,
                latent,
                use_tiled_vae,
                tile_size,
                overlap,
            )
            return (output_image, latent)

        pixels, mode, scale_by = _apply_quality_upscale_pixels(
            pixels,
            mode,
            scale_by,
            pixel_upscale_method,
            prompt_goal,
            upscale_model=upscale_model,
        )

        passes = max(1, upscale_passes) if mode == "progressive" else 1

        latent = None
        current_pixels = pixels

        for pass_index in range(passes):
            if mode == "progressive" and scale_by > 1.0:
                current_pixels = _scale_image_pixels(
                    current_pixels,
                    pixel_upscale_method,
                    scale_by,
                )

            latent = _encode_pixels(
                vae,
                current_pixels,
                use_tiled_vae,
                tile_size,
                overlap,
            )
            latent = _attach_mask(
                latent,
                mask,
                current_pixels.shape[1],
                current_pixels.shape[2],
            )
            _validate_vae_latent_compatibility(vae, work_model, latent)
            reference_latent = latent["samples"].clone()

            if mode not in ("refine", "progressive") and scale_by > 1.0:
                latent = _upscale_latent(latent, latent_upscale_method, scale_by)

            if free_vram_between_steps:
                _unload_for_sampling(vae, work_model, unload_all_models=unload_all)

            if positive is not None:
                pass_positive = positive
            elif clip is not None and positive_prompt.strip():
                pass_positive = _build_plain_text_conditioning(clip, positive_prompt)
            else:
                raise RuntimeError("ZiT-Upscale: connect positive conditioning or CLIP + positive_prompt.")

            if model_patch is not None:
                pass_negative = _zero_out_conditioning(pass_positive)
            else:
                pass_negative = negative
                pass_positive, pass_negative = _apply_reference_conditioning(
                    pass_positive,
                    pass_negative,
                    reference_latent,
                    use_reference_cond,
                )

            pass_model = work_model
            if model_patch is not None:
                control_src = control_image if control_image is not None else current_pixels
                pass_model = _apply_zimage_controlnet(
                    work_model,
                    model_patch,
                    vae,
                    control_src,
                    control_strength,
                    mask=mask,
                )

            if free_vram_between_steps:
                _unload_for_sampling(vae, work_model, unload_all_models=unload_all)
                if clip is not None and hasattr(clip, "patcher"):
                    _unload_patcher(clip.patcher)

            pass_seed = seed + pass_index if passes > 1 else seed
            (latent,) = common_ksampler(
                pass_model,
                pass_seed,
                sample_steps,
                sample_cfg,
                sampler_name,
                scheduler,
                pass_positive,
                pass_negative,
                latent,
                denoise=denoise,
            )

            if pass_index < passes - 1:
                if free_vram_between_steps:
                    _unload_for_decode(work_model)
                current_pixels = _decode_latent(
                    vae,
                    latent,
                    use_tiled_vae,
                    tile_size,
                    overlap,
                )
                if free_vram_between_steps:
                    _unload_for_sampling(vae, work_model)

        if free_vram_between_steps:
            _unload_for_decode(work_model)

        output_image = _decode_latent(
            vae,
            latent,
            use_tiled_vae,
            tile_size,
            overlap,
        )
        return (output_image, latent)


class ZiTPrepareUpscaleLatent:
    """
    Prepare a latent from an existing image without sampling.

    Useful when you want full control over KSampler settings in separate nodes
    while still using the same Z-Image upscale prep logic.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "upscale_factor": (
                    UPSCALE_FACTOR_OPTIONS,
                    {
                        "default": "2x",
                        "tooltip": "Output resolution vs source. 1x refine = same size, better details.",
                    },
                ),
                "use_tiled_vae": ("BOOLEAN", {"default": True}),
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 32, "max": 4096, "step": 32}),
            },
            "optional": {
                "mask": ("MASK",),
                "max_output_side": (
                    "INT",
                    {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8},
                ),
                "target_width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "latent_upscale_method": (LATENT_UPSCALE_METHODS,),
                "pixel_scale_by": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05}),
                "pixel_upscale_method": (PIXEL_UPSCALE_METHODS,),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "prepared_image")
    FUNCTION = "prepare"
    CATEGORY = "ZiT-Upscale"

    def prepare(
        self,
        vae,
        image,
        upscale_factor,
        use_tiled_vae,
        tile_size,
        overlap,
        mask=None,
        max_output_side=0,
        target_width=0,
        target_height=0,
        latent_upscale_method="bislerp",
        pixel_scale_by=1.0,
        pixel_upscale_method="lanczos",
    ):
        preset = _resolve_upscale_factor(upscale_factor)
        scale_by = preset["scale_by"]
        mode = preset["mode"]
        if upscale_factor == "4x":
            scale_by = 4.0
            mode = "latent_upscale"

        compression = vae.spacial_compression_encode()

        pixels = _scale_image_pixels(image, pixel_upscale_method, pixel_scale_by)
        pixels = _fit_image(
            pixels,
            target_width,
            target_height,
            0,
            compression,
            pixel_upscale_method,
        )
        pixels = _cap_for_output_side(
            pixels,
            mode,
            scale_by,
            max_output_side,
            compression,
            pixel_upscale_method,
        )
        pixels = _ensure_minimum_vae_size(pixels, compression, pixel_upscale_method)

        pixels, mode, scale_by = _apply_quality_upscale_pixels(
            pixels,
            mode,
            scale_by,
            pixel_upscale_method,
        )

        latent = _encode_pixels(
            vae,
            pixels,
            use_tiled_vae,
            tile_size,
            overlap,
        )
        latent = _attach_mask(
            latent,
            mask,
            pixels.shape[1],
            pixels.shape[2],
        )

        if mode not in ("refine", "progressive") and scale_by > 1.0:
            latent = _upscale_latent(latent, latent_upscale_method, scale_by)

        return (latent, pixels)


NODE_CLASS_MAPPINGS = {
    "ZiTUpscaleRefine": ZiTUpscaleRefine,
    "ZiTPrepareUpscaleLatent": ZiTPrepareUpscaleLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZiTUpscaleRefine": "ZiT Upscale & Refine (Turbo)",
    "ZiTPrepareUpscaleLatent": "ZiT Prepare Upscale Latent",
}
