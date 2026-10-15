"""Build ComfyUI API prompts for MiniMax-H3 FL2VA / Ref2VA."""

from __future__ import annotations

import math
from typing import Any

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
FPS = 24

DIT_DEFAULT = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
DIT_REF2VA_DEFAULT = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
CLIP_DEFAULT = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE_DEFAULT = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE_DEFAULT = "minimax_h3_audio_vae_fp32.safetensors"
LORA_TURBO_DEFAULT = "minimax_h3_fl2v_turbo_4step_v0.1.safetensors"
LORA_REALISM_DEFAULT = "h3-realism-people-t2v-i2v-r2v.safetensors"
REALISM_TRIGGER = "r34l1sm"

# TE-Speed (H3-lite-Te) defaults — aligned with h3lite_variant_workflows tespeed bench.
TESPEED_DEFAULTS: dict[str, Any] = {
    "processing_control_value": 0.12,
    "processing_percent_1": 0.1,
    "processing_percent_2": 0.9,
    "mcs": 2,
    "device": "auto",
    "cache_depth": 0.75,
}


def _attach_tespeed(g: dict[str, Any], model_ref: list[Any], *, node_id: str = "19") -> list[Any]:
    """Insert TESpeedMiniMaxH3 after SigmaShift; return new model ref."""
    g[node_id] = {
        "class_type": "TESpeedMiniMaxH3",
        "inputs": {"model": model_ref, **TESPEED_DEFAULTS},
    }
    return [node_id, 0]


def align_frame_count(n: int) -> int:
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def duration_to_length(duration_seconds: float | int) -> int:
    return align_frame_count(max(5, round(float(duration_seconds) * FPS)))


def adapt_canvas(
    width: float,
    height: float,
    short_edge: int = BASE_SHORT_EDGE,
) -> tuple[int, int]:
    """Fit AR to a target short edge (aligned to CANVAS_MULTIPLE), with scaled pixel budget."""
    se = max(CANVAS_MULTIPLE, int(short_edge))
    max_pixels = MAX_PIXELS * (se / float(BASE_SHORT_EDGE)) ** 2
    ratio = width / height if height else 1.0
    if ratio >= 1.0:
        nom_w, nom_h = se * ratio, float(se)
    else:
        nom_w, nom_h = float(se), se / ratio
    if nom_w * nom_h > max_pixels:
        s = math.sqrt(max_pixels / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (
        max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


def resolve_wh(
    short_edge: int = 768,
    aspect_ratio: str = "16:9",
    first_frame_wh: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Resolve output canvas.

    - ``adaptive``/``auto`` + reference WH: keep image AR, set short side ≈ ``short_edge``
      (aligned to 32, within scaled pixel budget).
    - Fixed ``W:H``: short side = ``short_edge``, long side from ratio.
    """
    se = max(CANVAS_MULTIPLE, int(short_edge))
    ar = (aspect_ratio or "16:9").strip().lower()
    if ar in {"adaptive", "auto"} and first_frame_wh:
        return adapt_canvas(float(first_frame_wh[0]), float(first_frame_wh[1]), short_edge=se)
    if ":" in ar:
        a, b = ar.split(":", 1)
        try:
            rw, rh = float(a), float(b)
        except ValueError:
            rw, rh = 16.0, 9.0
    else:
        rw, rh = 16.0, 9.0
    # short_edge on the shorter side
    if rw >= rh:
        h = float(se)
        w = h * (rw / rh)
    else:
        w = float(se)
        h = w * (rh / rw)
    return adapt_canvas(w, h, short_edge=se)


def build_fl2va_prompt(
    *,
    prompt: str,
    width: int,
    height: int,
    length: int,
    seed: int = 0,
    steps: int = 25,
    scheduler: str = "simple",
    sampler_name: str = "res_multistep",
    first_image_name: str | None = None,
    last_image_name: str | None = None,
    bgm_audio_name: str | None = None,
    dit_name: str = DIT_DEFAULT,
    clip_name: str = CLIP_DEFAULT,
    video_vae_name: str = VIDEO_VAE_DEFAULT,
    audio_vae_name: str = AUDIO_VAE_DEFAULT,
    use_turbo_lora: bool = False,
    lora_name: str = LORA_TURBO_DEFAULT,
    lora_strength: float = 1.0,
    use_tespeed: bool = False,
    filename_prefix: str = "h3l/h3l",
) -> dict[str, Any]:
    """Return ComfyUI `/prompt` graph (API format)."""
    g: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": dit_name, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip_name, "type": "minimax", "device": "default"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": video_vae_name},
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": audio_vae_name},
        },
    }

    model_ref: list[Any] = ["1", 0]
    if use_turbo_lora:
        g["18"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": lora_name,
                "strength_model": float(lora_strength),
            },
        }
        model_ref = ["18", 0]

    g["6"] = {
        "class_type": "MiniMaxH3SigmaShift",
        "inputs": {
            "model": model_ref,
            "shift_video": 12.0,
            "shift_audio": 3.0,
        },
    }
    model_ref = ["6", 0]
    if use_tespeed:
        model_ref = _attach_tespeed(g, model_ref)

    i2v_inputs: dict[str, Any] = {
        "clip": ["2", 0],
        "vae": ["3", 0],
        "prompt": prompt,
        "width": int(width),
        "height": int(height),
        "length": int(length),
    }
    if first_image_name:
        g["16"] = {
            "class_type": "LoadImage",
            "inputs": {"image": first_image_name},
        }
        i2v_inputs["first_frame"] = ["16", 0]
    if last_image_name:
        g["17"] = {
            "class_type": "LoadImage",
            "inputs": {"image": last_image_name},
        }
        i2v_inputs["last_frame"] = ["17", 0]

    g["5"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": i2v_inputs}
    g["7"] = {
        "class_type": "BasicGuider",
        "inputs": {"model": model_ref, "conditioning": ["5", 0]},
    }
    g["8"] = {
        "class_type": "BasicScheduler",
        "inputs": {
            "model": model_ref,
            "scheduler": scheduler,
            "steps": int(steps),
            "denoise": 1.0,
        },
    }
    g["9"] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": sampler_name},
    }
    g["10"] = {
        "class_type": "RandomNoise",
        "inputs": {"noise_seed": int(seed)},
    }
    g["11"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["10", 0],
            "guider": ["7", 0],
            "sampler": ["9", 0],
            "sigmas": ["8", 0],
            "latent_image": ["5", 1],
        },
    }
    g["12"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["11", 0], "vae": ["3", 0]},
    }
    g["13"] = {
        "class_type": "VAEDecodeAudio",
        "inputs": {"samples": ["11", 0], "vae": ["4", 0]},
    }
    audio_ref: list[Any] = ["13", 0]
    if bgm_audio_name:
        g["400"] = {"class_type": "LoadAudio", "inputs": {"audio": bgm_audio_name}}
        audio_ref = ["400", 0]

    g["14"] = {
        "class_type": "CreateVideo",
        "inputs": {
            "images": ["12", 0],
            "fps": float(FPS),
            "audio": audio_ref,
            "bit_depth": 8,
        },
    }
    g["15"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["14", 0],
            "filename_prefix": filename_prefix,
            "format": "auto",
            "codec": "auto",
        },
    }
    return g


def ensure_realism_trigger(prompt: str, *, enabled: bool) -> str:
    """fal realism LoRA 建议 prompt 以 trigger ``r34l1sm`` 开头。"""
    t = (prompt or "").strip()
    if not enabled or not t:
        return t
    head = t[:32].lower().replace(" ", "")
    if head.startswith(REALISM_TRIGGER.lower()):
        return t
    return f"{REALISM_TRIGGER}, {t}"


def build_ref2va_prompt(
    *,
    prompt: str,
    width: int,
    height: int,
    length: int,
    ref_image_names: list[str] | None = None,
    ref_video_names: list[str] | None = None,
    ref_video_audio_names: list[str | None] | None = None,
    ref_audio_names: list[str] | None = None,
    bgm_audio_name: str | None = None,
    use_video_soundtrack: bool = True,
    seed: int = 0,
    steps: int = 25,
    scheduler: str = "simple",
    sampler_name: str = "res_multistep",
    ref_image_size: str = "match",
    dit_name: str = DIT_REF2VA_DEFAULT,
    clip_name: str = CLIP_DEFAULT,
    video_vae_name: str = VIDEO_VAE_DEFAULT,
    audio_vae_name: str = AUDIO_VAE_DEFAULT,
    use_tespeed: bool = False,
    use_realism_lora: bool = False,
    realism_lora_name: str = LORA_REALISM_DEFAULT,
    realism_lora_strength: float = 1.0,
    filename_prefix: str = "h3l/h3l",
) -> dict[str, Any]:
    """Return ComfyUI `/prompt` graph for MiniMaxH3ReferenceToVideo (Ref2VA).

    Prompt tags (1-based, per type): ``<Picture i>`` / ``<Video k>`` / ``<Audio j>``.
    Presentation order in the model: images → videos (optional paired soundtrack
    emits ``<Audio j>`` before ``<Video k>``) → standalone audios.
    """
    ref_image_names = list(ref_image_names or [])
    ref_video_names = list(ref_video_names or [])
    ref_audio_names = list(ref_audio_names or [])
    ref_video_audio_names = list(ref_video_audio_names or [])

    if not (ref_image_names or ref_video_names or ref_audio_names):
        raise ValueError(
            "ref2va requires at least one of ref_images / ref_videos / ref_audios"
        )
    if len(ref_image_names) > 9:
        raise ValueError("ref2va supports at most 9 reference images")
    if len(ref_video_names) > 3:
        raise ValueError("ref2va supports at most 3 reference videos")
    if len(ref_audio_names) > 3:
        raise ValueError("ref2va supports at most 3 standalone reference audios")
    if ref_video_audio_names and len(ref_video_audio_names) > len(ref_video_names):
        raise ValueError("ref_video_audios length cannot exceed ref_videos")

    prompt = ensure_realism_trigger(prompt, enabled=use_realism_lora)

    g: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": dit_name, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip_name, "type": "minimax", "device": "default"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": video_vae_name},
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": audio_vae_name},
        },
    }

    model_ref: list[Any] = ["1", 0]
    if use_realism_lora:
        g["18"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": realism_lora_name,
                "strength_model": float(realism_lora_strength),
            },
        }
        model_ref = ["18", 0]

    g["6"] = {
        "class_type": "MiniMaxH3SigmaShift",
        "inputs": {
            "model": model_ref,
            "shift_video": 12.0,
            "shift_audio": 3.0,
        },
    }
    model_ref = ["6", 0]
    if use_tespeed:
        model_ref = _attach_tespeed(g, model_ref)

    r2v_inputs: dict[str, Any] = {
        "clip": ["2", 0],
        "vae": ["3", 0],
        "audio_vae": ["4", 0],
        "prompt": prompt,
        "width": int(width),
        "height": int(height),
        "length": int(length),
        "ref_image_size": ref_image_size if ref_image_size in {"match", "max"} else "match",
    }
    # Autogrow API ids: ref_images.ref_image_0 .. (see ComfyUI Autogrow.TemplatePrefix)
    for i, name in enumerate(ref_image_names):
        node_id = str(100 + i)
        g[node_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
        r2v_inputs[f"ref_images.ref_image_{i}"] = [node_id, 0]

    # Videos: LoadVideo -> GetVideoComponents (images[+audio]) -> MiniMaxH3ReferenceToVideo
    for i, name in enumerate(ref_video_names):
        load_id = str(200 + i)
        comp_id = str(210 + i)
        g[load_id] = {"class_type": "LoadVideo", "inputs": {"file": name}}
        g[comp_id] = {"class_type": "GetVideoComponents", "inputs": {"video": [load_id, 0]}}
        r2v_inputs[f"ref_videos.ref_video_{i}"] = [comp_id, 0]

        explicit_audio = None
        if i < len(ref_video_audio_names):
            explicit_audio = ref_video_audio_names[i]
        if explicit_audio:
            aud_id = str(220 + i)
            g[aud_id] = {"class_type": "LoadAudio", "inputs": {"audio": explicit_audio}}
            r2v_inputs[f"ref_video_audios.ref_video_audio_{i}"] = [aud_id, 0]
        elif use_video_soundtrack:
            # Pair video's own soundtrack (GetVideoComponents audio output index 1)
            r2v_inputs[f"ref_video_audios.ref_video_audio_{i}"] = [comp_id, 1]

    for i, name in enumerate(ref_audio_names):
        node_id = str(300 + i)
        g[node_id] = {"class_type": "LoadAudio", "inputs": {"audio": name}}
        r2v_inputs[f"ref_audios.ref_audio_{i}"] = [node_id, 0]

    g["5"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": r2v_inputs}
    g["7"] = {
        "class_type": "BasicGuider",
        "inputs": {"model": model_ref, "conditioning": ["5", 0]},
    }
    g["8"] = {
        "class_type": "BasicScheduler",
        "inputs": {
            "model": model_ref,
            "scheduler": scheduler,
            "steps": int(steps),
            "denoise": 1.0,
        },
    }
    g["9"] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": sampler_name},
    }
    g["10"] = {
        "class_type": "RandomNoise",
        "inputs": {"noise_seed": int(seed)},
    }
    g["11"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["10", 0],
            "guider": ["7", 0],
            "sampler": ["9", 0],
            "sigmas": ["8", 0],
            "latent_image": ["5", 1],
        },
    }
    g["12"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["11", 0], "vae": ["3", 0]},
    }
    g["13"] = {
        "class_type": "VAEDecodeAudio",
        "inputs": {"samples": ["11", 0], "vae": ["4", 0]},
    }
    audio_ref: list[Any] = ["13", 0]
    if bgm_audio_name:
        g["400"] = {"class_type": "LoadAudio", "inputs": {"audio": bgm_audio_name}}
        audio_ref = ["400", 0]

    g["14"] = {
        "class_type": "CreateVideo",
        "inputs": {
            "images": ["12", 0],
            "fps": float(FPS),
            "audio": audio_ref,
            "bit_depth": 8,
        },
    }
    g["15"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["14", 0],
            "filename_prefix": filename_prefix,
            "format": "auto",
            "codec": "auto",
        },
    }
    return g
