"""MiniMax-H3-lite FastAPI gateway (platform submit + poll) wrapping ComfyUI.

启动:
  python gateway_h3lite_fastapi.py --port 8960

协议（docs/AI推理服务部署规范.md §5 / §2.7）:
  - POST /H3L 或 /H3  提交: {"mode":"t2va|fl2va|i2va","input":{...}}
  - POST /H3L 或 /H3  轮询: {"id":"h3lite-..."} / {"id":"h3-..."}
  - GET  /health
  （/H3 与平台 modelserver「H3 → :8960」对齐；/H3L 为 lite 专用路径）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from gateway_logging import setup_root_logging, uvicorn_log_config
from h3lite_workflow import (
    DIT_DEFAULT,
    DIT_REF2VA_DEFAULT,
    LORA_REALISM_DEFAULT,
    build_fl2va_prompt,
    build_ref2va_prompt,
    duration_to_length,
    resolve_wh,
)

# Standalone URL cache; cloud-object-storage publishing is intentionally disabled.
from shared_media_compat import (  # noqa: E402
    PrefetchError, ensure_url_cached, extract_http_urls, prefetch_urls,
    save_prefetch_failed, schedule_result_upload, oss_enabled,
)

ROOT = Path(__file__).resolve().parent
_LOG = logging.getLogger("h3l_gateway")
_GATEWAY_LOG_FILE = setup_root_logging("h3l_gateway", logger=_LOG)

LOGS = ROOT / "logs"
OUTPUTS = ROOT / "outputs"
DOWNLOAD_DIR = Path(os.environ.get("H3L_GATEWAY_DOWNLOAD_DIR", "/tmp/h3l_gateway_inputs"))
COMFY_ROOT = Path(
    os.environ.get(
        "H3L_COMFY_ROOT",
        str(ROOT / "third_party" / "ComfyUI"),
    )
).expanduser()
COMFY_INPUT = Path(os.environ.get("H3L_COMFY_INPUT", str(COMFY_ROOT / "input")))
COMFY_OUTPUT = Path(os.environ.get("H3L_COMFY_OUTPUT", str(COMFY_ROOT / "output")))

LOGS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
COMFY_INPUT.mkdir(parents=True, exist_ok=True)
COMFY_OUTPUT.mkdir(parents=True, exist_ok=True)

MAX_PENDING = int(os.environ.get("H3L_MAX_PENDING_JOBS", "16"))
JOB_TTL_SEC = int(os.environ.get("H3L_JOB_TTL_SEC", str(24 * 3600)))
JOB_TIMEOUT_SEC = int(os.environ.get("H3L_JOB_TIMEOUT_SEC", str(3 * 3600)))
COMFY_BASE = os.environ.get("H3L_COMFY_BASE", "http://127.0.0.1:8962").rstrip("/")
POLL_INTERVAL = float(os.environ.get("H3L_COMFY_POLL_INTERVAL", "5"))
# Comfy can block HTTP while busy on GPU; poll read timeout must be generous,
# and transient timeouts must not fail the whole job.
COMFY_HTTP_READ_TIMEOUT = float(os.environ.get("H3L_COMFY_HTTP_READ_TIMEOUT", "300"))
COMFY_HTTP_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=COMFY_HTTP_READ_TIMEOUT,
    write=120.0,
    pool=10.0,
)
ACCEPT_WHEN_UNHEALTHY = os.environ.get("H3L_ACCEPT_WHEN_UNHEALTHY", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MEDIA_DOWNLOAD_TIMEOUT = float(os.environ.get("H3L_MEDIA_DOWNLOAD_TIMEOUT", "300"))
MAX_MEDIA_BYTES = int(os.environ.get("H3L_MAX_MEDIA_BYTES", str(200 * 1024 * 1024)))
DEFAULT_STEPS = int(os.environ.get("H3L_DEFAULT_STEPS", "25"))
TURBO_STEPS = int(os.environ.get("H3L_TURBO_STEPS", "4"))
# H3-lite-Te: TE-Speed block-cache on by default for external API.
DEFAULT_USE_TESPEED = os.environ.get("H3L_USE_TESPEED", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# FL2VA/I2VA: rewrite prompt via Gemma + MiniMax-H3 h3-prompt-writing skill
# (first+last → FL2VA; first-only → I2VA).
DEFAULT_FL2VA_PROMPT_REWRITE = os.environ.get("H3L_FL2VA_PROMPT_REWRITE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
GEMMA_BASE = os.environ.get("H3L_GEMMA_BASE", "http://127.0.0.1:8952").rstrip("/")


@dataclass
class Job:
    id: str
    status: str
    mode: str
    created_at: float
    updated_at: float
    input: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] | None = None
    error: str | None = None
    msg: str = "queued"
    code: int = 0
    upstream_id: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    timing: dict[str, Any] = field(default_factory=dict)
    prefetch_task: asyncio.Task | None = field(default=None, repr=False)


def _gpu_mem_used_mb() -> float | None:
    """Return GPU0 used MiB (respects CUDA_VISIBLE_DEVICES remap)."""
    try:
        import subprocess

        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        lines = [x.strip() for x in out.splitlines() if x.strip()]
        if not lines:
            return None
        # After CUDA_VISIBLE_DEVICES=0, process sees physical GPU0 as device 0.
        return float(lines[0])
    except Exception:  # noqa: BLE001
        return None


def _comfy_exec_seconds(entry: dict[str, Any]) -> float | None:
    """Best-effort ComfyUI execution duration from history status messages."""
    status = entry.get("status") if isinstance(entry.get("status"), dict) else {}
    msgs = status.get("messages") or []
    t_start = None
    t_end = None
    for m in msgs:
        if not isinstance(m, (list, tuple)) or len(m) < 2:
            continue
        kind, payload = m[0], m[1]
        ts = None
        if isinstance(payload, dict):
            ts = payload.get("timestamp") or payload.get("ts")
        if kind == "execution_start" and ts is not None:
            try:
                t_start = float(ts)
            except (TypeError, ValueError):
                pass
        if kind in ("execution_success", "execution_error", "execution_interrupted") and ts is not None:
            try:
                t_end = float(ts)
            except (TypeError, ValueError):
                pass
    if t_start is not None and t_end is not None and t_end >= t_start:
        # Comfy timestamps are often milliseconds
        if t_end > 1e12:
            return (t_end - t_start) / 1000.0
        if t_end > 1e10:
            return (t_end - t_start) / 1000.0
        return float(t_end - t_start)
    return None


_jobs: dict[str, Job] = {}
_queue: asyncio.Queue[str] = asyncio.Queue()
_lock = threading.Lock()
_worker_task: asyncio.Task | None = None
_comfy_ok = False
_comfy_error: str | None = None


class SubmitBody(BaseModel):
    mode: Literal["t2va", "fl2va", "i2va", "ref2va"]
    input: dict[str, Any] = Field(default_factory=dict)


def _now() -> float:
    return time.time()


def _job_id(prefix: str = "h3lite") -> str:
    # 勿用 h3l-：与全量 H3 前缀 h3- 冲突（startswith）
    return f"{prefix}-{uuid.uuid4().hex}"


def _truthy(v: str | None) -> bool:
    return bool(v) and v.strip().lower() in {"1", "true", "yes", "on"}


def _want_fl2va_prompt_rewrite(payload: dict[str, Any]) -> bool:
    """Per-request override for FL2VA/I2VA prompt rewrite; else env default."""
    for key in ("prompt_rewrite", "rewrite_prompt", "use_prompt_rewrite"):
        if key in payload:
            return _truthy(str(payload.get(key)))
    return DEFAULT_FL2VA_PROMPT_REWRITE


def _check_ssrf(url: str) -> None:
    if not _truthy(os.environ.get("H3L_STRICT_SSRF", "1")):
        return
    host = (urlparse(url).hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}:
        raise ValueError(f"URL host blocked by SSRF policy: {host}")
    if host.startswith(("10.", "192.168.", "127.", "169.254.")):
        raise ValueError(f"URL host blocked by SSRF policy: {host}")
    if host.startswith("172."):
        try:
            second = int(host.split(".")[1])
        except (IndexError, ValueError):
            return
        if 16 <= second <= 31:
            raise ValueError(f"URL host blocked by SSRF policy: {host}")


async def _download_url(url: str, dest: Path) -> Path:
    """Download via shared_asset_cache (max 5 attempts), then copy to dest."""
    _check_ssrf(url)
    cached = await asyncio.to_thread(ensure_url_cached, url)
    if cached.stat().st_size > MAX_MEDIA_BYTES:
        raise ValueError(f"download exceeds H3L_MAX_MEDIA_BYTES={MAX_MEDIA_BYTES}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if cached.resolve() != dest.resolve():
        await asyncio.to_thread(shutil.copy2, cached, dest)
    return dest


def _prefetch_job_urls_sync(job_id: str, request_obj: dict[str, Any]) -> None:
    urls = extract_http_urls(request_obj)
    if not urls:
        return
    _LOG.info("prefetch start | job=%s urls=%s", job_id, len(urls))
    prefetch_urls(urls)
    _LOG.info("prefetch done | job=%s", job_id)


def _fail_job_prefetch(job: Job, err: PrefetchError) -> None:
    snap = save_prefetch_failed(
        service="h3l",
        job_id=job.id,
        request_obj={"mode": job.mode, "input": job.input},
        failed_urls=list(err.failed_urls),
        error=str(err),
    )
    job.status = "failed"
    job.code = 50001
    job.msg = "failed"
    job.error = f"prefetch failed: {err}; snapshot={snap}"
    job.finished_at = _now()
    job.updated_at = job.finished_at
    _LOG.error("prefetch failed | id=%s | %s", job.id, job.error)


def _schedule_oss_and_finish(job: Job, result: dict[str, Any], *, t_infer_done: float) -> None:
    """Keep status=running; upload in background; then finished/failed."""
    if not oss_enabled():
        t_done = _now()
        timing = dict(job.timing or {})
        timing["oss_upload_s"] = 0.0
        timing["finished_at"] = t_done
        timing["wall_clock_s"] = round(t_done - (job.started_at or t_done), 3)
        result = dict(result)
        result["timing"] = timing
        job.timing = timing
        job.status = "finished"
        job.msg = "finished"
        job.data = result
        job.finished_at = t_done
        job.updated_at = t_done
        return

    def _ok(enriched: dict[str, Any]) -> None:
        t_done = _now()
        timing = dict(job.timing or {})
        timing["oss_upload_s"] = round(max(0.0, t_done - t_infer_done), 3)
        timing["finished_at"] = t_done
        timing["wall_clock_s"] = round(t_done - (job.started_at or t_done), 3)
        enriched = dict(enriched)
        enriched["timing"] = timing
        with _lock:
            job.timing = timing
            job.status = "finished"
            job.msg = "finished"
            job.data = enriched
            job.finished_at = t_done
            job.updated_at = t_done
        _LOG.info(
            "job finished (async oss) | id=%s path=%s oss_s=%.1f",
            job.id,
            (enriched or {}).get("output_path"),
            timing["oss_upload_s"],
        )

    def _fail(err: str) -> None:
        with _lock:
            job.status = "failed"
            job.code = 50001
            job.msg = "failed"
            job.error = f"OSS upload failed: {err}"
            job.finished_at = _now()
            job.updated_at = job.finished_at
        _LOG.error("OSS upload failed | id=%s | %s", job.id, err)

    schedule_result_upload(
        job.id,
        result,
        service="h3l",
        on_success=_ok,
        on_failure=_fail,
        default_prefix="h3l/jobs/",
    )
    _LOG.info("oss upload scheduled | id=%s (status stays running)", job.id)


async def _materialize_image_to_comfy(ref: str, job_id: str, name: str) -> tuple[str, tuple[int, int] | None]:
    """Copy/download image into ComfyUI input/; return (filename, optional WH)."""
    from PIL import Image

    ref = (ref or "").strip()
    if not ref:
        raise ValueError(f"missing image ref: {name}")

    suffix = ".png"
    local: Path | None = None
    if ref.startswith("data:"):
        import base64

        header, b64 = ref.split(",", 1)
        if "jpeg" in header or "jpg" in header:
            suffix = ".jpg"
        elif "webp" in header:
            suffix = ".webp"
        raw = base64.b64decode(b64)
        local = DOWNLOAD_DIR / job_id / f"{name}{suffix}"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(raw)
    elif ref.startswith("http://") or ref.startswith("https://"):
        suffix = Path(urlparse(ref).path).suffix or ".png"
        local = DOWNLOAD_DIR / job_id / f"{name}{suffix}"
        await _download_url(ref, local)
    else:
        path = Path(ref)
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {ref}")
        local = path
        suffix = path.suffix or ".png"

    fname = f"{job_id}_{name}{suffix}"
    dest = COMFY_INPUT / fname
    shutil.copy2(local, dest)
    wh = None
    try:
        with Image.open(dest) as im:
            wh = im.size  # (w,h)
    except Exception:  # noqa: BLE001
        wh = None
    return fname, wh


_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}


async def _materialize_media_to_comfy(
    ref: str,
    job_id: str,
    name: str,
    *,
    kind: str,
) -> str:
    """Copy/download video or audio into ComfyUI input/; return filename."""
    ref = (ref or "").strip()
    if not ref:
        raise ValueError(f"missing {kind} ref: {name}")

    default_suffix = ".mp4" if kind == "video" else ".wav"
    allowed = _VIDEO_SUFFIXES if kind == "video" else _AUDIO_SUFFIXES
    local: Path | None = None
    if ref.startswith("http://") or ref.startswith("https://"):
        suffix = Path(urlparse(ref).path).suffix.lower() or default_suffix
        if suffix not in allowed and kind == "audio" and suffix in _VIDEO_SUFFIXES:
            # LoadAudio also accepts video containers for soundtrack extract
            pass
        elif suffix not in allowed:
            suffix = default_suffix
        local = DOWNLOAD_DIR / job_id / f"{name}{suffix}"
        await _download_url(ref, local)
    else:
        path = Path(ref)
        if not path.is_file():
            raise FileNotFoundError(f"{kind} not found: {ref}")
        local = path
        suffix = path.suffix.lower() or default_suffix

    fname = f"{job_id}_{name}{suffix}"
    dest = COMFY_INPUT / fname
    shutil.copy2(local, dest)
    return fname


def _as_str_list(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, str):
        s = val.strip()
        return [s] if s else []
    if isinstance(val, list):
        out: list[str] = []
        for x in val:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    s = str(val).strip()
    return [s] if s else []


def _as_optional_str_list(val: Any, *, length: int) -> list[str | None]:
    """Normalize parallel soundtrack list; pad/truncate to ``length``."""
    if val is None:
        return [None] * length
    if isinstance(val, str):
        items: list[Any] = [val]
    elif isinstance(val, list):
        items = list(val)
    else:
        items = [val]
    out: list[str | None] = []
    for i in range(length):
        if i >= len(items) or items[i] is None:
            out.append(None)
            continue
        s = str(items[i]).strip()
        out.append(s or None)
    return out


async def _fit_audio_to_duration(
    audio_name: str,
    job_id: str,
    duration_seconds: int,
    *,
    loop: bool = True,
) -> str:
    """Trim/pad a supplied soundtrack to the generated clip duration."""
    import subprocess

    source = COMFY_INPUT / audio_name
    fitted = DOWNLOAD_DIR / job_id / f"{job_id}_bgm_fit.wav"
    fitted.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y"]
    if loop:
        cmd += ["-stream_loop", "-1"]
    cmd += ["-i", str(source), "-t", str(max(1, int(duration_seconds))), "-vn"]
    if not loop:
        cmd += ["-af", "apad"]
    cmd += ["-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(fitted)]
    result = await asyncio.to_thread(
        subprocess.run, cmd, capture_output=True, text=True, timeout=300, check=False
    )
    if result.returncode != 0 or not fitted.is_file() or fitted.stat().st_size == 0:
        raise RuntimeError(f"failed to prepare soundtrack: {(result.stderr or '')[-800:]}")
    fitted_name = f"{job_id}_bgm_fit.wav"
    shutil.copy2(fitted, COMFY_INPUT / fitted_name)
    return fitted_name


def _video_wh_ffprobe(path: Path) -> tuple[int, int] | None:
    try:
        import json as _json
        import subprocess

        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if r.returncode != 0:
            return None
        streams = (_json.loads(r.stdout or "{}").get("streams") or [])
        if not streams:
            return None
        w, h = int(streams[0]["width"]), int(streams[0]["height"])
        if w > 0 and h > 0:
            return (w, h)
    except Exception:  # noqa: BLE001
        return None
    return None


def _purge_expired() -> None:
    """Drop finished results after TTL. Never purge queued/running jobs —
    long Comfy queues can exceed JOB_TTL_SEC before a job even starts."""
    now = _now()
    terminal = {"succeeded", "failed"}
    with _lock:
        dead = [
            jid
            for jid, j in _jobs.items()
            if j.status in terminal and now - j.updated_at > JOB_TTL_SEC
        ]
        for jid in dead:
            _jobs.pop(jid, None)


def _pending_count() -> int:
    with _lock:
        return sum(1 for j in _jobs.values() if j.status in ("queued", "running"))


async def _probe_comfy() -> bool:
    global _comfy_ok, _comfy_error
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            for path in ("/system_stats", "/queue", "/"):
                try:
                    r = await client.get(f"{COMFY_BASE}{path}")
                    if r.status_code < 500:
                        _comfy_ok = True
                        _comfy_error = None
                        return True
                except Exception:
                    continue
        _comfy_ok = False
        _comfy_error = "unreachable"
        return False
    except Exception as e:  # noqa: BLE001
        _comfy_ok = False
        _comfy_error = f"{type(e).__name__}: {e}"
        return False


def _find_output_mp4(prompt_id: str, filename_prefix: str, since: float) -> Path | None:
    """Locate newest mp4 under Comfy output matching prefix after job start."""
    # SaveVideo writes to output/<prefix>_<counter>.mp4 (subfolder from prefix)
    prefix_path = Path(filename_prefix)
    search_roots = [COMFY_OUTPUT]
    if prefix_path.parent.as_posix() not in {".", ""}:
        search_roots.append(COMFY_OUTPUT / prefix_path.parent)
    candidates: list[Path] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        for p in root.rglob("*.mp4"):
            try:
                if p.stat().st_mtime >= since - 2:
                    candidates.append(p)
            except OSError:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    stem = prefix_path.name
    for p in candidates:
        if stem in p.name:
            return p
    return candidates[0]


async def _run_comfy_job(job: Job) -> dict[str, Any]:
    payload = dict(job.input or {})
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("input.prompt is required")
    audio_track = payload.get("audio_track") if isinstance(payload.get("audio_track"), dict) else {}
    storyboard_obj = payload.get("storyboard")
    if isinstance(storyboard_obj, dict):
        if not audio_track and isinstance(storyboard_obj.get("audio_track"), dict):
            audio_track = storyboard_obj["audio_track"]
        storyboard_scenes = storyboard_obj.get("storyboard") or []
    else:
        storyboard_scenes = storyboard_obj if isinstance(storyboard_obj, list) else []
    bgm_prompts = [str(audio_track.get("bgm_prompt") or "").strip()]
    bgm_prompts.extend(
        str(scene.get("bgm_prompt") or "").strip()
        for scene in storyboard_scenes
        if isinstance(scene, dict)
    )
    bgm_prompt = "；".join(dict.fromkeys(x for x in bgm_prompts if x))
    if bgm_prompt:
        prompt += f"\n\n背景音乐要求（按解析字段还原）：{bgm_prompt}"
    prompt_original = prompt
    prompt_rewrite_meta: dict[str, Any] | None = None

    bgm_audio_ref = (
        payload.get("bgm_audio")
        or payload.get("bgm_audio_url")
        or payload.get("bgm_audio_path")
        or audio_track.get("source_audio_url")
        or audio_track.get("source_audio_path")
        or audio_track.get("bgm_audio_url")
        or audio_track.get("asset_url")
    )
    bgm_audio_loop = _truthy(str(payload.get("bgm_audio_loop", True)))
    bgm_audio_name: str | None = None

    short_edge = int(payload.get("short_edge") or 768)
    default_ar = "16:9" if job.mode == "t2va" else "adaptive"
    aspect_ratio_raw = str(payload.get("aspect_ratio") or default_ar).strip()
    aspect_ratio = aspect_ratio_raw
    duration_seconds = int(payload.get("duration_seconds") or 5)
    # Out-of-range values are clamped (not rejected): <1 → 1, >15 → 15.
    if duration_seconds < 1 or duration_seconds > 15:
        clamped = max(1, min(15, duration_seconds))
        _LOG.info(
            "duration_seconds clamped | job=%s raw=%s -> %s",
            job.id,
            duration_seconds,
            clamped,
        )
        duration_seconds = clamped
    seed = int(payload.get("seed") or 0)
    use_turbo = _truthy(str(payload.get("use_turbo") or payload.get("turbo") or ""))
    steps = int(payload.get("steps") or (TURBO_STEPS if use_turbo else DEFAULT_STEPS))
    if job.mode == "ref2va" and use_turbo:
        # Turbo LoRA is FL2V-tuned; keep flag visible but do not attach for Ref2VA DiT.
        use_turbo = False
        steps = int(payload.get("steps") or DEFAULT_STEPS)
    # Per-request override; default follows H3L_USE_TESPEED (Te on).
    if "use_tespeed" in payload or "tespeed" in payload:
        use_tespeed = _truthy(str(payload.get("use_tespeed", payload.get("tespeed"))))
    else:
        use_tespeed = DEFAULT_USE_TESPEED

    # fal Realism People LoRA（ref2va 降 AI 感）；默认关，请求显式打开
    use_realism_lora = _truthy(
        str(
            payload.get("use_realism_lora")
            or payload.get("realism_lora")
            or payload.get("use_realism")
            or ""
        )
    )
    try:
        realism_lora_strength = float(
            payload.get("realism_lora_strength")
            or payload.get("lora_strength")
            or 1.0
        )
    except (TypeError, ValueError):
        realism_lora_strength = 1.0
    realism_lora_strength = max(0.0, min(2.0, realism_lora_strength))
    realism_lora_name = str(
        payload.get("realism_lora_name") or LORA_REALISM_DEFAULT
    ).strip() or LORA_REALISM_DEFAULT
    if use_realism_lora and job.mode not in ("ref2va", "t2va", "i2va", "fl2va"):
        use_realism_lora = False

    first_name = None
    last_name = None
    first_wh = None
    last_wh = None
    ref_names: list[str] = []
    ref_video_names: list[str] = []
    ref_video_audio_names: list[str | None] = []
    ref_audio_names: list[str] = []
    use_video_soundtrack = True
    canvas_wh = None

    if job.mode == "ref2va":
        refs = payload.get("ref_images") or payload.get("references") or []
        if isinstance(refs, str):
            refs = [refs]
        if not isinstance(refs, list) or not refs:
            # also accept people/background convenience fields
            people = payload.get("people") or payload.get("subject")
            background = payload.get("background") or payload.get("scene")
            refs = [x for x in (people, background) if x]
        refs = [x for x in (refs or []) if x]

        videos = _as_str_list(
            payload.get("ref_videos")
            or payload.get("videos")
            or payload.get("references_video")
        )
        standalone_audios = _as_str_list(
            payload.get("ref_audios")
            or payload.get("audios")
            or payload.get("references_audio")
        )
        if "use_video_soundtrack" in payload or "extract_video_audio" in payload:
            use_video_soundtrack = _truthy(
                str(payload.get("use_video_soundtrack", payload.get("extract_video_audio")))
            )
        else:
            use_video_soundtrack = True

        if not refs and not videos and not standalone_audios:
            raise ValueError(
                "ref2va requires at least one of input.ref_images (or people+background), "
                "input.ref_videos, input.ref_audios"
            )

        for i, ref in enumerate(refs):
            name, wh = await _materialize_image_to_comfy(str(ref), job.id, f"ref{i}")
            ref_names.append(name)
            # Prefer last ref image (usually environment) for adaptive canvas when auto.
            if wh:
                canvas_wh = wh

        paired_raw = payload.get("ref_video_audios") or payload.get("video_audios")
        paired = _as_optional_str_list(paired_raw, length=len(videos))
        for i, vref in enumerate(videos):
            vname = await _materialize_media_to_comfy(str(vref), job.id, f"vid{i}", kind="video")
            ref_video_names.append(vname)
            if canvas_wh is None:
                wh = _video_wh_ffprobe(COMFY_INPUT / vname)
                if wh:
                    canvas_wh = wh
            paired_a = paired[i] if i < len(paired) else None
            if paired_a:
                aname = await _materialize_media_to_comfy(
                    str(paired_a), job.id, f"vid{i}_aud", kind="audio"
                )
                ref_video_audio_names.append(aname)
            else:
                ref_video_audio_names.append(None)

        for i, aref in enumerate(standalone_audios):
            aname = await _materialize_media_to_comfy(str(aref), job.id, f"aud{i}", kind="audio")
            ref_audio_names.append(aname)

        dit_name = str(payload.get("dit_name") or DIT_REF2VA_DEFAULT)
        ref_image_size = str(payload.get("ref_image_size") or "match")
    elif job.mode in ("fl2va", "i2va"):
        first = payload.get("first_frame") or payload.get("image") or payload.get("first_frame_url")
        last = payload.get("last_frame") or payload.get("last_frame_url")
        if not first and not last:
            raise ValueError("fl2va/i2va requires first_frame and/or last_frame")
        if first:
            first_name, first_wh = await _materialize_image_to_comfy(str(first), job.id, "first")
        if last:
            last_name, last_wh = await _materialize_image_to_comfy(str(last), job.id, "last")
        dit_name = str(payload.get("dit_name") or DIT_DEFAULT)
        ref_image_size = "match"
    else:
        dit_name = str(payload.get("dit_name") or DIT_DEFAULT)
        ref_image_size = "match"

    if bgm_audio_ref:
        bgm_source_name = await _materialize_media_to_comfy(
            str(bgm_audio_ref), job.id, "bgm", kind="audio"
        )
        bgm_audio_name = await _fit_audio_to_duration(
            bgm_source_name,
            job.id,
            duration_seconds,
            loop=bgm_audio_loop,
        )
        _LOG.info(
            "external soundtrack attached | job=%s source=%s loop=%s duration=%ss",
            job.id,
            bgm_source_name,
            bgm_audio_loop,
            duration_seconds,
        )

    # FL2VA/I2VA：有首帧即按官方 h3-prompt-writing skill 经 Gemma 重写（可关闭）。
    # 首+尾 → FL2VA；仅首帧 → I2VA。
    if (
        job.mode == "fl2va"
        and first_name
        and _want_fl2va_prompt_rewrite(payload)
    ):
        from h3lite_prompt_rewrite import rewrite_fl2va_prompt

        first_path = COMFY_INPUT / first_name
        last_path = (COMFY_INPUT / last_name) if last_name else None
        _LOG.info(
            "fl2va prompt rewrite start | job=%s gemma=%s first=%s last=%s variant=%s",
            job.id,
            GEMMA_BASE,
            first_name,
            last_name or "-",
            "first_last" if last_name else "first_only",
        )
        try:
            rw = await rewrite_fl2va_prompt(
                prompt=prompt,
                first_image_path=first_path,
                last_image_path=last_path,
                duration_seconds=float(duration_seconds),
            )
        except Exception as e:  # noqa: BLE001
            _LOG.exception("fl2va prompt rewrite failed | job=%s", job.id)
            raise RuntimeError(f"fl2va prompt rewrite failed: {type(e).__name__}: {e}") from e
        prompt = str(rw["prompt"])
        prompt_rewrite_meta = {
            "enabled": True,
            "variant": rw.get("variant") or ("first_last" if last_name else "first_only"),
            "rewrite_s": rw.get("rewrite_s"),
            "queue_wait_s": rw.get("queue_wait_s"),
            "run_s": rw.get("run_s"),
            "gemma_id": rw.get("gemma_id"),
            "model": rw.get("model"),
            "skill_root": rw.get("skill_root"),
            "chars_before": len(prompt_original),
            "chars_after": len(prompt),
        }
        # Persist for debugging / compare.
        try:
            out_dir = OUTPUTS / job.id
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "prompt_before.txt").write_text(prompt_original, encoding="utf-8")
            (out_dir / "prompt_after.txt").write_text(prompt, encoding="utf-8")
        except OSError:
            pass
        _LOG.info(
            "fl2va prompt rewrite done | job=%s rewrite_s=%s chars=%s->%s",
            job.id,
            prompt_rewrite_meta.get("rewrite_s"),
            prompt_rewrite_meta.get("chars_before"),
            prompt_rewrite_meta.get("chars_after"),
        )

    # fl2va/i2va only: with an input image, follow its AR and use request short_edge.
    # Ignore fixed client aspect_ratio such as "16:9" so portrait inputs stay portrait.
    # ref2va keeps prior behavior (client aspect_ratio + last-ref guide when adaptive).
    guide_wh = canvas_wh or first_wh or last_wh
    if job.mode in ("fl2va", "i2va") and guide_wh is not None:
        if aspect_ratio_raw.lower() not in {"adaptive", "auto"}:
            _LOG.info(
                "canvas follow image AR | job=%s mode=%s client_ar=%s image_wh=%sx%s short_edge=%s",
                job.id,
                job.mode,
                aspect_ratio_raw,
                guide_wh[0],
                guide_wh[1],
                short_edge,
            )
        aspect_ratio = "adaptive"
    width, height = resolve_wh(short_edge, aspect_ratio, guide_wh)
    _LOG.info(
        "canvas resolved | job=%s short_edge=%s ar=%s -> %sx%s",
        job.id,
        short_edge,
        aspect_ratio,
        width,
        height,
    )
    length = duration_to_length(duration_seconds)
    filename_prefix = f"h3l/{job.id}"
    t_prep_done = _now()
    if job.mode == "ref2va":
        graph = build_ref2va_prompt(
            prompt=prompt,
            width=width,
            height=height,
            length=length,
            seed=seed,
            steps=steps,
            ref_image_names=ref_names,
            ref_video_names=ref_video_names,
            ref_video_audio_names=ref_video_audio_names,
            ref_audio_names=ref_audio_names,
            bgm_audio_name=bgm_audio_name,
            use_video_soundtrack=use_video_soundtrack,
            ref_image_size=ref_image_size,
            dit_name=dit_name,
            use_tespeed=use_tespeed,
            use_realism_lora=use_realism_lora,
            realism_lora_name=realism_lora_name,
            realism_lora_strength=realism_lora_strength,
            filename_prefix=filename_prefix,
        )
    else:
        # fl2va/i2va/t2va：可选挂 realism LoRA（与 turbo 互斥时优先 turbo 的步数逻辑已在上方处理）
        graph = build_fl2va_prompt(
            prompt=prompt,
            width=width,
            height=height,
            length=length,
            seed=seed,
            steps=steps,
            first_image_name=first_name,
            last_image_name=last_name,
            bgm_audio_name=bgm_audio_name,
            use_turbo_lora=use_turbo,
            dit_name=dit_name,
            use_tespeed=use_tespeed,
            filename_prefix=filename_prefix,
        )
        if use_realism_lora and not use_turbo:
            # 在 FL 图上额外插入 realism：复用 ref2va 同款节点插入方式重建
            from h3lite_workflow import ensure_realism_trigger

            graph = build_fl2va_prompt(
                prompt=ensure_realism_trigger(prompt, enabled=True),
                width=width,
                height=height,
                length=length,
                seed=seed,
                steps=steps,
                first_image_name=first_name,
                last_image_name=last_name,
                bgm_audio_name=bgm_audio_name,
                use_turbo_lora=True,  # 借用 LoraLoader 槽位加载 realism
                lora_name=realism_lora_name,
                lora_strength=realism_lora_strength,
                dit_name=dit_name,
                use_tespeed=use_tespeed,
                filename_prefix=filename_prefix,
            )

    _LOG.info(
        "graph built | job=%s mode=%s steps=%s use_turbo=%s use_tespeed=%s use_realism_lora=%s strength=%s",
        job.id,
        job.mode,
        steps,
        use_turbo,
        use_tespeed,
        use_realism_lora,
        realism_lora_strength if use_realism_lora else None,
    )

    t_job_start = job.started_at or _now()
    mem_peak = _gpu_mem_used_mb()
    started = _now()
    async with httpx.AsyncClient(timeout=COMFY_HTTP_TIMEOUT) as client:
        r = await client.post(f"{COMFY_BASE}/prompt", json={"prompt": graph, "client_id": job.id})
        if r.status_code >= 400:
            raise RuntimeError(f"ComfyUI /prompt HTTP {r.status_code}: {r.text[:800]}")
        body = r.json()
        prompt_id = str(body.get("prompt_id") or "")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI /prompt missing prompt_id: {body}")
        job.upstream_id = prompt_id
        t_comfy_queued = _now()
        _LOG.info("comfy queued | job=%s prompt_id=%s", job.id, prompt_id)

        deadline = started + JOB_TIMEOUT_SEC
        entry: dict[str, Any] | None = None
        while _now() < deadline:
            cur_mem = _gpu_mem_used_mb()
            if cur_mem is not None:
                mem_peak = max(mem_peak or 0.0, cur_mem)
            try:
                hr = await client.get(f"{COMFY_BASE}/history/{prompt_id}")
            except (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError, httpx.WriteTimeout) as e:
                # Comfy's HTTP thread can stall under heavy GPU load; keep waiting.
                _LOG.warning(
                    "comfy poll transient %s | job=%s prompt_id=%s — continue waiting",
                    type(e).__name__,
                    job.id,
                    prompt_id,
                )
                await asyncio.sleep(POLL_INTERVAL)
                continue
            hist = hr.json() if hr.status_code == 200 else {}
            entry = hist.get(prompt_id) if isinstance(hist, dict) else None
            if isinstance(entry, dict):
                status = (entry.get("status") or {}) if isinstance(entry.get("status"), dict) else {}
                if status.get("status_str") == "error" or status.get("completed") is False and status.get("messages"):
                    msgs = status.get("messages") or []
                    err_txt = json.dumps(msgs, ensure_ascii=False)[:2000]
                    if "error" in err_txt.lower() or status.get("status_str") == "error":
                        raise RuntimeError(f"ComfyUI job failed: {err_txt}")
                outputs = entry.get("outputs") or {}
                if outputs or status.get("completed") is True:
                    mp4 = _find_output_mp4(prompt_id, filename_prefix, started)
                    if mp4 and mp4.is_file():
                        t_comfy_done = _now()
                        out_dir = OUTPUTS / job.id
                        out_dir.mkdir(parents=True, exist_ok=True)
                        dest = out_dir / mp4.name
                        shutil.copy2(mp4, dest)
                        duration_s = length / 24.0
                        comfy_s = _comfy_exec_seconds(entry)
                        if comfy_s is None:
                            comfy_s = max(0.0, t_comfy_done - t_comfy_queued)
                        t_infer_done = _now()
                        result = {
                            "output_path": str(dest),
                            "output_dir": str(out_dir),
                            "comfy_prompt_id": prompt_id,
                            "mode": job.mode,
                            "dit_name": dit_name,
                            "width": width,
                            "height": height,
                            "length": length,
                            "duration_seconds": round(duration_s, 4),
                            "steps": steps,
                            "use_turbo": use_turbo,
                            "use_tespeed": use_tespeed,
                            "use_realism_lora": use_realism_lora,
                            "realism_lora_name": realism_lora_name if use_realism_lora else None,
                            "realism_lora_strength": realism_lora_strength if use_realism_lora else None,
                            "bgm_audio_used": bool(bgm_audio_name),
                            "bgm_audio_loop": bgm_audio_loop if bgm_audio_name else None,
                            "bgm_prompt": bgm_prompt or None,
                            "variant": "tespeed" if use_tespeed else "baseline",
                            "prompt_rewrite": prompt_rewrite_meta,
                            "prompt_original": prompt_original if prompt_rewrite_meta else None,
                            "prompt": prompt if prompt_rewrite_meta else None,
                            "ref_images": ref_names or None,
                            "ref_videos": ref_video_names or None,
                            "ref_video_audios": [
                                x for x in ref_video_audio_names if x
                            ]
                            or None,
                            "ref_audios": ref_audio_names or None,
                            "use_video_soundtrack": use_video_soundtrack
                            if job.mode == "ref2va"
                            else None,
                        }
                        # OSS is scheduled asynchronously by the worker (status stays running).
                        queue_wait_s = max(0.0, (job.started_at or t_job_start) - job.created_at)
                        rewrite_s = float((prompt_rewrite_meta or {}).get("rewrite_s") or 0.0)
                        timing = {
                            "created_at": job.created_at,
                            "started_at": job.started_at or t_job_start,
                            "finished_at": None,
                            "queue_wait_s": round(queue_wait_s, 3),
                            "prepare_s": round(max(0.0, t_prep_done - t_job_start), 3),
                            "prompt_rewrite_s": round(rewrite_s, 3) if prompt_rewrite_meta else None,
                            "comfy_queue_to_done_s": round(comfy_s, 3),
                            "comfy_poll_wall_s": round(max(0.0, t_comfy_done - t_comfy_queued), 3),
                            "oss_upload_s": None,
                            "inference_time_s": round(comfy_s, 3),
                            "wall_clock_s": round(t_infer_done - t_job_start, 3),
                            "sec_per_video_sec": round(comfy_s / duration_s, 3) if duration_s > 0 else None,
                            "peak_memory_mb": round(mem_peak, 1) if mem_peak is not None else None,
                        }
                        job.timing = timing
                        result["timing"] = timing
                        _LOG.info(
                            "infer done | job=%s queue=%.1fs prepare=%.1fs rewrite=%s comfy=%.1fs peak_mem=%.0fMB (oss async)",
                            job.id,
                            timing["queue_wait_s"],
                            timing["prepare_s"],
                            f"{rewrite_s:.1f}s" if prompt_rewrite_meta else "off",
                            timing["inference_time_s"],
                            timing["peak_memory_mb"] or -1,
                        )
                        return result
            await asyncio.sleep(POLL_INTERVAL)

    raise TimeoutError(f"ComfyUI job timed out after {JOB_TIMEOUT_SEC}s (prompt_id={prompt_id})")


async def _worker() -> None:
    while True:
        jid = await _queue.get()
        with _lock:
            job = _jobs.get(jid)
            if not job or job.status != "queued":
                _queue.task_done()
                continue
            if _apply_force_fail(job):
                _queue.task_done()
                continue
        # Await queued-stage prefetch before taking the inference lock.
        try:
            if job.prefetch_task is not None:
                await job.prefetch_task
            else:
                await asyncio.to_thread(
                    _prefetch_job_urls_sync,
                    job.id,
                    {"mode": job.mode, "input": job.input},
                )
        except PrefetchError as e:
            with _lock:
                if job.status == "queued":
                    _fail_job_prefetch(job, e)
            _queue.task_done()
            continue
        except Exception as e:  # noqa: BLE001
            pe = PrefetchError(f"{type(e).__name__}: {e}", failed_urls=extract_http_urls(job.input))
            with _lock:
                if job.status == "queued":
                    _fail_job_prefetch(job, pe)
            _queue.task_done()
            continue

        with _lock:
            if job.status != "queued":
                _queue.task_done()
                continue
            job.status = "running"
            job.msg = "running"
            job.started_at = _now()
            job.updated_at = job.started_at
        _LOG.info("job start | id=%s mode=%s", job.id, job.mode)
        try:
            data = await _run_comfy_job(job)
            t_infer_done = _now()
            # Keep running; schedule OSS; free this worker for the next GPU job.
            with _lock:
                job.data = data  # partial (no oss yet); pollers still see running
                job.updated_at = t_infer_done
            _schedule_oss_and_finish(job, data, t_infer_done=t_infer_done)
        except Exception as e:  # noqa: BLE001
            with _lock:
                job.status = "failed"
                job.code = 50001
                job.msg = "failed"
                job.error = f"{type(e).__name__}: {e}"
                job.finished_at = _now()
                job.updated_at = job.finished_at
            _LOG.exception("job failed | id=%s", job.id)
        finally:
            _queue.task_done()


app = FastAPI(title="MiniMax-H3-lite Gateway", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    global _worker_task
    _LOG.info("H3-lite gateway starting | comfy=%s log=%s", COMFY_BASE, _GATEWAY_LOG_FILE)
    await _probe_comfy()
    if _worker_task is None:
        _worker_task = asyncio.create_task(_worker())


@app.get("/health")
async def health() -> dict[str, Any]:
    ok = await _probe_comfy()
    return {
        "ok": True,
        "service": "H3L",
        "status": "up" if ok else "degraded",
        "variant": "tespeed" if DEFAULT_USE_TESPEED else "baseline",
        "use_tespeed": DEFAULT_USE_TESPEED,
        "fl2va_prompt_rewrite": DEFAULT_FL2VA_PROMPT_REWRITE,
        "gemma_base": GEMMA_BASE,
        "shared_media": True,
        "oss_enabled": oss_enabled(),
        "comfy": ok,
        "comfy_error": _comfy_error,
        "comfy_base": COMFY_BASE,
        "pending": _pending_count(),
        "max_pending": MAX_PENDING,
        "log_file": str(_GATEWAY_LOG_FILE),
    }


_FORCE_FAIL_FILE = Path(os.environ.get("H3L_FORCE_FAIL_FILE", "/tmp/h3l_force_fail_jobs.json"))


def _force_fail_map() -> dict[str, str]:
    try:
        raw = json.loads(_FORCE_FAIL_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if k:
            out[str(k)] = str(v or "cancelled by operator")
    return out


def _apply_force_fail(job: Job) -> bool:
    """If job id is in force-fail file, mutate to failed and return True."""
    err = _force_fail_map().get(job.id)
    if not err:
        return False
    if job.status == "failed" and job.code == 49901:
        return True
    job.status = "failed"
    job.code = 49901
    job.msg = "failed"
    job.error = err
    job.data = None
    job.timing = None
    job.finished_at = _now()
    job.updated_at = job.finished_at
    _LOG.warning("force-fail applied | id=%s | %s", job.id, err)
    return True


def _job_response(job: Job) -> dict[str, Any]:
    _apply_force_fail(job)
    out: dict[str, Any] = {
        "id": job.id,
        "code": job.code,
        "msg": job.msg,
        "status": job.status,
    }
    if job.data is not None:
        out["data"] = job.data
    if job.error:
        out["error"] = job.error
    if job.upstream_id:
        out["upstream_id"] = job.upstream_id
    timing = job.timing or ((job.data or {}).get("timing") if isinstance(job.data, dict) else None)
    if timing:
        out["timing"] = timing
    return out


@app.api_route("/H3L", methods=["POST"])
@app.api_route("/h3l", methods=["POST"])
@app.api_route("/H3", methods=["POST"])
@app.api_route("/h3", methods=["POST"])
async def h3l_endpoint(request: Request) -> JSONResponse:
    _purge_expired()
    # Platform modelserver routes /H3 → :8960 and expects job ids with h3- prefix.
    path = (request.url.path or "").rstrip("/") or "/"
    id_prefix = "h3" if path.lower() in {"/h3"} else "h3lite"
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"code": 40001, "status": "bad_request", "msg": "invalid JSON", "error": "invalid JSON"}, status_code=400)

    if isinstance(body, dict) and set(body.keys()) <= {"id"} and "id" in body:
        jid = str(body["id"])
        with _lock:
            job = _jobs.get(jid)
        if not job:
            return JSONResponse(
                {"id": jid, "code": 40401, "status": "not_found", "msg": "not_found", "error": "job not found or expired"},
                status_code=404,
            )
        return JSONResponse(_job_response(job))

    try:
        parsed = SubmitBody.model_validate(body)
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            {"code": 40001, "status": "bad_request", "msg": "bad_request", "error": str(e)},
            status_code=400,
        )

    if not ACCEPT_WHEN_UNHEALTHY:
        if not await _probe_comfy():
            return JSONResponse(
                {
                    "code": 50301,
                    "status": "unavailable",
                    "msg": "unavailable",
                    "error": f"ComfyUI not ready: {_comfy_error}",
                },
                status_code=503,
            )

    if _pending_count() >= MAX_PENDING:
        return JSONResponse(
            {"code": 42901, "status": "queue_full", "msg": "queue_full", "error": f"max_pending_jobs={MAX_PENDING}"},
            status_code=429,
        )

    job = Job(
        id=_job_id(id_prefix),
        status="queued",
        mode=parsed.mode,
        created_at=_now(),
        updated_at=_now(),
        input=dict(parsed.input or {}),
        msg="queued",
    )
    with _lock:
        _jobs[job.id] = job
    # §2.8: start prefetch while queued (does not hold inference lock).
    async def _prefetch_and_maybe_fail() -> None:
        try:
            await asyncio.to_thread(
                _prefetch_job_urls_sync,
                job.id,
                {"mode": job.mode, "input": job.input},
            )
        except PrefetchError as e:
            with _lock:
                j = _jobs.get(job.id)
                if j and j.status == "queued":
                    _fail_job_prefetch(j, e)
            raise
        except Exception as e:  # noqa: BLE001
            pe = PrefetchError(f"{type(e).__name__}: {e}", failed_urls=extract_http_urls(job.input))
            with _lock:
                j = _jobs.get(job.id)
                if j and j.status == "queued":
                    _fail_job_prefetch(j, pe)
            raise pe

    job.prefetch_task = asyncio.create_task(_prefetch_and_maybe_fail())
    await _queue.put(job.id)
    _LOG.info("job queued | id=%s mode=%s path=%s", job.id, job.mode, path)
    return JSONResponse(_job_response(job))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("H3L_GATEWAY_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GATEWAY_H3L_PORT", "8960")))
    args = parser.parse_args()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_config=uvicorn_log_config("h3l_gateway"),
        access_log=True,
    )


if __name__ == "__main__":
    main()
