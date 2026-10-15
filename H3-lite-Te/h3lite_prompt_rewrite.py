#!/usr/bin/env python3
"""FL2VA/I2VA prompt rewrite via Gemma, using MiniMax-H3 h3-prompt-writing skill.

Supports first+last (FL2VA) and first-only (I2VA / first-frame).
"""

from __future__ import annotations

import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

_LOG = logging.getLogger("h3l_prompt_rewrite")

_DEFAULT_SKILL_ROOT = Path(__file__).resolve().parent / "skills" / "h3-prompt-writing"
SKILL_ROOT = Path(
    os.environ.get("H3L_PROMPT_REWRITE_SKILL_ROOT", str(_DEFAULT_SKILL_ROOT))
).expanduser()
GEMMA_BASE = os.environ.get("H3L_GEMMA_BASE", "http://127.0.0.1:8952").rstrip("/")
GEMMA_MODEL = os.environ.get("H3L_PROMPT_REWRITE_MODEL", "omni")
# Applies only after Gemma leaves "queued" (queue wait does not count).
REWRITE_TIMEOUT = float(os.environ.get("H3L_PROMPT_REWRITE_TIMEOUT", "600"))
REWRITE_MAX_TOKENS = int(os.environ.get("H3L_PROMPT_REWRITE_MAX_TOKENS", "4096"))
REWRITE_TEMPERATURE = float(os.environ.get("H3L_PROMPT_REWRITE_TEMPERATURE", "0.3"))
# Per HTTP call (submit/poll); independent of run timeout.
REWRITE_HTTP_TIMEOUT = float(os.environ.get("H3L_PROMPT_REWRITE_HTTP_TIMEOUT", "60"))


@lru_cache(maxsize=1)
def load_system_prompt() -> str:
    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    base = (SKILL_ROOT / "references" / "base-en.txt").read_text(encoding="utf-8")
    openai_yaml = (SKILL_ROOT / "agents" / "openai.yaml").read_text(encoding="utf-8")
    return (
        "You are executing the MiniMax H3 prompt-writing skill for FL2VA/I2VA rewrite.\n"
        "Follow the skill and base guide exactly. Output ONLY the final rewritten H3 prompt text "
        "(correct mode instruction line for I2VA or FL2VA + blank line + three core fields). "
        "No markdown fences, no commentary.\n\n"
        f"=== agents/openai.yaml ===\n{openai_yaml}\n\n"
        f"=== SKILL.md ===\n{skill}\n\n"
        f"=== references/base-en.txt ===\n{base}\n"
    )


def strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _extract_text(resp: dict[str, Any]) -> str:
    data = resp.get("data") if isinstance(resp.get("data"), dict) else {}
    for key in ("text", "content", "response", "output"):
        v = data.get(key) if data else None
        if isinstance(v, str) and v.strip():
            return v.strip()
    if isinstance(data.get("message"), dict):
        c = data["message"].get("content")
        if isinstance(c, str) and c.strip():
            return c.strip()
    choices = data.get("choices") if data else None
    if isinstance(choices, list) and choices:
        msg = (choices[0] or {}).get("message") or {}
        c = msg.get("content")
        if isinstance(c, str) and c.strip():
            return c.strip()
    raise RuntimeError(f"gemma response missing text: {json.dumps(resp, ensure_ascii=False)[:1200]}")


async def rewrite_fl2va_prompt(
    *,
    prompt: str,
    first_image_path: str | Path | None = None,
    last_image_path: str | Path | None = None,
    duration_seconds: float,
) -> dict[str, Any]:
    """Rewrite FL2VA/I2VA user prompt via Gemma gateway.

    - first + last → FL2VA path (Picture 1 @ 0s, Picture 2 @ end)
    - first only → I2VA / first-frame path (Picture 1 @ 0s, develop forward)

    Returns dict with keys: prompt, rewrite_s, gemma_id, model, variant.
    """
    first = Path(first_image_path).resolve() if first_image_path else None
    last = Path(last_image_path).resolve() if last_image_path else None
    if first is not None and not first.is_file():
        raise FileNotFoundError(f"first frame missing: {first}")
    if last is not None and not last.is_file():
        raise FileNotFoundError(f"last frame missing: {last}")
    if first is None:
        raise ValueError("rewrite requires first_frame (first-only or first+last)")

    duration = float(duration_seconds)
    content: list[dict[str, Any]] = []
    if last is not None:
        variant = "first_last"
        user_text = (
            f"Mode: FL2VA\n"
            f"Target duration: {duration:.2f} seconds\n"
            f"Picture 1 = first frame (opening). Picture 2 = last frame (ending).\n"
            f"Rewrite the multimodal request below into a MiniMax H3 FL2VA generation prompt "
            f"following the skill/base guide. Preserve dialogue/spoken copy verbatim inside <d> "
            f"in the original language. Align Picture 1 at 0.00s and Picture 2 at {duration:.2f}s.\n\n"
            f"--- ORIGINAL USER REQUEST ---\n{prompt}\n"
        )
        content = [
            {"type": "text", "text": user_text},
            {"type": "image", "image": str(first)},
            {"type": "image", "image": str(last)},
        ]
    else:
        variant = "first_only"
        user_text = (
            f"Mode: I2VA (first-frame only; no last frame)\n"
            f"Target duration: {duration:.2f} seconds\n"
            f"Picture 1 = first frame at 0.00s; develop the shot forward from this image.\n"
            f"Rewrite the multimodal request below into a MiniMax H3 I2VA generation prompt "
            f"following the skill/base guide (I2VA instruction line + three core fields). "
            f"Preserve dialogue/spoken copy verbatim inside <d> in the original language. "
            f"Fully reference <Picture 1> at 0.00s.\n\n"
            f"--- ORIGINAL USER REQUEST ---\n{prompt}\n"
        )
        content = [
            {"type": "text", "text": user_text},
            {"type": "image", "image": str(first)},
        ]

    body: dict[str, Any] = {
        "mode": "chat",
        "model": GEMMA_MODEL,
        "input": {
            "messages": [
                {"role": "system", "content": load_system_prompt()},
                {"role": "user", "content": content},
            ],
            "max_tokens": REWRITE_MAX_TOKENS,
            "temperature": REWRITE_TEMPERATURE,
        },
    }

    t0 = time.time()
    timeout = httpx.Timeout(REWRITE_HTTP_TIMEOUT, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        sub: dict[str, Any] | None = None
        last_conn_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                r = await client.post(f"{GEMMA_BASE}/gemma", json=body)
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                last_conn_err = e
                _LOG.warning(
                    "gemma rewrite submit connect fail attempt=%s/3: %s",
                    attempt,
                    e,
                )
                await _async_sleep(2.0 * attempt)
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"gemma submit HTTP {r.status_code}: {r.text[:800]}")
            sub = r.json()
            break
        if sub is None:
            raise RuntimeError(f"gemma submit connect failed after retries: {last_conn_err}")
        gid = str(sub.get("id") or "")
        if not gid:
            raise RuntimeError(f"gemma submit missing id: {sub}")

        # Queue wait is unbounded; the 600s budget starts only after status=running.
        run_started_at: float | None = None
        done: dict[str, Any] | None = None
        while True:
            try:
                pr = await client.post(f"{GEMMA_BASE}/gemma", json={"id": gid})
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                _LOG.warning("gemma rewrite poll transient %s id=%s — retry", type(e).__name__, gid)
                await _async_sleep(1.5)
                continue
            if pr.status_code >= 400:
                raise RuntimeError(f"gemma poll HTTP {pr.status_code}: {pr.text[:800]}")
            done = pr.json()
            st = str(done.get("status") or "")
            if st in {"finished", "success"}:
                break
            if st in {"failed", "error"}:
                raise RuntimeError(f"gemma rewrite failed: {json.dumps(done, ensure_ascii=False)[:1500]}")
            if st == "queued":
                await _async_sleep(1.5)
                continue
            # running (or any other non-terminal status): count against run timeout
            if run_started_at is None:
                run_started_at = time.time()
                _LOG.info(
                    "fl2va rewrite left queue | gemma=%s queue_wait_s=%.2f run_timeout_s=%s",
                    gid,
                    run_started_at - t0,
                    REWRITE_TIMEOUT,
                )
            if time.time() - run_started_at > REWRITE_TIMEOUT:
                raise TimeoutError(
                    f"gemma rewrite timed out after {REWRITE_TIMEOUT}s of execution "
                    f"(queue wait excluded) id={gid}"
                )
            await _async_sleep(1.5)

    assert done is not None
    rewritten = strip_fences(_extract_text(done))
    if not rewritten:
        raise RuntimeError("gemma rewrite returned empty prompt")
    elapsed = time.time() - t0
    queue_wait_s = (run_started_at - t0) if run_started_at is not None else 0.0
    run_s = elapsed - queue_wait_s
    _LOG.info(
        "fl2va rewrite ok | gemma=%s model=%s rewrite_s=%.2f queue_wait_s=%.2f run_s=%.2f chars=%s->%s",
        gid,
        GEMMA_MODEL,
        elapsed,
        queue_wait_s,
        run_s,
        len(prompt),
        len(rewritten),
    )
    return {
        "prompt": rewritten,
        "rewrite_s": round(elapsed, 3),
        "queue_wait_s": round(queue_wait_s, 3),
        "run_s": round(run_s, 3),
        "gemma_id": gid,
        "model": GEMMA_MODEL,
        "skill_root": str(SKILL_ROOT),
        "variant": variant,
    }


async def _async_sleep(sec: float) -> None:
    import asyncio

    await asyncio.sleep(sec)
