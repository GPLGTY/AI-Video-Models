"""Local-only replacements for the originating deployment's private shared_media package."""
from __future__ import annotations
import hashlib, json, os, tempfile, urllib.request
from pathlib import Path

CACHE = Path(os.environ.get("H3L_URL_CACHE", str(Path(tempfile.gettempdir()) / "h3lite-url-cache")))
CACHE.mkdir(parents=True, exist_ok=True)
class PrefetchError(RuntimeError):
    def __init__(self, message, failed_urls=()):
        super().__init__(message); self.failed_urls=list(failed_urls)
def extract_http_urls(value):
    out=[]
    def walk(x):
        if isinstance(x, dict):
            for v in x.values(): walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x: walk(v)
        elif isinstance(x, str) and x.startswith(("http://", "https://")): out.append(x)
    walk(value); return list(dict.fromkeys(out))
def ensure_url_cached(url):
    name=hashlib.sha256(url.encode()).hexdigest()
    path=CACHE/name
    if path.is_file(): return path
    req=urllib.request.Request(url, headers={"User-Agent":"H3-lite-Te/1.0"})
    with urllib.request.urlopen(req, timeout=int(os.environ.get("H3L_MEDIA_DOWNLOAD_TIMEOUT", "300"))) as resp:
        data=resp.read(int(os.environ.get("H3L_MAX_MEDIA_BYTES", str(200*1024*1024)))+1)
    if len(data)>int(os.environ.get("H3L_MAX_MEDIA_BYTES", str(200*1024*1024))): raise ValueError("download exceeds H3L_MAX_MEDIA_BYTES")
    path.write_bytes(data); return path
def prefetch_urls(urls):
    failed=[]
    for url in urls:
        try: ensure_url_cached(url)
        except Exception as exc: failed.append((url, exc))
    if failed: raise PrefetchError("URL prefetch failed: "+", ".join(u for u,_ in failed), [u for u,_ in failed])
def save_prefetch_failed(**kwargs):
    out=Path(__file__).resolve().parent/"logs"/"prefetch-failures.jsonl"; out.parent.mkdir(exist_ok=True)
    with out.open("a", encoding="utf-8") as f: f.write(json.dumps(kwargs, ensure_ascii=False, default=str)+"\n")
    return str(out)
def oss_enabled(): return False
def schedule_result_upload(*args, **kwargs): raise RuntimeError("Object-storage upload is intentionally disabled in this standalone repository")
