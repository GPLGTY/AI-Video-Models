"""Weekly rotating file logging for MiniMax-H3-lite gateway.

Natural week (Mon 00:00 – Sun 23:59), Asia/Shanghai, per platform §6.4:
  h3_gateway_YYYY-MM-DD_YYYY-MM-DD.partNN.log
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date, datetime, timedelta
from logging import Handler, LogRecord
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

LOG_TS_FMT = "%Y-%m-%d %H:%M:%S"
_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_TZ = ZoneInfo("Asia/Shanghai")


def repo_log_dir() -> Path:
    env = os.environ.get("H3L_LOG_DIR", "").strip()
    if env:
        directory = Path(env).expanduser().resolve()
    else:
        directory = Path(__file__).resolve().parent / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _now_shanghai() -> datetime:
    return datetime.now(_TZ)


def week_bounds(*, when: datetime | None = None) -> tuple[date, date]:
    dt = when or _now_shanghai()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_TZ)
    else:
        dt = dt.astimezone(_TZ)
    d = dt.date()
    monday = d - timedelta(days=d.weekday())
    sunday = monday + timedelta(days=6)
    return monday, sunday


def weekly_log_stem(prefix: str, *, when: datetime | None = None) -> str:
    monday, sunday = week_bounds(when=when)
    return f"{prefix}_{monday.isoformat()}_{sunday.isoformat()}"


def weekly_log_filename_part(prefix: str, part: int, *, when: datetime | None = None) -> str:
    return f"{weekly_log_stem(prefix, when=when)}.part{part:02d}.log"


def weekly_log_path(prefix: str, *, log_dir: Path | None = None, part: int = 1) -> Path:
    directory = log_dir or repo_log_dir()
    return directory / weekly_log_filename_part(prefix, part=part)


def daily_log_path(prefix: str, *, log_dir: Path | None = None, part: int = 1) -> Path:
    """Alias kept for start-script compatibility; uses weekly naming."""
    return weekly_log_path(prefix, log_dir=log_dir, part=part)


def _log_max_bytes() -> int:
    raw = os.environ.get("H3L_LOG_MAX_MB", "50").strip()
    try:
        mb = max(1, int(raw))
    except ValueError:
        mb = 50
    return mb * 1024 * 1024


class WeeklyLogFileHandler(Handler):
    def __init__(
        self,
        prefix: str = "h3l_gateway",
        *,
        log_dir: str | Path | None = None,
        encoding: str = "utf-8",
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.log_dir = Path(log_dir).expanduser().resolve() if log_dir else repo_log_dir()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.encoding = encoding
        self._week_key: tuple[date, date] | None = None
        self._part = 1
        self._max_bytes = _log_max_bytes()
        self._stream = None

    def _week_key_now(self) -> tuple[date, date]:
        return week_bounds()

    def _current_stem(self) -> str:
        return weekly_log_stem(self.prefix)

    def _path_for_part(self, part: int) -> Path:
        return self.log_dir / weekly_log_filename_part(self.prefix, part=part)

    def _detect_latest_part(self) -> int:
        stem = self._current_stem()
        latest = 0
        for p in self.log_dir.glob(f"{stem}.part*.log"):
            name = p.name
            if not name.startswith(f"{stem}.part") or not name.endswith(".log"):
                continue
            raw = name[len(stem) + len(".part") : -len(".log")]
            try:
                n = int(raw)
            except ValueError:
                continue
            latest = max(latest, n)
        return latest or 1

    def _ensure_part_has_capacity(self, next_line: str) -> None:
        if self._stream is None:
            return
        path = self._path_for_part(self._part)
        try:
            current_size = path.stat().st_size
        except OSError:
            current_size = 0
        incoming_size = len((next_line + "\n").encode(self.encoding, errors="replace"))
        if current_size + incoming_size <= self._max_bytes:
            return
        self._stream.close()
        self._stream = None
        self._part += 1
        self._stream = open(self._path_for_part(self._part), "a", encoding=self.encoding)

    def _ensure_stream(self) -> None:
        week_key = self._week_key_now()
        if self._week_key == week_key and self._stream is not None:
            return
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        self._week_key = week_key
        self._part = self._detect_latest_part()
        self._stream = open(self._path_for_part(self._part), "a", encoding=self.encoding)

    def emit(self, record: LogRecord) -> None:
        try:
            self._ensure_stream()
            assert self._stream is not None
            line = self.format(record)
            self._ensure_part_has_capacity(line)
            assert self._stream is not None
            self._stream.write(line + "\n")
            self._stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        super().close()


def _log_to_console() -> bool:
    raw = os.environ.get("H3_GATEWAY_LOG_CONSOLE", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _log_level() -> int:
    level_name = os.environ.get("H3_GATEWAY_LOG_LEVEL", "INFO").strip().upper()
    return getattr(logging, level_name, logging.INFO)


def _access_log_enabled() -> bool:
    raw = os.environ.get("H3_GATEWAY_ACCESS_LOG", "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def setup_root_logging(prefix: str, *, logger: logging.Logger | None = None) -> Path:
    if getattr(setup_root_logging, "_prefixes", None) is None:
        setup_root_logging._prefixes = set()  # type: ignore[attr-defined]
    if prefix in setup_root_logging._prefixes:  # type: ignore[attr-defined]
        return weekly_log_path(prefix)
    setup_root_logging._prefixes.add(prefix)  # type: ignore[attr-defined]

    level = _log_level()
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=LOG_TS_FMT)
    log_path = weekly_log_path(prefix)

    root = logging.getLogger()
    root.setLevel(level)

    file_handler = WeeklyLogFileHandler(prefix=prefix)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    if _log_to_console():
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(level)
        console.setFormatter(formatter)
        root.addHandler(console)

    if logger is not None:
        logger.setLevel(level)
        logger.propagate = True

    return log_path


def uvicorn_log_config(prefix: str = "h3l_gateway") -> dict[str, Any]:
    log_dir = str(repo_log_dir())
    access_enabled = _access_log_enabled()
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(asctime)s %(levelprefix)s %(message)s",
                "datefmt": LOG_TS_FMT,
                "use_colors": False,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
                "datefmt": LOG_TS_FMT,
                "use_colors": False,
            },
        },
        "handlers": {
            "weekly_file": {
                "()": "gateway_logging.WeeklyLogFileHandler",
                "prefix": prefix,
                "log_dir": log_dir,
                "formatter": "default",
            },
            "weekly_access": {
                "()": "gateway_logging.WeeklyLogFileHandler",
                "prefix": prefix,
                "log_dir": log_dir,
                "formatter": "access",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["weekly_file"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"level": "INFO", "handlers": ["weekly_file"], "propagate": False},
            "uvicorn.access": {
                "handlers": ["weekly_access"] if access_enabled else [],
                "level": "INFO" if access_enabled else "WARNING",
                "propagate": False,
            },
        },
    }
