"""获取音源：B 站等公开音频流用 yt-dlp 下载；或使用用户自己放进来的本地文件。
统一用 ffmpeg 转成 44.1kHz 立体声 wav（可截取区间），记录来源到 source/meta.json。
不处理任何加密格式（如 .ncm）。下载前必须已获得用户确认（由调用方 / skill 负责）。"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from .paths import SongPaths


def download(url: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "-f", "bestaudio", "--no-playlist",
         "-o", str(raw_dir / "%(id)s.%(ext)s"), "--print", "after_move:filepath", "--", url],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp failed (exit {proc.returncode}): {(proc.stderr or '')[-2000:]}")
    return Path(proc.stdout.strip().splitlines()[-1])


def _kind(url: str | None) -> str:
    if not url:
        return "local"
    host = (urlparse(url).hostname or "").lower()
    if host in ("bilibili.com", "b23.tv") or host.endswith((".bilibili.com", ".b23.tv")):
        return "bilibili"
    return "url"


def normalize(raw: Path, wav: Path, trim: tuple[float, float] | None = None) -> Path:
    if trim is not None and not (0 <= trim[0] < trim[1]):
        raise ValueError(f"invalid trim {trim}: need 0 <= start < end")
    wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(raw)]
    if trim:
        cmd += ["-ss", str(trim[0]), "-to", str(trim[1])]
    cmd += ["-ac", "2", "-ar", "44100", str(wav)]
    subprocess.run(cmd, check=True)
    return wav


def acquire(paths: SongPaths, url: str | None = None, raw: Path | None = None,
            trim: tuple[float, float] | None = None) -> Path:
    if (url is None) == (raw is None):
        raise ValueError("give exactly one of url / raw")
    if trim is not None and not (0 <= trim[0] < trim[1]):
        raise ValueError(f"invalid trim {trim}: need 0 <= start < end")
    raw_path = download(url, paths.source_dir / "raw") if url else Path(raw)
    normalize(raw_path, paths.original_wav, trim)
    meta = {
        "kind": _kind(url),
        "url": url,
        "raw": str(raw_path),
        "trim": list(trim) if trim else None,
        "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }
    (paths.source_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return paths.original_wav
