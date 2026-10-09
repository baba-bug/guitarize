"""生成本地交互谱页面：alphaTab 渲染 + 合成播放 + 按小节跳转原曲。需从仓库根目录起 HTTP 服务打开。"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

from ..config import ARR_NAMES
from ..paths import REPO_ROOT, SongPaths
from .synth import ALPHATAB_DIST

TEMPLATE = REPO_ROOT / "web" / "viewer.html"


def _rel(target: Path, start: Path) -> str:
    return Path(os.path.relpath(target, start)).as_posix()


def build_viewer(paths: SongPaths, arr: str, title: str) -> Path:
    out = paths.out(arr, "html")
    out.parent.mkdir(parents=True, exist_ok=True)
    bar_times: list[list[float]] = []
    if paths.leadsheet_json.exists():
        leadsheet = json.loads(paths.leadsheet_json.read_text(encoding="utf-8"))
        bar_times = [[b["t_start"], b["t_end"]] for b in leadsheet["bars"]]
    replacements = {
        "{{TITLE}}": html.escape(f"{title} · {ARR_NAMES[arr]}"),
        "{{ALPHATAB_BASE}}": _rel(ALPHATAB_DIST, out.parent) + "/",
        "{{AUDIO_URL}}": _rel(paths.original_wav, out.parent),
        "{{BAR_TIMES}}": json.dumps(bar_times),
        "{{ATEX}}": paths.atex(arr).read_text(encoding="utf-8").replace("</script", "<\\/script"),
    }
    page = TEMPLATE.read_text(encoding="utf-8")
    for key, value in replacements.items():
        page = page.replace(key, value)
    out.write_text(page, encoding="utf-8")
    return out
