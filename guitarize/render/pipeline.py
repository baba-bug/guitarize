"""渲染编排：alphaTex → build/ 中间产物 → out/ 交付物（gp / txt / pdf / html）。"""
from __future__ import annotations

import shutil
from pathlib import Path

from ..config import SongConfig
from ..paths import SongPaths
from ..verify.model import load_score
from . import alphatab
from .ascii_tab import render_ascii
from .pdf import svg_to_pdf
from .viewer import build_viewer


def render_all(paths: SongPaths, cfg: SongConfig, arr: str) -> dict[str, Path]:
    built = alphatab.build(paths.atex(arr), paths.build_dir, arr)
    score = load_score(built["json"])
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    out = {"gp": paths.out(arr, "gp")}
    shutil.copyfile(built["gp"], out["gp"])
    if "txt" in cfg.outputs:
        out["txt"] = paths.out(arr, "txt")
        out["txt"].write_text(render_ascii(score), encoding="utf-8")
    if "pdf" in cfg.outputs:
        out["pdf"] = svg_to_pdf(built["svg"], paths.out(arr, "pdf"), score.title)
    if "html" in cfg.outputs:
        out["html"] = build_viewer(paths, arr, score.title)
    return out
