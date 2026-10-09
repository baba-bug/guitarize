"""编配编排：leadsheet → 选变调夹 → 生成弹唱 / 指弹草稿 alphaTex。默认不覆盖已有编配（那是迭代修改过的成果）。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import SongConfig, load_profile
from ..paths import SongPaths
from ..verify.metrics import onset_slots
from .capo import CapoOption, choose_capo
from .finger import generate_finger
from .patterns import DEFAULT, choose_pattern, slots_per_beat
from .strum import beat_chords, generate_strum


def chord_beats(leadsheet: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for b in leadsheet["bars"]:
        for c in beat_chords(b, leadsheet.get("beats_per_bar", 4)):
            if c != "N":
                out[c] = out.get(c, 0.0) + 1.0
    return out


def section_patterns(paths: SongPaths, leadsheet: dict) -> list[str]:
    """按段落（section 标记延续到下一个标记）取原曲鼓 + 吉他分轨的平均起音向量，选节奏型。"""
    bars = leadsheet["bars"]
    stems = [paths.stem("drums"), paths.stem("guitar")]
    if not all(p.exists() for p in stems):
        return [DEFAULT] * len(bars)
    spb = slots_per_beat(leadsheet.get("feel", "straight"))
    onset = onset_slots(stems, [(b["t_start"], b["t_end"]) for b in bars], slots=4 * spb)
    groups: list[list[int]] = []
    for i, b in enumerate(bars):
        if i == 0 or b.get("section"):
            groups.append([])
        groups[-1].append(i)
    out = [DEFAULT] * len(bars)
    for g in groups:
        name = choose_pattern(np.mean(onset[g], axis=0), leadsheet.get("feel", "straight"))
        for i in g:
            out[i] = name
    return out


def arrange(paths: SongPaths, cfg: SongConfig, arr: str, capo: int | None = None,
            force: bool = False) -> tuple[Path, CapoOption]:
    target = paths.atex(arr)
    if target.exists() and not force:
        raise FileExistsError(f"{target} exists; edit it directly or pass --force to regenerate")
    leadsheet = json.loads(paths.leadsheet_json.read_text(encoding="utf-8"))
    profile = load_profile(cfg.level)
    melody = [m["midi"] for b in leadsheet["bars"] for m in b["melody"]] if arr == "finger" else []
    options = choose_capo(chord_beats(leadsheet), profile, melody)
    wanted = capo if capo is not None else cfg.overrides.get("capo")
    chosen = next(o for o in options if o.capo == int(wanted)) if wanted is not None else options[0]
    if arr == "strum":
        text = generate_strum(leadsheet, chosen.capo, profile, section_patterns(paths, leadsheet), cfg.title)
    else:
        text = generate_finger(leadsheet, chosen.capo, profile, cfg.title)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target, chosen
