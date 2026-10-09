"""编配编排：leadsheet → 选变调夹 → 生成弹唱 / 指弹草稿 alphaTex → 自检。默认不覆盖已有编配（那是迭代修改过的成果）。"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import numpy as np

from ..config import Profile, SongConfig, load_profile
from ..paths import SongPaths
from ..render import alphatab
from ..render.alphatab import AlphaTexError
from ..verify.metrics import onset_slots
from ..verify.model import load_score
from ..verify.playability import Violation, check_playability
from .capo import CapoOption, choose_capo
from .finger import generate_finger, section_bars
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


def self_check(text: str, profile: Profile, kind: str) -> list[Violation]:
    """草稿自检：用 alphaTab 构建后跑 verify 同一套可弹性检查。profile 用难度档本身（品位上限由检查自己按变调夹收紧）。"""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "draft.atex"
        src.write_text(text, encoding="utf-8")
        built = alphatab.build(src, Path(tmp), "draft")
        score = load_score(built["json"])
    return check_playability(score, profile, kind)


def _first_line(error: Exception) -> str:
    """错误里最有用的一行：alphaTex 的定位诊断（`L行:列 …`）优先；临时文件路径那行没用（文件已删）。"""
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    return next((line for line in lines if re.match(r"L\d+:\d+", line)), lines[0] if lines else type(error).__name__)


def arrange(paths: SongPaths, cfg: SongConfig, arr: str, capo: int | None = None,
            force: bool = False) -> tuple[Path, CapoOption, list[Violation] | str]:
    """返回 (草稿路径, 选中的变调夹, 自检结果)。自检结果是违规列表；alphaTab 构建失败或找不到 node 时
    是一行 `自检失败：<错误第一行>`。有违规、自检失败都照常写出草稿，留给审阅。"""
    target = paths.atex(arr)
    if target.exists() and not force:
        raise FileExistsError(f"{target} exists; edit it directly or pass --force to regenerate")
    leadsheet = json.loads(paths.leadsheet_json.read_text(encoding="utf-8"))
    bpb = leadsheet.get("beats_per_bar", 4)
    if bpb != 4:
        raise ValueError(f"only 4/4 is supported for now (beats_per_bar={bpb})")
    profile = load_profile(cfg.level)
    notes = [(m["start_beats"], m["midi"]) for b in leadsheet["bars"] for m in b["melody"]]
    melody = sorted(notes) if arr == "finger" else []
    options = choose_capo(chord_beats(leadsheet), profile, melody, section_bars=section_bars(leadsheet))
    wanted = capo if capo is not None else cfg.overrides.get("capo")
    if wanted is None:
        chosen = options[0]
    else:
        valid = sorted(o.capo for o in options)
        chosen = next((o for o in options if o.capo == int(wanted)), None)
        if chosen is None:
            raise ValueError(f"capo must be one of {valid[0]}..{valid[-1]} (got {wanted})")
    checked: dict[str, list[Violation] | str] = {}

    def result(text: str) -> list[Violation] | str:   # 同一份文本只构建一次；构建失败不让编配失败
        if text not in checked:
            try:
                checked[text] = self_check(text, profile, arr)
            except (AlphaTexError, FileNotFoundError) as e:
                checked[text] = f"自检失败：{_first_line(e)}"
        return checked[text]

    def check(text: str) -> list[Violation]:   # 给生成器的修补循环：自检失败当作无违规，修补就此停下
        found = result(text)
        return found if isinstance(found, list) else []

    if arr == "strum":
        text = generate_strum(leadsheet, chosen.capo, profile, section_patterns(paths, leadsheet), cfg.title,
                              check=check)
    else:
        text = generate_finger(leadsheet, chosen.capo, profile, cfg.title, check=check)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target, chosen, result(text)
