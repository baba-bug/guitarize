"""编配编排：leadsheet → 选变调夹 → 生成弹唱 / 指弹 / 旋律谱草稿 alphaTex → 自检。默认不覆盖已有编配（那是迭代修改过的成果）。
卡祖笛谱（kazoo）按实际音高写，没有变调夹、指法，自检只查记谱。
强力和弦谱（power）按实际音高写（不夹变调夹），节奏型按段落只看原曲吉他分轨选（power_patterns）。"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import numpy as np

from ..config import Profile, SongConfig, finger_bass, finger_style, load_profile, power_patterns, strum_pattern
from ..paths import SongPaths
from ..render import alphatab
from ..render.alphatab import AlphaTexError
from ..verify.metrics import onset_slots
from ..verify.model import load_score
from ..verify.playability import Violation, check_notation, check_playability
from .capo import CapoOption, choose_capo
from .finger import generate_finger, section_bars
from .kazoo import generate_kazoo
from .melody import generate_melody
from .patterns import DEFAULT, choose_pattern, slots_per_beat
from .power import choose_patterns, generate_power
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


def power_patterns_for(paths: SongPaths, leadsheet: dict, overrides: dict) -> list[str | None]:
    """强力和弦谱逐小节的节奏型（power.choose_patterns）：只取原曲吉他分轨的起音——鼓的起音强得多，和吉他混在一起
    选出来的是鼓的节奏型（なんもねえ 选成了 D-DU-UDU）。没有吉他分轨时用默认型；overrides.power_patterns 逐段指定。"""
    guitar = paths.stem("guitar")
    onset = None
    if guitar.exists():
        spb = slots_per_beat(leadsheet.get("feel", "straight"))
        onset = onset_slots([guitar], [(b["t_start"], b["t_end"]) for b in leadsheet["bars"]], slots=4 * spb)
    return choose_patterns(onset, leadsheet, power_patterns(overrides))


def self_check(text: str, profile: Profile, kind: str) -> list[Violation]:
    """草稿自检：用 alphaTab 构建后跑 verify 同一套可弹性检查。profile 用难度档本身（品位上限由检查自己按变调夹收紧）。
    卡祖笛谱（kind = kazoo）没有指法，只查记谱（check_notation）。"""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "draft.atex"
        src.write_text(text, encoding="utf-8")
        built = alphatab.build(src, Path(tmp), "draft")
        score = load_score(built["json"])
    return check_notation(score) if kind == "kazoo" else check_playability(score, profile, kind)


def _first_line(error: Exception) -> str:
    """错误里最有用的一行：alphaTex 的定位诊断（`L行:列 …`）优先；临时文件路径那行没用（文件已删）。"""
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    return next((line for line in lines if re.match(r"L\d+:\d+", line)), lines[0] if lines else type(error).__name__)


def _checked(text: str, profile: Profile, arr: str) -> list[Violation] | str:
    """self_check；alphaTab 构建失败或找不到 node 时是一行 `自检失败：<错误第一行>`（不让编配失败）。"""
    try:
        return self_check(text, profile, arr)
    except (AlphaTexError, FileNotFoundError) as e:
        return f"自检失败：{_first_line(e)}"


def arrange(paths: SongPaths, cfg: SongConfig, arr: str, capo: int | None = None,
            force: bool = False) -> tuple[Path, CapoOption | None, list[Violation] | str]:
    """返回 (草稿路径, 选中的变调夹, 自检结果)。自检结果是违规列表；alphaTab 构建失败或找不到 node 时
    是一行 `自检失败：<错误第一行>`。有违规、自检失败都照常写出草稿，留给审阅。
    卡祖笛谱（kazoo）、强力和弦谱（power）没有变调夹（选中的变调夹是 None，capo / overrides.capo 不起作用）。"""
    target = paths.atex(arr)
    if target.exists() and not force:
        raise FileExistsError(f"{target} exists; edit it directly or pass --force to regenerate")
    leadsheet = json.loads(paths.leadsheet_json.read_text(encoding="utf-8"))
    bpb = leadsheet.get("beats_per_bar", 4)
    if bpb != 4:
        raise ValueError(f"only 4/4 is supported for now (beats_per_bar={bpb})")
    profile = load_profile(cfg.level)
    if arr in ("kazoo", "power"):
        text = (generate_kazoo(leadsheet, cfg.title) if arr == "kazoo" else
                generate_power(leadsheet, profile, power_patterns_for(paths, leadsheet, cfg.overrides), cfg.title))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target, None, _checked(text, profile, arr)
    notes = [(m["start_beats"], m["midi"]) for b in leadsheet["bars"] for m in b["melody"]]
    melody = sorted(notes) if arr in ("finger", "melody") else []   # 旋律谱与指弹按同样的旋律音域选变调夹：指法调一致
    octave = cfg.overrides.get("melody_octave", "song")
    options = choose_capo(chord_beats(leadsheet), profile, melody, section_bars=section_bars(leadsheet), octave=octave)
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
            checked[text] = _checked(text, profile, arr)
        return checked[text]

    def check(text: str) -> list[Violation]:   # 给生成器的修补循环：自检失败当作无违规，修补就此停下
        found = result(text)
        return found if isinstance(found, list) else []

    if arr == "strum":
        text = generate_strum(leadsheet, chosen.capo, profile, ([fixed] * len(leadsheet["bars"]) if (fixed := strum_pattern(cfg.overrides)) else section_patterns(paths, leadsheet)), cfg.title,
                              check=check)
    elif arr == "melody":
        text = generate_melody(leadsheet, chosen.capo, profile, cfg.title, check=check, octave=octave)
    else:
        text = generate_finger(leadsheet, chosen.capo, profile, cfg.title, check=check, octave=octave,
                               style=finger_style(cfg.overrides), bass=finger_bass(cfg.overrides))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target, chosen, result(text)
