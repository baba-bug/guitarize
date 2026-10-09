"""和弦指型库：开放和弦、E/A 型横按、替代指型、斜杠低音调整，以及兜底的高音弦部分指型；按难度档过滤，代价升序。"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, replace

from .. import theory
from ..config import Profile
from ..verify.playability import fingers_needed

OPEN_STRINGS = (64, 59, 55, 50, 45, 40)  # 1 弦 → 6 弦

OPEN_SHAPES: dict[str, tuple] = {
    "C": (0, 1, 0, 2, 3, None), "A": (0, 2, 2, 2, 0, None), "G": (3, 0, 0, 0, 2, 3),
    "E": (0, 0, 1, 2, 2, 0), "D": (2, 3, 2, 0, None, None),
    "Am": (0, 1, 2, 2, 0, None), "Em": (0, 0, 0, 2, 2, 0), "Dm": (1, 3, 2, 0, None, None),
    "C7": (0, 1, 3, 2, 3, None), "A7": (0, 2, 0, 2, 0, None), "G7": (1, 0, 0, 0, 2, 3),
    "E7": (0, 0, 1, 0, 2, 0), "D7": (2, 1, 2, 0, None, None), "B7": (2, 0, 2, 1, 2, None),
    "Am7": (0, 1, 0, 2, 0, None), "Em7": (0, 0, 0, 0, 2, 0), "Dm7": (1, 1, 2, 0, None, None),
    "Cmaj7": (0, 0, 0, 2, 3, None), "Fmaj7": (0, 1, 2, 3, None, None),
    "Asus4": (0, 3, 2, 2, 0, None), "Dsus4": (3, 3, 2, 0, None, None), "Esus4": (0, 0, 2, 2, 2, 0),
}
# 横按形状：(根音所在空弦的 pc, {性质: 以空弦为根时的品位})
_BARRE = {
    "E": (4, {"maj": (0, 0, 1, 2, 2, 0), "min": (0, 0, 0, 2, 2, 0), "7": (0, 0, 1, 0, 2, 0), "min7": (0, 0, 0, 0, 2, 0)}),
    "A": (9, {"maj": (0, 2, 2, 2, 0, None), "min": (0, 1, 2, 2, 0, None), "7": (0, 2, 0, 2, 0, None), "min7": (0, 1, 0, 2, 0, None)}),
}
_SUFFIX = {"maj": "", "min": "m", "7": "7", "min7": "m7", "maj7": "maj7", "sus4": "sus4", "sus2": "sus2",
           "dim": "dim", "aug": "aug", "min7b5": "m7-5"}
# 找不到该性质的指型时可用的替代（性质, 额外代价）
_SUBSTITUTE = {"maj": [("7", 0.3)], "7": [("maj", 0.2)], "min7": [("min", 0.2)], "maj7": [("maj", 0.2)],
               "sus4": [("maj", 0.4)], "sus2": [("maj", 0.4)], "aug": [("maj", 0.6)], "dim": [("min", 0.6)],
               "min7b5": [("dim", 0.3), ("min", 0.6)]}
_POWER = re.compile(r"^([A-G][#b]?)5$")
# 部分指型（只弹 1–3 弦或 1–4 弦）：排在完整指型之后的兜底
PARTIAL_EXTRA = 1.5   # 比最好的完整指型贵这么多
PARTIAL_ALONE = 2.0   # 没有完整指型时的代价
PARTIAL_SLASH = 0.5   # 高音弦上做不到斜杠低音：沿用原名并加罚


@dataclass(frozen=True)
class Voicing:
    name: str      # 和弦图名称（指法调），如 "G"、"B7/D#"
    frets: tuple   # 1 弦 → 6 弦，None = 不弹
    cost: float
    barre: bool = False


def sounding(frets: tuple) -> list[int]:
    return [OPEN_STRINGS[i] + f for i, f in enumerate(frets) if f is not None]


def playable(frets: tuple, profile: Profile) -> bool:
    fretted = [f for f in frets if f is not None and f > 0]
    if not fretted:
        return True
    return (max(fretted) - min(fretted) <= profile.max_span
            and fingers_needed(fretted) <= profile.max_fingers
            and max(fretted) <= profile.max_fret)


def _shapes_for(root: int, quality: str, profile: Profile) -> list[Voicing]:
    name = theory.pc_name(root) + _SUFFIX[quality]
    out = []
    if name in OPEN_SHAPES:
        out.append(Voicing(name, OPEN_SHAPES[name], 0.0))
    if profile.allow_barre:
        for open_pc, table in _BARRE.values():
            k = (root - open_pc) % 12
            if quality in table and k:
                frets = tuple(None if f is None else f + k for f in table[quality])
                out.append(Voicing(name, frets, 1.0 + 0.05 * k, barre=True))
    return out


def with_bass(v: Voicing, bass_pc: int, profile: Profile) -> Voicing | None:
    """把最低音换成 bass_pc：依次尝试 6、5、4 弦（其下的弦不弹）；须仍为最低音且可弹。"""
    for idx in (5, 4, 3):
        fret = (bass_pc - OPEN_STRINGS[idx]) % 12
        frets = list(v.frets)
        frets[idx] = fret
        for j in range(idx + 1, 6):
            frets[j] = None
        frets = tuple(frets)
        if min(sounding(frets)) == OPEN_STRINGS[idx] + fret and playable(frets, profile):
            return Voicing(f"{v.name}/{theory.pc_name(bass_pc)}", frets, v.cost + 0.2, v.barre)
    return None


def partial_shapes(root: int, quality: str, profile: Profile) -> list[tuple[tuple, bool]]:
    """高音弦上的三和弦 / 七和弦（任意转位）：1–3 弦或 1–4 弦全弹，每根弦都是和弦音，且含全部必要音
    （四音和弦可省纯五度）。品位 0..max_fret，须可弹；手指不够、只能靠横按时，只有允许横按才收。
    返回 [(品位, 是否横按)]，把位低、弦多、按弦少的在前。"""
    tones = theory.chord_tones(theory.pc_name(root) + _SUFFIX[quality])
    needed = tones - {(root + 7) % 12} if len(tones) == 4 else tones
    out = []
    for n in (3, 4):
        options = [[f for f in range(profile.max_fret + 1) if (OPEN_STRINGS[s] + f) % 12 in tones] for s in range(n)]
        for combo in itertools.product(*options):
            if not needed <= {(OPEN_STRINGS[s] + f) % 12 for s, f in enumerate(combo)}:
                continue
            frets = combo + (None,) * (6 - n)
            barre = len([f for f in combo if f > 0]) > profile.max_fingers
            if playable(frets, profile) and (profile.allow_barre or not barre):
                out.append((frets, barre))

    def order(item):
        fretted = [f for f in item[0][:4] if f]
        return (max(fretted, default=0), max(fretted, default=0) - min(fretted, default=0),
                -sum(f is not None for f in item[0]), len(fretted), tuple(-1 if f is None else f for f in item[0]))
    return sorted(out, key=order)


def voicings(label: str, profile: Profile) -> list[Voicing]:
    """指法调里的和弦名 → 可弹指型（代价升序）。完整指型在前，高音弦部分指型兜底，所以能解析的和弦总有指型。"""
    m = _POWER.match(label)
    if m:
        label = m.group(1)
    parsed = theory.parse_chord(label)
    if parsed is None:
        return []
    root, quality, bass = parsed
    cands = _shapes_for(root, quality, profile)
    for alt, extra in _SUBSTITUTE.get(quality, []):
        cands += [replace(v, cost=v.cost + extra) for v in _shapes_for(root, alt, profile)]
    if bass is not None and bass != root:
        cands = [with_bass(v, bass, profile) or replace(v, cost=v.cost + 0.5) for v in cands]
    cands = [v for v in cands if playable(v.frets, profile)]
    cost = (min(v.cost for v in cands) + PARTIAL_EXTRA if cands else PARTIAL_ALONE)
    cost += PARTIAL_SLASH if bass is not None and bass != root else 0.0
    for q, extra in [(quality, 0.0)] + _SUBSTITUTE.get(quality, []):   # 本性质凑不出时才用替代性质
        partial = [Voicing(theory.pc_name(root) + _SUFFIX[q], frets, cost + extra, barre)
                   for frets, barre in partial_shapes(root, q, profile)]
        if partial:
            cands += partial
            break
    if not cands:
        raise ValueError(f"no playable voicing for {label!r} at level {profile.name}")
    return sorted(cands, key=lambda v: v.cost)
