"""和弦指型库：开放和弦、E/A 型横按、替代指型、斜杠低音调整；按难度档过滤，代价升序。"""
from __future__ import annotations

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
               "min7b5": [("min", 0.6)]}
_POWER = re.compile(r"^([A-G][#b]?)5$")


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


def voicings(label: str, profile: Profile) -> list[Voicing]:
    """指法调里的和弦名 → 可弹指型（代价升序）。"""
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
    if not cands:
        raise ValueError(f"no playable voicing for {label!r} at level {profile.name}")
    return sorted(cands, key=lambda v: v.cost)
