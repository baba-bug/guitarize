"""变调夹 / 指法调选择：对 0–7 品逐一打分（和弦指型代价 × 拍数 + 变调夹罚分 + 指弹旋律音域罚分）。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .. import theory
from ..config import Profile
from ..verify.playability import NECK_JOIN_FRET
from .shapes import Voicing, voicings

MELODY_LOW, MELODY_HIGH, MELODY_CENTER = 55, 74, 62   # G3–D5，中心 D4（指法调音高）


def effective_profile(profile: Profile, capo: int) -> Profile:
    """变调夹之上可用的品位上限不超过琴颈与琴身交界（14 品）。"""
    return replace(profile, max_fret=min(profile.max_fret, NECK_JOIN_FRET - capo))


@dataclass(frozen=True)
class CapoOption:
    capo: int
    cost: float
    shapes: dict          # 实际和弦名 → Voicing（指法调）
    melody_shift: int     # 指弹旋律整体移的半音数（八度倍数）


def melody_octave(pitches: Sequence[int]) -> tuple[int, float]:
    """在 −24 / −12 / 0 / +12 中选整体移位：先让越界音最少，再让中位数最接近 D4。"""
    if not len(pitches):
        return 0, 0.0
    best = None
    for shift in (-24, -12, 0, 12):
        p = np.asarray(pitches) + shift
        out = float(np.mean((p < MELODY_LOW) | (p > MELODY_HIGH)))
        key = (out, abs(float(np.median(p)) - MELODY_CENTER))
        if best is None or key < best[0]:
            best = (key, shift, out)
    return best[1], best[2]


def choose_capo(chord_beats: dict[str, float], profile: Profile, melody: Sequence[int] = (),
                max_capo: int = 7) -> list[CapoOption]:
    total = sum(chord_beats.values()) or 1.0
    options = []
    for capo in range(max_capo + 1):
        shapes: dict[str, Voicing] = {}
        cost = 0.0
        for label, beats in chord_beats.items():
            try:
                v = voicings(theory.transpose_label(label, -capo), effective_profile(profile, capo))[0]
            except ValueError:
                cost += beats * 5.0
                continue
            shapes[label] = v
            cost += beats * v.cost
        cost += 0.02 * capo * total
        shift = 0
        if len(melody):
            shift, out = melody_octave([m - capo for m in melody])
            cost += 2.0 * out * total
        options.append(CapoOption(capo, round(cost, 3), shapes, shift))
    return sorted(options, key=lambda o: o.cost)
