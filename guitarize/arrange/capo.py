"""变调夹 / 指法调选择：对 0–7 品逐一打分（和弦指型代价 × 拍数 + 变调夹罚分 + 指弹旋律音域罚分）。
指弹旋律按段落（无段落标记时按乐句）分组，各组单独选八度移位。"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .. import theory
from ..config import Profile
from ..verify.playability import NECK_JOIN_FRET
from .shapes import Voicing, voicings

MELODY_LOW, MELODY_HIGH, MELODY_CENTER = 55, 74, 62   # G3–D5，中心 D4（指法调音高）
PHRASE_GAP = 3.0      # 无段落标记时，相邻旋律音间隔 ≥ 3 拍处断句
MIN_PHRASE = 8        # 少于 8 个音的乐句并入前一组
KEEP_OUT = 0.1        # 滞回：上一组的移位让本组越界比例 ≤ 0.1 时沿用


def effective_profile(profile: Profile, capo: int) -> Profile:
    """变调夹之上可用的品位上限不超过琴颈与琴身交界（14 品）。"""
    return replace(profile, max_fret=min(profile.max_fret, NECK_JOIN_FRET - capo))


@dataclass(frozen=True)
class CapoOption:
    capo: int
    cost: float
    shapes: dict          # 实际和弦名 → Voicing（指法调）


def _out_of_range(pitches: np.ndarray) -> float:
    return float(np.mean((pitches < MELODY_LOW) | (pitches > MELODY_HIGH)))


def melody_octave(pitches: Sequence[int]) -> tuple[int, float]:
    """在 −24 / −12 / 0 / +12 中选整体移位：先让越界音最少，再让中位数最接近 D4。"""
    if not len(pitches):
        return 0, 0.0
    best = None
    for shift in (-24, -12, 0, 12):
        p = np.asarray(pitches) + shift
        out = _out_of_range(p)
        key = (out, abs(float(np.median(p)) - MELODY_CENTER))
        if best is None or key < best[0]:
            best = (key, shift, out)
    return best[1], best[2]


def melody_groups(positions: Sequence[float], section_bars: Sequence[int]) -> list[list[int]]:
    """旋律音分组（positions 已排序，单位拍，4/4）→ 每组的下标。
    有段落标记（section_bars：段落起始小节，0-based）时按段落分，标记延续到下一个标记，第一个标记之前的音单独成组；
    否则在相邻音间隔 ≥ 3 拍处断句，少于 8 个音的乐句并入前一组（开头的短乐句并入后一组）。"""
    if section_bars:
        starts = sorted(section_bars)
        by_section: dict[int, list[int]] = {}
        for i, pos in enumerate(positions):
            by_section.setdefault(bisect_right(starts, int(pos // 4)), []).append(i)
        return [by_section[k] for k in sorted(by_section)]
    phrases: list[list[int]] = []
    for i, pos in enumerate(positions):
        if not phrases or pos - positions[i - 1] >= PHRASE_GAP:
            phrases.append([])
        phrases[-1].append(i)
    groups: list[list[int]] = []
    for ph in phrases:
        if groups and (len(ph) < MIN_PHRASE or len(groups[-1]) < MIN_PHRASE):
            groups[-1] += ph
        else:
            groups.append(ph)
    return groups


def octave_plan(notes: Sequence[tuple[float, int]],
                section_bars: Sequence[int]) -> list[tuple[list[int], int, float]]:
    """notes：(位置拍, 指法调音高)，按位置排序。每组 → (下标, 八度移位, 越界比例)。
    上一组的移位对本组越界比例 ≤ 0.1 时沿用（避免段落之间无谓地跳八度），否则用 melody_octave 重选。"""
    pitches = np.asarray([p for _, p in notes])
    plan: list[tuple[list[int], int, float]] = []
    for idx in melody_groups([pos for pos, _ in notes], section_bars):
        group = pitches[idx]
        kept = _out_of_range(group + plan[-1][1]) if plan else None
        if kept is not None and kept <= KEEP_OUT:
            plan.append((idx, plan[-1][1], kept))
        else:
            plan.append((idx, *melody_octave(group)))
    return plan


def choose_capo(chord_beats: dict[str, float], profile: Profile, melody: Sequence[tuple[float, int]] = (),
                max_capo: int = 7, section_bars: Sequence[int] = ()) -> list[CapoOption]:
    """melody：指弹旋律 (位置拍, 实际音高)；音域罚分按 octave_plan 分组移位后的越界比例计。"""
    total = sum(chord_beats.values()) or 1.0
    options = []
    for capo in range(max_capo + 1):
        shapes: dict[str, Voicing] = {}
        cost = 0.0
        for label, beats in chord_beats.items():
            try:
                opts = voicings(theory.transpose_label(label, -capo), effective_profile(profile, capo))
            except ValueError:
                cost += beats * 5.0
                continue
            if not opts:  # 无和弦（N）：不计代价、不出指型
                continue
            v = opts[0]
            shapes[label] = v
            cost += beats * v.cost
        cost += 0.02 * capo * total
        if len(melody):
            plan = octave_plan([(pos, p - capo) for pos, p in melody], section_bars)
            out = sum(frac * len(idx) for idx, _, frac in plan) / len(melody)
            cost += 2.0 * out * total
        options.append(CapoOption(capo, round(cost, 3), shapes))
    return sorted(options, key=lambda o: o.cost)
