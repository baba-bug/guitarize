"""按原曲吉他分轨逐乐句选节奏型（patterns.POWER_PATTERNS）：强力和弦弹唱（power）与弹唱的 guitar 节奏（strum）共用。

- 乐句：段落（标记延续到下一个标记），再在整小节无和弦处断开（phrases）。
- 只看吉他分轨的起音（鼓的起音强得多，混在一起选出来的是鼓的节奏型）：每个乐句取平均起音向量选最像的型；
  乐句内前后两半差得多时切一刀（なんもねえ 的 Bメロ）；起音没有可辨认节奏的乐句是一击（choose_patterns）。
- overrides（如 config.power_patterns）：{小节(1 起): 型}，从该小节起用到乐句末尾或下一个覆盖起点。"""
from __future__ import annotations

import numpy as np

from .. import theory
from .patterns import POWER_DEFAULT, POWER_FLAT, POWER_PATTERNS, choose_pattern, correlations
from .strum import beat_chords

MIN_CORR = 0.5      # 最像的节奏型与段落起音的相关度也不到它：没有可辨认的扫弦节奏（持续的音墙 / 余音）→ 一击
MIN_PART = 3        # 段落内切一刀时每段至少几小节
SPLIT_GAIN = 0.05   # 切开后（按小节数加权的）相关度至少比不切高这么多才切


def phrases(leadsheet: dict) -> list[list[int]]:
    """选节奏型的乐句（小节下标，0 起）：段落（标记延续到下一个标记），再在整小节无和弦处断开——无和弦小节只有休止，
    不在任何一组里（なんもねえ 間奏的 riff 前的停顿把一击和 riff 分开）。"""
    groups: list[list[int]] = []
    current: list[int] = []
    for i, b in enumerate(leadsheet["bars"]):
        if b.get("section") and current:
            groups.append(current)
            current = []
        if all(theory.parse_chord(c) is None for c in beat_chords(b, leadsheet.get("beats_per_bar", 4))):
            if current:
                groups.append(current)
            current = []
            continue
        current.append(i)
    return groups + ([current] if current else [])


def _best(rows: np.ndarray, feel: str) -> tuple[str, float]:
    """一组小节的平均起音向量 → (节奏型, 最像的型的相关度)；相关度不到 MIN_CORR 时是一击（POWER_FLAT）。"""
    mean = rows.mean(axis=0)
    cors = correlations(mean, feel, POWER_PATTERNS)
    if not cors:
        return POWER_FLAT, 0.0
    name = choose_pattern(mean, feel, POWER_PATTERNS, POWER_DEFAULT)
    return (name if cors[name] >= MIN_CORR else POWER_FLAT), cors[name]


def _split(group: list[int], onset: np.ndarray, feel: str) -> list[tuple[list[int], str]]:
    """一个乐句选型：前后两半（各至少 MIN_PART 小节）选出的型不同、且按小节数加权的相关度比不切至少高 SPLIT_GAIN 时
    切成两段（取提高最多的切点，只切一刀）。"""
    whole, r = _best(onset[group], feel)
    best = None
    for k in range(MIN_PART, len(group) - MIN_PART + 1):
        (head, ra), (tail, rb) = _best(onset[group[:k]], feel), _best(onset[group[k:]], feel)
        gain = (k * ra + (len(group) - k) * rb) / len(group) - r
        if head != tail and gain >= SPLIT_GAIN - 1e-9 and (best is None or gain > best[0]):
            best = (gain, k, head, tail)
    if best is None:
        return [(group, whole)]
    _, k, head, tail = best
    return [(group[:k], head), (group[k:], tail)]


def choose_patterns(onset: np.ndarray | None, leadsheet: dict,
                    overrides: dict[int, str] | None = None) -> list[str | None]:
    """逐小节的节奏型（整小节无和弦是 None）。onset：原曲吉他分轨逐小节的起音向量（metrics.onset_slots，每拍
    slots_per_beat 格），None = 没有分轨，全用 POWER_DEFAULT。overrides：{小节(1 起): 型}，
    从该小节起用到它所在乐句的末尾或下一个覆盖起点。"""
    feel = leadsheet.get("feel", "straight")
    out: list[str | None] = [None] * len(leadsheet["bars"])
    groups = phrases(leadsheet)
    for group in groups:
        for idx, name in ([(group, POWER_DEFAULT)] if onset is None else _split(group, onset, feel)):
            for i in idx:
                out[i] = name
    starts = {bar - 1: name for bar, name in (overrides or {}).items()}
    for start, name in sorted(starts.items()):
        group = next((g for g in groups if start in g), None)
        if group is None:
            raise ValueError(f"overrides.power_patterns: bar {start + 1} has no chord (or is out of range)")
        for i in group[group.index(start):]:
            if i != start and i in starts:
                break
            out[i] = name
    return out
