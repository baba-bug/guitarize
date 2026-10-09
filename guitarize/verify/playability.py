"""可弹性检查：硬规则（违反即不合格）+ 软统计（只报告）。"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Profile
from .model import Attack, Score


@dataclass(frozen=True)
class Violation:
    bar: int      # 1-based
    start: float  # 拍
    rule: str
    detail: str


def fingers_needed(frets: list[int]) -> int:
    """按弦手指数：最低品上 ≥ 2 根弦视为横按，只算 1 指。"""
    fretted = [f for f in frets if f > 0]
    if not fretted:
        return 0
    low = min(fretted)
    return len(fretted) - (fretted.count(low) - 1)


def _fretted(a: Attack) -> list[int]:
    return [n.fret for n in a.notes if n.fret > 0 and not n.dead]


def check_playability(score: Score, profile: Profile, kind: str) -> list[Violation]:
    out: list[Violation] = []
    for (bar, voice), length in sorted(score.voice_lengths().items()):
        expected = score.bars[bar].length
        if abs(length - expected) > 1e-6:
            out.append(Violation(bar + 1, score.bars[bar].start, "bar_length",
                                 f"声部 {voice + 1} 共 {length:g} 拍，应为 {expected:g} 拍"))
    last: tuple[float, int] | None = None  # (时间, 把位)
    for a in score.attacks():
        bar = a.bar + 1
        strings = [n.string for n in a.notes]
        if len(strings) != len(set(strings)):
            out.append(Violation(bar, a.start, "same_string", f"同一弦同时两个音：{sorted(strings)}"))
        fr = _fretted(a)
        if fr:
            if max(fr) - min(fr) > profile.max_span:
                out.append(Violation(bar, a.start, "span", f"跨度 {min(fr)}–{max(fr)} 品，超过 {profile.max_span}"))
            if fingers_needed(fr) > profile.max_fingers:
                out.append(Violation(bar, a.start, "fingers", f"需要 {fingers_needed(fr)} 根手指"))
            if max(fr) > profile.max_fret:
                out.append(Violation(bar, a.start, "max_fret", f"最高 {max(fr)} 品，超过 {profile.max_fret}"))
            pos = min(fr)
            if last is not None and a.start - last[0] < 1.0 and abs(pos - last[1]) > profile.max_shift:
                out.append(Violation(bar, a.start, "shift",
                                     f"{a.start - last[0]:.2f} 拍内换把 {last[1]}→{pos} 品"))
            last = (a.start, pos)
        if kind == "finger" and not a.strum and len(a.notes) > profile.max_plucks:
            out.append(Violation(bar, a.start, "plucks", f"同时拨 {len(a.notes)} 个音"))
    return out


def _is_barre(a: Attack) -> bool:
    fr = _fretted(a)
    return bool(fr) and fr.count(min(fr)) >= 3


def soft_stats(score: Score) -> dict[int, dict]:
    attacks = score.attacks()
    stats: dict[int, dict] = {}
    for bar in score.bars:
        mine = [a for a in attacks if a.bar == bar.index]
        spans = [max(f) - min(f) for f in (_fretted(a) for a in mine) if f]
        stats[bar.index + 1] = {
            "attacks_per_beat": round(len(mine) / bar.length, 2),
            "max_span": max(spans, default=0),
            "barre": any(_is_barre(a) for a in mine),
            "chord_marks": [e.chord for e in score.events if e.bar == bar.index and e.chord],
        }
    return stats
