"""可弹性检查：硬规则（违反即不合格）+ 软统计（只报告）。"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from ..config import Profile
from .model import Attack, Event, Note, Score

NECK_JOIN_FRET = 14  # 琴颈与琴身相接处的绝对品位（多数民谣吉他）
EPS = 1e-6  # 浮点容差（拍）：摇摆节奏下相邻一拍的间隔可能算成 0.9999999999999999


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


def _sounding_index(score: Score) -> tuple[list[Event], list[float], float]:
    evs = sorted((e for e in score.events if not e.rest), key=lambda e: e.start)
    return evs, [e.start for e in evs], max((e.dur for e in evs), default=0.0)


def _held(index: tuple[list[Event], list[float], float], t: float) -> list[Note]:
    """t 时刻仍在发声的有品非闷音（含连音线延续音），用二分缩小扫描范围。"""
    evs, starts, max_dur = index
    lo = bisect_left(starts, t - max_dur - EPS)
    hi = bisect_right(starts, t + EPS)
    out: list[Note] = []
    for e in evs[lo:hi]:
        if t < e.start + e.dur - EPS:
            out.extend(n for n in e.notes if n.fret > 0 and not n.dead)
    return out


def _hand_frets(a: Attack, index: tuple[list[Event], list[float], float]) -> list[int]:
    """按住的手形：本拍新拨的有品音 + 仍在发声的有品音；被新拨的同弦旧音让位。"""
    fresh = [(n.string, n.fret) for n in a.notes if n.fret > 0 and not n.dead]
    fresh_strings = {s for s, _ in fresh}
    held = [(n.string, n.fret) for n in _held(index, a.start) if n.string not in fresh_strings]
    return [f for _, f in sorted(set(fresh + held))]


def _straight_offbeats(score: Score) -> list[Violation]:
    """三连音感觉小节里，写在反拍八分位上却会被 alphaTab 按直拍播放的起音。"""
    out: list[Violation] = []
    for e in score.events:
        bar = score.bars[e.bar]
        if bar.feel != "triplet8th" or e.rest or e.tuplet != 1:
            continue
        if not any(not n.tie for n in e.notes):
            continue  # 只有连音线延续音，听不到起音
        rel = e.written_start - bar.start
        if abs(rel - int(rel) - 0.5) < EPS and abs(e.start - e.written_start) < EPS:
            out.append(Violation(e.bar + 1, e.written_start, "straight_offbeat",
                                 "反拍音会按直拍播放：把它和前一个八分写成一对八分音符（剩余时值用连音线接上）"))
    return out


def check_playability(score: Score, profile: Profile, kind: str) -> list[Violation]:
    out: list[Violation] = []
    for (bar, voice), length in sorted(score.voice_lengths().items()):
        expected = score.bars[bar].length
        if abs(length - expected) > 1e-6:
            out.append(Violation(bar + 1, score.bars[bar].start, "bar_length",
                                 f"声部 {voice + 1} 共 {length:g} 拍，应为 {expected:g} 拍"))
    out += _straight_offbeats(score)
    index = _sounding_index(score)
    neck_limit = NECK_JOIN_FRET - score.capo
    max_fret = min(profile.max_fret, neck_limit)
    last: tuple[float, int] | None = None  # (时间, 把位)
    for a in score.attacks():
        bar = a.bar + 1
        strings = [n.string for n in a.notes]
        if len(strings) != len(set(strings)):
            out.append(Violation(bar, a.start, "same_string", f"同一弦同时两个音：{sorted(strings)}"))
        fr = _fretted(a)
        hand = _hand_frets(a, index)
        if hand:
            if max(hand) - min(hand) > profile.max_span:
                out.append(Violation(bar, a.start, "span", f"跨度 {min(hand)}–{max(hand)} 品，超过 {profile.max_span}"))
            if fingers_needed(hand) > profile.max_fingers:
                out.append(Violation(bar, a.start, "fingers", f"需要 {fingers_needed(hand)} 根手指"))
        if fr and max(fr) > max_fret:
            note = f"（变调夹 {score.capo} 品，琴颈在绝对 {NECK_JOIN_FRET} 品处并入琴身）" if neck_limit < profile.max_fret else ""
            out.append(Violation(bar, a.start, "max_fret", f"最高 {max(fr)} 品，超过 {max_fret}{note}"))
        if hand:
            pos = min(hand)
            if last is not None and a.start - last[0] < 1.0 - EPS and abs(pos - last[1]) > profile.max_shift:
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
