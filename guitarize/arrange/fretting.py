"""指弹指法分配：每个起音时刻枚举可弹的（弦, 品）组合，Viterbi 选出整体最顺手的把位。

组合规则：音高从高到低依次放在编号严格递增的弦上（旋律在上、低音在下）；按弦跨度 ≤ max_span，品位 ≤ max_fret。
静态代价：把位越高越贵、跨度越大越贵、7 品以上加罚、空弦减分、旋律不在 1–3 弦加罚。
转移代价：手型框移动的品数 ×（1 + 1/时间间隔）；一拍以内最低品移动超过 max_shift（校验的换把硬规则）再罚 1000。
全空弦组合不占把位：Viterbi 状态带上"路径上最近一次按弦的手型框"，空弦事件原样继承，下一次按弦的转移代价
按它计算（时间间隔仍取相邻两个事件之差：校验会把空弦起音时仍按着的低音算进手型，按相邻间隔算更保守）。
音高从高到低必须是 旋律 > 填充 > 低音。某时刻无解时按顺序降级：低音降八度 → 低音升八度 → 去掉填充 →
（去掉填充后）低音降 / 升八度 → 去掉低音 → 旋律升八度 → 旋律降八度，并记录在 fallback。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from ..config import Profile
from .shapes import OPEN_STRINGS


@dataclass
class FretEvent:
    start: float               # 记谱位置（拍）
    melody: int | None         # 指法调音高
    bass: int | None
    fill: int | None = None    # 低音之上的和弦内音
    melody_pos: tuple[int, int] | None = None
    fill_pos: tuple[int, int] | None = None
    bass_pos: tuple[int, int] | None = None
    fallback: str | None = None


def _candidates(pitches: list[int], profile: Profile) -> list[tuple]:
    per = [[(s + 1, p - OPEN_STRINGS[s]) for s in range(6) if 0 <= p - OPEN_STRINGS[s] <= profile.max_fret]
           for p in pitches]
    out = []
    for combo in itertools.product(*per):
        strings = [c[0] for c in combo]
        if any(b <= a for a, b in zip(strings, strings[1:])):
            continue
        fretted = [f for _, f in combo if f > 0]
        if fretted and max(fretted) - min(fretted) > profile.max_span:
            continue
        out.append(combo)
    return out


def _static(combo: tuple, has_melody: bool) -> float:
    fretted = [f for _, f in combo if f > 0]
    cost = 0.0
    if fretted:
        cost += 0.15 * min(fretted) + 0.3 * (max(fretted) - min(fretted)) + 0.5 * max(0, max(fretted) - 7)
    cost -= 0.3 * sum(1 for _, f in combo if f == 0)
    if has_melody and combo:
        cost += 0.5 * max(0, combo[0][0] - 3)
    return cost


def _position(combo: tuple, span: int) -> tuple[int, int] | None:
    """手型框（四品格）起点的可行区间 (lo, hi)；全空弦则无约束。"""
    fretted = [f for _, f in combo if f > 0]
    if not fretted:
        return None
    return max(1, max(fretted) - span), min(fretted)


HARD = 1000.0  # 违反校验硬规则的转移罚分（不设为无穷：总要有一条路径）


def _transition(pa: tuple[int, int] | None, pb: tuple[int, int] | None, gap: float, profile: Profile) -> float:
    """手型框 pa → pb 的代价；任一为 None（尚无按弦）不计。框的 hi 就是最低按弦品，与校验的把位定义一致。"""
    if pa is None or pb is None:
        return 0.0
    dist = max(0, pa[0] - pb[1], pb[0] - pa[1])
    cost = dist * (1.0 + 1.0 / max(gap, 0.25))
    if gap < 1.0 - 1e-6 and abs(pa[1] - pb[1]) > profile.max_shift:
        cost += HARD
    return cost


def _ordered(m: int | None, f: int | None, b: int | None) -> tuple[int | None, int | None, int | None]:
    """保证 旋律 > 填充 > 低音：低音不低于旋律时降八度；填充不在两者之间时丢弃。"""
    if m is not None and b is not None and b >= m:
        b -= 12 * ((b - m) // 12 + 1)
    if f is not None:
        if m is not None:
            while f >= m:
                f -= 12
        if b is not None and f <= b:
            f = None
    return m, f, b


def _variants(m: int | None, f: int | None, b: int | None) -> list[tuple]:
    """(旋律, 填充, 低音, 降级说明)，按尝试顺序排列。"""
    m, f, b = _ordered(m, f, b)
    out = [(m, f, b, None)]
    if b is not None:
        out.append(_ordered(m, f, b - 12) + ("bass-12",))
        if (m is None or b + 12 < m) and (f is None or b + 12 < f):
            out.append((m, f, b + 12, "bass+12"))
    if f is not None:
        out.append((m, None, b, "drop fill"))
        if b is not None:
            out.append((m, None, b - 12, "drop fill, bass-12"))
            if m is None or b + 12 < m:
                out.append((m, None, b + 12, "drop fill, bass+12"))
    if b is not None and m is not None:
        out.append((m, None, None, "drop bass"))
    if m is not None:
        out += [(m + 12, None, None, "melody+12"), (m - 12, None, None, "melody-12")]
    return out


def assign_frets(events: list[FretEvent], profile: Profile) -> list[FretEvent]:
    if not events:
        return events
    layers = []  # 每个事件：(候选组合列表, (旋律, 填充, 低音), 降级说明)
    for ev in events:
        for m, f, b, label in _variants(ev.melody, ev.fill, ev.bass):
            pitches = [p for p in (m, f, b) if p is not None]
            cands = _candidates(pitches, profile) if pitches else [()]
            if cands:
                if ev.fill is not None and f is None and not (label or "").startswith("drop fill"):
                    label = f"{label}, fill dropped" if label else "fill dropped"
                layers.append((cands, (m, f, b), label))
                break
        else:
            layers.append(([()], (None, None, None), "dropped"))
    span = profile.max_span
    # 状态 = (候选下标, 手型框)；全空弦候选继承前驱状态的手型框，每种来路各成一个状态
    states = [[(k, _position(c, span)) for k, c in enumerate(layers[0][0])]]
    cost = [_static(c, layers[0][1][0] is not None) for c in layers[0][0]]
    back = []
    for i in range(1, len(layers)):
        gap = events[i].start - events[i - 1].start
        cands, used, _ = layers[i]
        best: dict[tuple, tuple[float, int]] = {}  # 状态 → (累计代价, 前驱状态下标)
        for k, c in enumerate(cands):
            own, static = _position(c, span), _static(c, used[0] is not None)
            for j, (_, hand) in enumerate(states[-1]):
                key = (k, hand if own is None else own)
                total = cost[j] + _transition(hand, own, gap, profile) + static
                if key not in best or total < best[key][0]:
                    best[key] = (total, j)
        states.append(list(best))
        cost = [best[st][0] for st in states[-1]]
        back.append([best[st][1] for st in states[-1]])
    s = min(range(len(cost)), key=cost.__getitem__)
    path = [s]
    for bp in reversed(back):
        path.append(bp[path[-1]])
    path.reverse()
    for ev, (cands, used, label), layer, s in zip(events, layers, states, path):
        k = layer[s][0]
        positions = iter(cands[k])
        ev.melody_pos, ev.fill_pos, ev.bass_pos = (next(positions) if p is not None else None for p in used)
        ev.fallback = label
    return events
