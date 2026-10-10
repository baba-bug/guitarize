"""指弹指法分配：每个起音时刻枚举可弹的（弦, 品）组合，Viterbi 选出整体最顺手的把位。

组合规则：音高从高到低依次放在编号严格递增的弦上（旋律在上、低音在下）；按弦跨度 ≤ max_span，品位 ≤ max_fret。
静态代价：把位越高越贵、跨度越大越贵、7 品以上加罚、空弦减分、旋律不在最高几根弦（默认 1–3 弦，top_strings）加罚。
转移代价：手型框移动的品数 ×（1 + 1/时间间隔）；一拍以内最低品移动超过 max_shift（仿校验的换把硬规则）再罚 1000。
全空弦组合不占把位：Viterbi 状态带上"路径上最近一次按弦的手型框"，空弦事件原样继承，下一次按弦的转移代价
按它计算（时间间隔仍取相邻两个事件之差：校验会把空弦起音时仍按着的低音算进手型，按相邻间隔算更保守）。
这里只看每个起音自己弹的音：框的最低品 ≠ 校验的把位。校验的把位是起音时手上所有按弦音的最低品，
包括之前起音、还在发声的低音 / 填充（以及跨过起音的旋律长音）。两者的差别（跨度、换把）由指弹草稿的
自检修补循环兜底（finger.repair）。
音高从高到低必须是 旋律 > 填充 > 低音。某时刻无解时按顺序降级：低音降八度 → 低音升八度 → 去掉填充 →
（去掉填充后）低音降 / 升八度 → 去掉低音 → 旋律升八度 → 旋律降八度，并记录在 fallback。
连奏（FretEvent.legato：旋律音在同一音节里从前一个旋律音滑过来，不重新拨弦）：旋律换弦加罚 LEGATO，
同弦才能弹成击勾弦 / 滑音。只是偏好：换弦也有路径，那就照常拨弦。同弦、两头都按弦时手顺着弦滑过去，
不计手型移动的距离代价（滑音本身就是换把），换把硬规则照旧。尾滑（FretEvent.slide_out：音尾滑出）要按着弦
才滑得出去：旋律放空弦加罚 OPEN_SLIDE_OUT。
合音（FretEvent.harmony：完整旋律的指弹在第 2、4 拍的和弦短音、长音下面的双音）：候选音高按偏好排好，最多取
harmony_max 个、可以一个都不取；每种取法的组合都进 Viterbi，少取一个加 HARMONY_MISS、取偏好靠后的加 HARMONY_RANK，
所以合音放不下（或放了手太别扭）时先丢合音，旋律和低音不为它降级。FretEvent.shape（这时该按的和弦指型）：
伴奏音（填充、合音、新拨的低音）落在指型的品位上减 SHAPE，左手按住指型、旋律在上面走。
接着响的音（held_melody / held_bass：前面起音、这一刻还在响）：音高沿用前一个事件实际弹的（含降级），
位置必须与前一个事件相同（否则罚 HARD），所以占着它的弦、按弦品算进这一刻的手型。接着响的低音放不下时可以在
这里提前止音（bass_pos = None，fallback "bass cut"；之后每个它本该还响的事件加 RELEASE）；交替低音放不下时先换 bass_alt（根音）再移八度。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from ..config import Profile
from ..verify.playability import fingers_needed
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
    legato: bool = False       # 旋律音与前一个事件的旋律音连奏（见模块说明）
    slide_out: bool = False    # 旋律音音尾滑出（见模块说明）
    harmony: tuple[int, ...] = ()        # 合音候选音高（偏好顺序），最多取 harmony_max 个
    harmony_max: int = 0
    harmony_pos: list = field(default_factory=list)   # 取了的合音 [(弦, 品)]（音高从高到低）
    shape: tuple | None = None           # 这时该按的和弦指型（1 → 6 弦品位，None = 不弹）
    bass_alt: int | None = None          # 低音放不下时先换成它（交替低音的五音 → 根音）
    held_melody: bool = False            # 旋律音不在这里起音、是前一个事件的旋律音接着响
    held_bass: bool = False              # 同上，低音（放不下时在这里止音）


def _candidates(pitches: list[int], profile: Profile) -> list[tuple]:
    per = [[(s + 1, p - OPEN_STRINGS[s]) for s in range(6) if 0 <= p - OPEN_STRINGS[s] <= profile.max_fret]
           for p in pitches]
    out = []
    for combo in itertools.product(*per):
        strings = [c[0] for c in combo]
        if any(b <= a for a, b in zip(strings, strings[1:])):
            continue
        fretted = [f for _, f in combo if f > 0]
        if fretted and (max(fretted) - min(fretted) > profile.max_span or fingers_needed(fretted) > profile.max_fingers):
            continue
        out.append(combo)
    return out


def _shape_bonus(positions, shape: tuple | None) -> int:
    """positions 里落在和弦指型品位上的音数。"""
    return 0 if shape is None else sum(1 for s, f in positions if shape[s - 1] == f)


def _static(combo: tuple, has_melody: bool, top_strings: int = 3) -> float:
    fretted = [f for _, f in combo if f > 0]
    cost = 0.0
    if fretted:
        cost += 0.15 * min(fretted) + 0.3 * (max(fretted) - min(fretted)) + 0.5 * max(0, max(fretted) - 7)
    cost -= 0.3 * sum(1 for _, f in combo if f == 0)
    if has_melody and combo:
        cost += 0.5 * max(0, combo[0][0] - top_strings)
    return cost


def _position(combo: tuple, span: int) -> tuple[int, int] | None:
    """手型框（四品格）起点的可行区间 (lo, hi)；全空弦则无约束。"""
    fretted = [f for _, f in combo if f > 0]
    if not fretted:
        return None
    return max(1, max(fretted) - span), min(fretted)


HARD = 1000.0  # 违反校验硬规则的转移罚分（不设为无穷：总要有一条路径）
LEGATO = 5.0   # 连奏的两个旋律音不在同一根弦上（只能重新拨弦）的罚分：用户最在意滑音别被切开，比低把位、1–4 弦的偏好重
OPEN_SLIDE_OUT = 1.5   # 要滑出的旋律音放在空弦上（滑不出去）的罚分
HARMONY_MISS = 1.0     # 合音少取一个
HARMONY_RANK = 0.2     # 合音取了偏好排第 k 个的候选：每靠后一位
SHAPE = 0.3            # 伴奏音落在和弦指型的品位上（减分）
RELEASE = 1.5          # 接着响的低音在这一刻已经止音（每个这样的事件都加）


def _transition(pa: tuple[int, int] | None, pb: tuple[int, int] | None, gap: float, profile: Profile,
                glide: bool = False) -> float:
    """手型框 pa → pb 的代价；任一为 None（尚无按弦）不计。框的 hi 是这一起音自己的最低按弦品；
    校验的把位还算上仍在发声的持续音，不是同一个量——差别由 finger.repair 兜底（见模块说明）。
    glide：同弦滑过去（连奏），移动不计距离代价，只看换把硬规则。"""
    if pa is None or pb is None:
        return 0.0
    dist = max(0, pa[0] - pb[1], pb[0] - pa[1])
    cost = 0.0 if glide else dist * (1.0 + 1.0 / max(gap, 0.25))
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


def _variants(m: int | None, f: int | None, b: int | None, alt: int | None = None,
              held_bass: bool = False, held_melody: bool = False) -> list[tuple]:
    """(旋律, 填充, 低音, 降级说明)，按尝试顺序排列。alt：替代低音，原低音放不下时先试它（"bass root"）。
    held_*：接着响的音，音高不能改：低音不换、不移八度，放不下就止音（"bass cut"）；旋律不移八度。"""
    m, f, b = _ordered(m, f, b)
    out = [(m, f, b, None)]
    if b is not None and not held_bass:
        if alt is not None and alt != b:
            out.append(_ordered(m, f, alt) + ("bass root",))
        out.append(_ordered(m, f, b - 12) + ("bass-12",))
        if (m is None or b + 12 < m) and (f is None or b + 12 < f):
            out.append((m, f, b + 12, "bass+12"))
    if f is not None:
        out.append((m, None, b, "drop fill"))
        if b is not None and not held_bass:
            out.append((m, None, b - 12, "drop fill, bass-12"))
            if m is None or b + 12 < m:
                out.append((m, None, b + 12, "drop fill, bass+12"))
    if b is not None and m is not None:
        out.append((m, None, None, "bass cut" if held_bass else "drop bass"))
    if m is not None and not held_melody:
        out += [(m + 12, None, None, "melody+12"), (m - 12, None, None, "melody-12")]
    return out


@dataclass(frozen=True)
class _Cand:
    """一个事件的一种弹法。combo：(弦, 品)，音高从高到低；其余是各声部的位置（None = 不弹）。
    extra：合音的取舍、接着响的低音止音的代价。"""
    combo: tuple
    melody: tuple[int, int] | None
    fill: tuple[int, int] | None
    harmony: tuple
    bass: tuple[int, int] | None
    extra: float = 0.0


def _subsets(ev: FretEvent, lo: int | None, hi: int | None) -> list[tuple[tuple[int, ...], float]]:
    """合音的取法：从候选里按偏好顺序取 ≤ harmony_max 个、都在 lo 与 hi 之间（不含两端），→ [(音高从高到低, 代价)]。"""
    cands = [(k, p) for k, p in enumerate(ev.harmony) if (lo is None or p > lo) and (hi is None or p < hi)]
    out = []
    for n in range(min(ev.harmony_max, len(cands)), -1, -1):
        for pick in itertools.combinations(cands, n):
            pitches = sorted((p for _, p in pick), reverse=True)
            if len(set(pitches)) == n:
                cost = HARMONY_MISS * (ev.harmony_max - n) + HARMONY_RANK * sum(k for k, _ in pick)
                out.append((tuple(pitches), cost))
    return out


def _cands(ev: FretEvent, m: int | None, f: int | None, b: int | None, profile: Profile) -> list[_Cand]:
    """(旋律, 填充, 低音) 定下之后的全部弹法：合音的每种取法；接着响的低音另有不带它（已止音）的弹法，
    每个这样的事件加 RELEASE：止得越早，付的次数越多。"""
    out = []
    basses = [(b, 0.0)] + ([(None, RELEASE)] if ev.held_bass and b is not None else [])
    for bass, release in basses:
        lo = max((p for p in (bass,) if p is not None), default=None)
        hi = min((p for p in (m, f) if p is not None), default=None)
        for harmony, cost in _subsets(ev, lo, hi) if ev.harmony_max else [((), 0.0)]:
            pitches = [p for p in (m, f, *harmony, bass) if p is not None]
            for combo in _candidates(pitches, profile) if pitches else [()]:
                parts = iter(combo)
                mp = next(parts) if m is not None else None
                fp = next(parts) if f is not None else None
                hp = tuple(next(parts) for _ in harmony)
                bp = next(parts) if bass is not None else None
                out.append(_Cand(combo, mp, fp, hp, bp, cost + release))
    return out


def assign_frets(events: list[FretEvent], profile: Profile, top_strings: int = 3) -> list[FretEvent]:
    """top_strings：旋律音放在 1–top_strings 弦不加罚（指弹 3；只有旋律的旋律谱 4）。"""
    if not events:
        return events
    layers = []  # 每个事件：(弹法列表, 降级说明)
    held_m = held_b = None   # 前一个事件实际弹的旋律 / 低音音高（接着响的音沿用）
    for ev in events:
        m = held_m if ev.held_melody else ev.melody
        b = held_b if ev.held_bass else ev.bass
        release = ev.held_bass and b is not None and m is not None and b >= m   # 旋律低到接着响的低音以下：止音
        for m2, f, b2, label in _variants(m, ev.fill, None if release else b, ev.bass_alt, ev.held_bass,
                                          ev.held_melody):
            cands = _cands(ev, m2, f, b2, profile)
            if cands:
                if ev.fill is not None and f is None and not (label or "").startswith("drop fill"):
                    label = f"{label}, fill dropped" if label else "fill dropped"
                layers.append((cands, "bass cut" if release else label))
                held_m, held_b = m2, b2
                break
        else:
            layers.append(([_Cand((), None, None, (), None)], "dropped"))
            held_m = held_b = None
    span = profile.max_span

    def static_cost(i: int, c: _Cand) -> float:
        ev = events[i]
        open_slide = c.melody is not None and ev.slide_out and c.melody[1] == 0
        fresh = [p for p in (c.fill, *c.harmony, None if ev.held_bass else c.bass) if p is not None]
        return (_static(c.combo, c.melody is not None, top_strings) + (OPEN_SLIDE_OUT if open_slide else 0.0)
                + c.extra - SHAPE * _shape_bonus(fresh, ev.shape))

    def held_cost(ev: FretEvent, a: _Cand, c: _Cand) -> float:
        """接着响的音须与前一个事件同一个位置；止过音的低音不能再响。"""
        moved = ev.held_melody and c.melody is not None and a.melody != c.melody
        moved |= ev.held_bass and c.bass is not None and a.bass != c.bass
        return HARD if moved else 0.0

    # 状态 = (弹法下标, 手型框)；全空弦弹法继承前驱状态的手型框，每种来路各成一个状态
    states = [[(k, _position(c.combo, span)) for k, c in enumerate(layers[0][0])]]
    cost = [static_cost(0, c) for c in layers[0][0]]
    back = []
    for i in range(1, len(layers)):
        gap = events[i].start - events[i - 1].start
        cands, prev_cands = layers[i][0], layers[i - 1][0]
        best: dict[tuple, tuple[float, int]] = {}  # 状态 → (累计代价, 前驱状态下标)
        for k, c in enumerate(cands):
            own, static = _position(c.combo, span), static_cost(i, c)
            for j, (kp, hand) in enumerate(states[-1]):
                a = prev_cands[kp]
                key = (k, hand if own is None else own)
                if events[i].legato and c.melody is not None and a.melody is not None:
                    (s0, f0), (s1, f1) = a.melody, c.melody
                    step = (_transition(hand, own, gap, profile, glide=s0 == s1 and f0 > 0 and f1 > 0)
                            + (0.0 if s0 == s1 else LEGATO))
                else:
                    step = _transition(hand, own, gap, profile)
                total = cost[j] + step + static + held_cost(events[i], a, c)
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
    prev = None
    for ev, (cands, label), layer, s in zip(events, layers, states, path):
        c = cands[layer[s][0]]
        ev.melody_pos, ev.fill_pos, ev.harmony_pos, ev.bass_pos = c.melody, c.fill, list(c.harmony), c.bass
        released = ev.held_bass and c.bass is None and prev is not None and prev.bass is not None
        ev.fallback = label or ("bass cut" if released else None)
        prev = c
    return events
