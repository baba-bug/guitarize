"""弹唱编配草稿：逐拍和弦 → 指法调指型（按换和弦时的换把整体挑选）→ 每小节套节奏型 → alphaTex（单声部 + 和弦图）。"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Collection

from .. import theory
from ..config import Profile
from ..verify.playability import Violation
from . import atex
from .capo import effective_profile
from .patterns import PATTERNS
from .shapes import Voicing, voicings

UP_STRINGS = 3        # 上扫只扫最细的 3 根弦
CANDIDATES = 4        # 每个和弦在代价最低的前几个指型里挑
BREAK_COST = 1000.0   # 换和弦违反换把规则（verify 的 shift）
MOVE_COST = 0.1       # 换和弦时把位每移动一品
CLASH_COST = 100.0    # 两个和弦用了同名但品位不同的指型（和弦图会对不上）
REPAIR_ROUNDS = 5     # check 驱动的修补最多几轮
EPS = 1e-6


def beat_chords(lbar: dict, bpb: int = 4) -> list[str]:
    """leadsheet 小节 → 逐拍和弦标签（无和弦为 'N'）。"""
    out = ["N"] * bpb
    for seg in lbar.get("chords", []):
        first = int(round(seg["beat"])) - 1
        for k in range(max(0, first), min(bpb, first + int(round(seg["dur_beats"])))):
            out[k] = seg["label"]
    return out


def _notes(v: Voicing, stroke: str) -> list[tuple[int, int]]:
    """一次扫弦弹到的 (弦, 品)。上扫只扫 1–3 弦；指型在 1–3 弦上没有音时，改扫它最高的 3 根发声弦。"""
    notes = [(i + 1, f) for i, f in enumerate(v.frets) if f is not None]
    if stroke != "U":
        return notes
    return [n for n in notes if n[0] <= UP_STRINGS] or notes[:UP_STRINGS]


def bar_slots(chords: list[str], pattern: str) -> list[str]:
    """一小节 8 个八分格的扫法：无和弦的格为休止；换和弦或休止之后的延续格改为下扫。"""
    slot_chord = [chords[k // 2] for k in range(8)]
    slots = list(pattern)
    for k in range(8):
        if slot_chord[k] == "N":
            slots[k] = "r"
        elif slots[k] == "-" and (k == 0 or slot_chord[k] != slot_chord[k - 1] or slots[k - 1] == "r"):
            slots[k] = "D"
    return slots


def bar_tokens(chords: list[str], pattern: str, shapes: dict[str, Voicing],
               prev: str | None, feel: str = "straight", mute: Collection[int] = ()) -> tuple[list[str], str | None]:
    """chords：指法调逐拍和弦（4 个）。换和弦或休止之后的延续格改为下扫；无和弦的格为休止。
    mute：改成闷音的格（保留扫弦方向和弦的范围）。Shuffle 下反拍扫弦按八分对切分（atex.voice_bar_tokens）。"""
    slot_chord = [chords[k // 2] for k in range(8)]
    slots = bar_slots(chords, pattern)
    items: list[atex.Item] = []
    last = prev
    k = 0
    while k < 8:
        n = 1
        if slots[k] == "r":
            while k + n < 8 and slots[k + n] == "r":
                n += 1
            items.append(atex.Item(k * 0.5, n * 0.5, []))
            last = None
        else:
            while k + n < 8 and slots[k + n] == "-":
                n += 1
            label = slot_chord[k]
            v = shapes[label]
            effects = ([f"ch {atex.quote(v.name)}"] if label != last else []) + ["su" if slots[k] == "U" else "sd"]
            items.append(atex.Item(k * 0.5, n * 0.5, _notes(v, slots[k]), effects,
                                   dead=slots[k] == "x" or k in mute))
            last = label
        k += n
    return atex.voice_bar_tokens(items, feel), last


@dataclass(frozen=True)
class Strum:
    """一次扫弦。time 是演奏时刻（绝对拍，第 1 小节第 1 拍 = 0；shuffle 的反拍在 2/3 拍处）。"""
    time: float
    bar: int      # 0-based
    slot: int     # 小节内第几个八分格
    label: str
    stroke: str   # D / U / x

    @property
    def key(self) -> tuple[int, int]:
        return self.bar, self.slot


def _performed(slot: int, feel: str) -> float:
    """八分格的演奏位置（小节内拍数）。shuffle 下每个反拍起音都被切成八分对的后一半（atex.split_for_swing），
    alphaTab 按 2/3 + 1/3 演奏。"""
    return slot // 2 + (0.0 if slot % 2 == 0 else 2 / 3 if feel == "shuffle" else 0.5)


def strums(chords: list[list[str]], patterns: list[str], feel: str) -> list[Strum]:
    """全曲的扫弦序列（按时间）。"""
    out = []
    for i, bar in enumerate(chords):
        for k, stroke in enumerate(bar_slots(bar, PATTERNS[patterns[i]])):
            if stroke in ("D", "U", "x"):
                out.append(Strum(4.0 * i + _performed(k, feel), i, k, bar[k // 2], stroke))
    return out


def _hand(v: Voicing, stroke: str) -> int | None:
    """这次扫弦的把位（最低按弦品位）；只有空弦时没有把位。"""
    fretted = [f for _, f in _notes(v, stroke) if f > 0]
    return min(fretted) if fretted else None


def shift_cost(seq: list[Strum], shapes: dict[str, Voicing], profile: Profile,
               mute: Collection[tuple[int, int]] = ()) -> tuple[float, list[float]]:
    """按 verify 的 shift 规则走一遍：有把位的起音与上一个有把位的起音相隔不到 1 拍且把位差超过 max_shift 即违规
    （只有空弦的扫弦和闷音没有把位，上一个把位跨过它们保留）。返回（换和弦代价, 违规时刻）：
    违规每次 BREAK_COST，其余换和弦每移动一品 MOVE_COST。"""
    cost, breaks = 0.0, []
    last: tuple[float, int, str] | None = None   # (时刻, 把位, 和弦)
    for s in seq:
        if s.stroke == "x" or s.key in mute:
            continue
        pos = _hand(shapes[s.label], s.stroke)
        if pos is None:
            continue
        if last is not None and s.time - last[0] < 1.0 - EPS and abs(pos - last[1]) > profile.max_shift:
            cost += BREAK_COST
            breaks.append(s.time)
        elif last is not None and s.label != last[2]:
            cost += MOVE_COST * abs(pos - last[1])
        last = (s.time, pos, s.label)
    return cost, breaks


def _clashes(shapes: dict[str, Voicing]) -> int:
    seen: dict[str, tuple] = {}
    n = 0
    for v in shapes.values():
        n += v.name in seen and seen[v.name] != v.frets
        seen.setdefault(v.name, v.frets)
    return n


def choose_shapes(seq: list[Strum], beats: dict[str, float], cands: dict[str, list[Voicing]],
                  profile: Profile) -> dict[str, Voicing]:
    """每个和弦标签定一个指型（同一和弦全曲一致，和弦图只出现一次）。
    总代价 = Σ 指型代价 × 拍数 + 换和弦代价（shift_cost）+ 同名异形罚分。
    从各自第 0 个候选出发做局部搜索：每轮取使总代价下降最多的单个标签改动，直到不再下降（顺序确定）。"""
    labels = list(beats)

    def total(pick: dict[str, int]) -> float:
        shapes = {lb: cands[lb][pick[lb]] for lb in labels}
        return (sum(shapes[lb].cost * beats[lb] for lb in labels) + CLASH_COST * _clashes(shapes)
                + shift_cost(seq, shapes, profile)[0])

    pick = {lb: 0 for lb in labels}
    best = total(pick)
    while True:
        move = None
        for lb in labels:
            for j in range(len(cands[lb])):
                if j != pick[lb]:
                    c = total({**pick, lb: j})
                    if c < (best if move is None else move[0]) - 1e-9:
                        move = (c, lb, j)
        if move is None:
            return {lb: cands[lb][pick[lb]] for lb in labels}
        best, lb, j = move
        pick[lb] = j


def fallback_mutes(seq: list[Strum], shapes: dict[str, Voicing], profile: Profile) -> set[tuple[int, int]]:
    """选完指型仍违反换把规则的换和弦（时刻 t）：把 t 之前不到 1 拍内起音的扫弦改成闷音，verify 不把闷音算进手形。
    规则是相隔 < 1 拍才算，恰好 1 拍前的正拍扫弦保留（D-DU-UDU 这类节奏型只闷掉换和弦前的反拍）。"""
    mute: set[tuple[int, int]] = set()
    while True:
        _, breaks = shift_cost(seq, shapes, profile, mute)
        new = {s.key for t in breaks for s in seq if t - 1.0 + EPS < s.time < t - EPS} - mute
        if not new:
            return mute
        mute |= new


def generate_strum(leadsheet: dict, capo: int, profile: Profile, patterns: list[str], title: str,
                   check: Callable[[str], list[Violation]] | None = None) -> str:
    """check：可选的校验回调（alphaTex 文本 → 违规列表，与 verify 同一套检查）。给出时生成后最多 REPAIR_ROUNDS 轮：
    对每个 shift 违规（演奏时刻 t），把 [t−1, t) 内起音的扫弦改成闷音后重新生成。"""
    profile = effective_profile(profile, capo)
    bpb = leadsheet.get("beats_per_bar", 4)
    feel = leadsheet.get("feel", "straight")
    bars = leadsheet["bars"]
    chords = [[c if c == "N" else theory.transpose_label(c, -capo) for c in beat_chords(b, bpb)] for b in bars]
    beats = Counter(c for bar in chords for c in bar if c != "N")   # 按首次出现排序
    cands = {lb: voicings(lb, profile)[:CANDIDATES] for lb in beats}
    chords = [[c if c != "N" and cands[c] else "N" for c in bar] for bar in chords]   # 解析为"无和弦"的标签
    beats = {lb: n for lb, n in beats.items() if cands[lb]}
    seq = strums(chords, patterns, feel)
    shapes = choose_shapes(seq, beats, cands, profile)
    mute = fallback_mutes(seq, shapes, profile)

    def render() -> str:
        lines: list[str] = []
        prev = None
        for i, b in enumerate(bars):
            muted = {k for j, k in mute if j == i}
            tokens, prev = bar_tokens(chords[i], PATTERNS[patterns[i]], shapes, prev, feel, muted)
            lines.append(atex.bar_line(tokens, i + 1, b.get("section")))
        diagrams = {v.name: v.frets for v in shapes.values()}
        head = atex.header(title, leadsheet["tempo_bpm"], capo, feel, "tabs", diagrams, key=atex.leadsheet_key(leadsheet))
        return atex.document(head, [lines])

    text = render()
    for _ in range(REPAIR_ROUNDS if check is not None else 0):
        found = {s.key for v in check(text) if v.rule == "shift"
                 for s in seq if v.start - 1.0 - EPS <= s.time < v.start - EPS}
        if found <= mute:
            break
        mute |= found
        text = render()
    return text
