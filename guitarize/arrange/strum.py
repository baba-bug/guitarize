"""弹唱编配草稿：逐拍和弦 → 指法调指型（按换和弦时的换把整体挑选）→ 每小节套节奏型 → alphaTex（单声部 + 和弦图）。

节奏（rhythm，overrides.strum_rhythm）：
- guitar（默认）：节奏型按原曲吉他分轨逐乐句选（rhythm.choose_patterns，与强力和弦谱同一套 POWER_PATTERNS），开放和弦
  照吉他手的弹法扫：D / A 扫整个指型（A 加重音 ac），M 只闷指型最低的 3 根发声弦（pm；整个开放和弦闷起来是糊的），
  U 上扫 1–3 弦、低音弦接着响（连音线，held_slots），d 是轻的整个指型下扫；刷弦 bd / bu 与力度见 patterns.STROKES。
  每下扫弦响到下一下。
- drums：旧的弹唱——节奏型按段落取鼓 + 吉他分轨的起音选（PATTERNS），sd / su 箭头，没有力度，上扫截断低音。"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Collection

from .. import theory
from ..config import Profile
from ..verify.playability import Violation
from . import atex
from .capo import effective_profile
from .patterns import ALL_PATTERNS, DEFAULT, stroke_effects
from .shapes import Voicing, voicings

UP_STRINGS = 3        # 上扫只扫最细的 3 根弦
BASS_STRINGS = 3      # guitar 节奏的闷音（M）只扫指型最低的 3 根发声弦
RINGING = ("D", "A", "d")   # guitar 节奏里扫整个指型、不闷的下扫：之后的上扫让它的低音弦接着响
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
    """一次扫弦弹到的 (弦, 品)。上扫只扫 1–3 弦；指型在 1–3 弦上没有音时，改扫它最高的 3 根发声弦。
    闷音（M）只扫指型最低的 3 根发声弦。"""
    notes = [(i + 1, f) for i, f in enumerate(v.frets) if f is not None]
    if stroke == "M":
        return notes[-BASS_STRINGS:]
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


def held_slots(chords: list[str], slots: list[str], mute: Collection[int] = ()) -> set[int]:
    """guitar 节奏里低音弦接着响的上扫（小节内第几格）：上扫只扫 1–3 弦，前一下（同一和弦、中间没有休止）是不闷的
    整个指型（RINGING）、或本身也让低音接着响的上扫时，没扫到的弦不断（连音线）。闷音（M）、改成闷音的格（mute）
    之后没有低音可接。chords：逐拍和弦；slots：bar_slots 的结果。"""
    out: set[int] = set()
    ringing = None   # 低音弦还在响的和弦
    for k, stroke in enumerate(slots):
        if stroke == "-":
            continue
        label = chords[k // 2]
        if stroke == "U" and k not in mute and ringing == label:
            out.add(k)
        else:
            ringing = label if stroke in RINGING and k not in mute else None
    return out


def _guitar_item(k: int, n: int, v: Voicing, stroke: str, effects: list[str], dead: bool,
                 held: bool) -> atex.Item:
    """guitar 节奏的一下扫弦（第 k 格起 n 格）：扫到的弦（_notes）+ 刷弦 / 力度（patterns.stroke_effects），
    重音 ac / 闷音 pm 挂在每个新拨的音上（改成闷音时不挂）；held 时指型其余的弦接着响（连音线）。"""
    notes = _notes(v, stroke)
    hold = [nt for nt in _notes(v, "D") if nt not in notes] if held else []
    beat, effect = stroke_effects(stroke)
    fx = {s: [effect] for s, _ in notes} if effect and not dead else None
    return atex.Item(k * 0.5, n * 0.5, sorted(notes + hold), effects + beat, dead=dead,
                     held=frozenset(s for s, _ in hold), head_effects=fx)


def bar_tokens(chords: list[str], pattern: str, shapes: dict[str, Voicing],
               prev: str | None, feel: str = "straight", mute: Collection[int] = (),
               rhythm: str = "guitar") -> tuple[list[str], str | None]:
    """chords：指法调逐拍和弦（4 个）。换和弦或休止之后的延续格改为下扫；无和弦的格为休止。
    指型名标在每小节第一下扫弦和每次换和弦处。
    mute：改成闷音的格（保留扫弦方向和弦的范围）。Shuffle 下反拍扫弦按八分对切分（atex.voice_bar_tokens）。
    rhythm：guitar 每下写刷弦、力度和音符效果（_guitar_item），drums 写 sd / su 箭头（旧的弹唱，见模块说明）。"""
    slot_chord = [chords[k // 2] for k in range(8)]
    slots = bar_slots(chords, pattern)
    held = held_slots(chords, slots, mute) if rhythm == "guitar" else set()
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
            first_in_bar = not any(it.notes for it in items)   # 每小节第一下扫弦都标指型（和弦没变也标）
            mark = [f"ch {atex.quote(v.name)}"] if label != last or first_in_bar else []
            dead = slots[k] == "x" or k in mute
            if rhythm == "guitar":
                items.append(_guitar_item(k, n, v, slots[k], mark, dead, k in held))
            else:
                items.append(atex.Item(k * 0.5, n * 0.5, _notes(v, slots[k]), mark + ["su" if slots[k] == "U" else "sd"],
                                       dead=dead))
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
    stroke: str   # 节奏型的记号（D / U / x，guitar 节奏另有 A / M / d）
    held: bool = False   # guitar 节奏的上扫，低音弦接着响（held_slots）

    @property
    def key(self) -> tuple[int, int]:
        return self.bar, self.slot


def _performed(slot: int, feel: str) -> float:
    """八分格的演奏位置（小节内拍数）。shuffle 下每个反拍起音都被切成八分对的后一半（atex.split_for_swing），
    alphaTab 按 2/3 + 1/3 演奏。"""
    return slot // 2 + (0.0 if slot % 2 == 0 else 2 / 3 if feel == "shuffle" else 0.5)


def strums(chords: list[list[str]], patterns: list[str | None], feel: str, rhythm: str = "guitar") -> list[Strum]:
    """全曲的扫弦序列（按时间）。patterns 里的 None（整小节无和弦）按默认型（反正全是休止）。"""
    out = []
    for i, bar in enumerate(chords):
        slots = bar_slots(bar, ALL_PATTERNS[patterns[i] or DEFAULT])
        held = held_slots(bar, slots) if rhythm == "guitar" else set()
        for k, stroke in enumerate(slots):
            if stroke not in ("-", "r"):
                out.append(Strum(4.0 * i + _performed(k, feel), i, k, bar[k // 2], stroke, k in held))
    return out


def _hand(v: Voicing, stroke: str, held: bool = False) -> int | None:
    """这次扫弦的把位（最低按弦品位，held 时含接着响的低音弦，与 verify 的手形一致）；只有空弦时没有把位。"""
    fretted = [f for _, f in _notes(v, "D" if held else stroke) if f > 0]
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
        pos = _hand(shapes[s.label], s.stroke, s.held)
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


def generate_strum(leadsheet: dict, capo: int, profile: Profile, patterns: list[str | None], title: str,
                   check: Callable[[str], list[Violation]] | None = None, rhythm: str = "guitar") -> str:
    """patterns：逐小节的节奏型名（patterns.ALL_PATTERNS；整小节无和弦可以是 None）。rhythm：guitar / drums（见模块
    说明）；guitar 时每个乐句的第一小节在 `// bar N` 注释里写上节奏型名（layout.pattern_remarks）。
    check：可选的校验回调（alphaTex 文本 → 违规列表，与 verify 同一套检查）。给出时生成后最多 REPAIR_ROUNDS 轮：
    对每个 shift 违规（演奏时刻 t），把 [t−1, t) 内起音的扫弦改成闷音后重新生成。"""
    from .layout import annotate, pattern_remarks   # layout 引用本模块的 beat_chords：在这里引入，避免循环引用

    profile = effective_profile(profile, capo)
    bpb = leadsheet.get("beats_per_bar", 4)
    feel = leadsheet.get("feel", "straight")
    bars = leadsheet["bars"]
    chords = [[c if c == "N" else theory.transpose_label(c, -capo) for c in beat_chords(b, bpb)] for b in bars]
    beats = Counter(c for bar in chords for c in bar if c != "N")   # 按首次出现排序
    cands = {lb: voicings(lb, profile)[:CANDIDATES] for lb in beats}
    chords = [[c if c != "N" and cands[c] else "N" for c in bar] for bar in chords]   # 解析为"无和弦"的标签
    beats = {lb: n for lb, n in beats.items() if cands[lb]}
    seq = strums(chords, patterns, feel, rhythm)
    remarks = pattern_remarks(leadsheet, patterns) if rhythm == "guitar" else {}
    shapes = choose_shapes(seq, beats, cands, profile)
    mute = fallback_mutes(seq, shapes, profile)

    def render() -> str:
        lines: list[str] = []
        prev = None
        for i, b in enumerate(bars):
            muted = {k for j, k in mute if j == i}
            tokens, prev = bar_tokens(chords[i], ALL_PATTERNS[patterns[i] or DEFAULT], shapes, prev, feel, muted, rhythm)
            lines.append(atex.bar_line(tokens, i + 1, b.get("section")))
        diagrams = {v.name: v.frets for v in shapes.values()}
        head = atex.header(title, leadsheet["tempo_bpm"], capo, feel, "tabs", diagrams, key=atex.leadsheet_key(leadsheet))
        return atex.document(head, [annotate(lines, remarks)])

    text = render()
    for _ in range(REPAIR_ROUNDS if check is not None else 0):
        found = {s.key for v in check(text) if v.rule == "shift"
                 for s in seq if v.start - 1.0 - EPS <= s.time < v.start - EPS}
        if found <= mute:
            break
        mute |= found
        text = render()
    return text
