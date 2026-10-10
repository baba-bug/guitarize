"""指弹编配草稿：旋律骨架（voice 1）+ 低音与填充（voice 2），指法由 Viterbi 分配。
给出自检（与 verify 同一套可弹性检查）时按违规逐轮修补：截短仍按着的音、去掉填充 / 低音；
换把找不到 t 时可修的音时，改在前一个起音处修，最后才把旋律音移八度。
每处指法降级与修补都写进该小节的 `// bar N` 注释，方便审阅。"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from .. import theory
from ..config import Profile
from ..skeleton import skeleton_mask
from ..verify.playability import Violation
from . import atex
from .capo import OctavePolicy, effective_profile, octave_plan
from .fretting import FretEvent, assign_frets
from .shapes import OPEN_STRINGS, voicings
from .strum import beat_chords

BASS_LOW = 40        # E2：低音区间 E2–D#3（指法调）
MAX_MELODY_NOTE = 2.0
FILL_GAP = 3         # 填充音至少比它持续期间的最低旋律音低 3 个半音
REPAIR_ROUNDS = 10
HAND_RULES = ("span", "fingers", "shift")   # 校验把仍按着的音算进手型的规则：可以靠截短 / 去音修补
EPS = 1e-6


def written_position(perf: float, feel: str) -> float:
    """演奏位置（拍）→ 记谱位置（八分格）。Shuffle：2/3 → 反拍八分，1/3 → 拍点；直拍取最近的八分。"""
    whole = int(np.floor(perf + 1e-6))
    frac = perf - whole
    if feel == "shuffle":
        if frac < 1 / 2:
            return float(whole)
        if frac < 5 / 6:
            return whole + 0.5
        return float(whole + 1)
    return round(perf * 2) / 2


def section_bars(leadsheet: dict) -> list[int]:
    """带段落标记的小节（0-based）。"""
    return [i for i, b in enumerate(leadsheet["bars"]) if b.get("section")]


def skeleton_notes(leadsheet: dict, profile: Profile) -> list[tuple[float, int]]:
    notes = sorted((m for b in leadsheet["bars"] for m in b["melody"]), key=lambda m: m["start_beats"])
    mask = skeleton_mask([m["start_beats"] for m in notes], [m["dur_beats"] for m in notes])
    feel = leadsheet.get("feel", "straight")
    at: dict[float, int] = {}
    for m, keep in zip(notes, mask):
        if keep:
            at.setdefault(written_position(m["start_beats"], feel), m["midi"])
    out, per_beat = [], {}
    for pos in sorted(at):
        beat = int(pos)
        if per_beat.get(beat, 0) < profile.melody_max_per_beat:
            per_beat[beat] = per_beat.get(beat, 0) + 1
            out.append((pos, at[pos]))
    return out


def melody_line(leadsheet: dict, capo: int, profile: Profile, octave: OctavePolicy = "song") -> list[tuple[float, int]]:
    """骨架旋律 → (记谱位置, 指法调音高)；八度移位见 capo.octave_plan（默认整曲一个移位，保持人声的高低关系）。"""
    mel = [(pos, p - capo) for pos, p in skeleton_notes(leadsheet, profile)]
    out = list(mel)
    for idx, shift, _ in octave_plan(mel, section_bars(leadsheet), octave):
        for i in idx:
            out[i] = (mel[i][0], mel[i][1] + shift)
    return out


FILL_LOW, FILL_HIGH = 50, 64   # 填充音优先区间 D3–E4


def _parse(label: str) -> tuple[int, str, int | None]:
    m = re.match(r"^([A-G][#b]?)5$", label)
    if m:
        return theory.pc(m.group(1)), "power", None
    return theory.parse_chord(label)


def _bass_and_fill(label: str) -> tuple[int, int]:
    """(低音指法调音高, 填充音 pc)。填充 = 三音；三音已是低音时用根音；强力和弦用五音。"""
    root, quality, bass = _parse(label)
    bass_pc = bass if bass is not None else root
    if quality == "power":
        fill = (root + 7) % 12
    else:
        third = (root + (3 if quality in ("min", "min7", "dim", "min7b5") else 4)) % 12
        fill = root if third == bass_pc else third
    return BASS_LOW + ((bass_pc - BASS_LOW) % 12), fill


def place_fill(pc: int, bass: int, top: int | None) -> int | None:
    """填充音：高于 bass、且比 top（填充持续期间的最低旋律音）至少低 3 个半音，取该音名最高的音，优先落在 50–64。
    无旋律时上限 E4。"""
    limit = 65 if top is None else top - FILL_GAP + 1
    inside = [p for p in range(bass + 1, limit) if p % 12 == pc]
    preferred = [p for p in inside if FILL_LOW <= p <= FILL_HIGH]
    if preferred:
        return max(preferred)
    return max(inside) if inside else None


def melody_floor(melody: list[tuple[float, float, int]], pos: float, dur: float) -> int | None:
    """melody：(位置, 时值, 音高)。[pos, pos + dur) 内发声的最低旋律音（填充音的天花板）；没有则 None。"""
    return min((p for s, d, p in melody if s < pos + dur - EPS and s + d > pos + EPS), default=None)


def bass_events(leadsheet: dict, capo: int, profile: Profile) -> list[tuple[float, float, int, int]]:
    """第 1 拍放低音 + 填充；第 3 拍换和弦则放新和弦的低音 + 填充，和弦不变且难度允许时重复根音 + 填充。"""
    out = []
    for i, b in enumerate(leadsheet["bars"]):
        chords = [c if c == "N" else theory.transpose_label(c, -capo) for c in beat_chords(b)]
        start = i * 4.0
        first, third = chords[0], chords[2]
        if first != "N":
            if third == first and profile.bass_on_beat3:
                out += [(start, 2.0, *_bass_and_fill(first)), (start + 2, 2.0, *_bass_and_fill(first))]
            elif third == first:
                out.append((start, 4.0, *_bass_and_fill(first)))
            else:
                out.append((start, 2.0, *_bass_and_fill(first)))
        if third != "N" and third != first:
            out.append((start + 2, 2.0, *_bass_and_fill(third)))
    return out


def _voice_lines(n_bars: int, items: list[tuple[float, float, list, bool]], sections: list[str | None],
                 feel: str, effects: dict[float, list[str]] | None = None) -> list[str]:
    """items：(记谱位置, 时值, [(弦, 品)], 是否连音线延续)（见 Placed.pieces）；按小节铺满（空隙补休止），
    Shuffle 下按八分对切分。effects：{记谱位置: 拍效果}（和弦指型、力度），挂在该位置起音的音 / 休止上
    （休止在这些位置切开，保证有一拍从这里开始）。"""
    marks = {round(k, 6): v for k, v in (effects or {}).items()}
    by_bar: dict[int, list] = {}
    for item in sorted(items, key=lambda x: x[0]):
        by_bar.setdefault(int(item[0] // 4), []).append(item)
    lines = []
    for i in range(n_bars):
        t, end, bar_items = i * 4.0, i * 4.0 + 4.0, []

        def rests(a: float, b: float) -> None:
            edges = [a, *sorted(k for k in marks if a + 1e-6 < k < b - 1e-6), b]
            for x, y in zip(edges, edges[1:]):
                bar_items.append(atex.Item(x - i * 4.0, y - x, [], marks.get(round(x, 6), ())))

        for pos, dur, notes, tie in by_bar.get(i, []):
            if pos > t + 1e-6:
                rests(t, pos)
            dur = min(dur, end - pos)
            bar_items.append(atex.Item(pos - i * 4.0, dur, notes, () if tie else marks.get(round(pos, 6), ()), tie=tie))
            t = pos + dur
        if t < end - 1e-6:
            rests(t, end)
        lines.append(atex.bar_line(atex.voice_bar_tokens(bar_items, feel), i + 1, sections[i]))
    return lines


def _starts_beat(pieces: list[tuple[float, float, list, bool]], x: float) -> bool:
    """声部在 x 处能挂拍效果：x 不在某个发声片段的中间，也不是连音线延续片段的起点（休止会在 x 处切开）。"""
    for pos, dur, notes, tie in pieces:
        if notes and pos - EPS < x < pos + dur - EPS and (pos < x - EPS or tie):
            return False
    return True


def shape_marks(leadsheet: dict, capo: int, profile: Profile) -> tuple[dict[float, str], dict[str, tuple]]:
    """指弹谱上该按的和弦指型：每小节第 1 拍都标；第 3 拍（低音所在）指型变了再标（无和弦之后重标）。
    → ({记谱位置: 指型名}, {指型名: 和弦图品位（1 弦 → 6 弦）})。"""
    marks: dict[float, str] = {}
    diagrams: dict[str, tuple] = {}
    last = None
    for i, b in enumerate(leadsheet["bars"]):
        chords = beat_chords(b)
        for k in (0, 2):
            if chords[k] == "N":
                last = None
                continue
            try:
                options = voicings(theory.transpose_label(chords[k], -capo), profile)
            except ValueError:
                continue
            if not options:
                continue
            v = options[0]
            if v.name != last or k == 0:   # 每小节第 1 拍都标（和弦没变也标），小节中间换了再标
                marks[i * 4.0 + k] = v.name
                diagrams.setdefault(v.name, v.frets)
                last = v.name
    return marks, diagrams


@dataclass
class Placed:
    """声部里的一个音 / 和弦（记谱位置、时值，拍）。where：voice 1 为 [旋律]，voice 2 为 [填充, 低音]；None = 不弹。
    stop：where 下标 → 这个音单独提前止音的位置（其余的音接着响，渲染成连音线延续）。moved：修补时已移过八度。"""
    pos: float
    dur: float
    where: list
    stop: dict = field(default_factory=dict)
    moved: bool = False

    def end(self, k: int) -> float:
        return min(self.stop.get(k, self.pos + self.dur), self.pos + self.dur)

    def sounding(self, t: float) -> list[int]:
        """起音早于 t、t 时仍在发声的音（where 下标）。"""
        if self.pos >= t - EPS:
            return []
        return [k for k, w in enumerate(self.where) if w is not None and t < self.end(k) - EPS]

    def holds(self, t: float) -> bool:
        return bool(self.sounding(t))

    def cut(self, t: float) -> None:
        """整体截止到 t。"""
        self.dur = t - self.pos

    def pieces(self) -> list[tuple[float, float, list, bool]]:
        """→ [(位置, 时值, [(弦, 品)], 是否连音线延续)]：在单独止音处切开，后面的片段只留还在响的音、写成连音线；
        都不响了写休止。"""
        end = self.pos + self.dur
        notes = [k for k, w in enumerate(self.where) if w is not None]
        edges = sorted({self.pos, end} | {self.end(k) for k in notes if self.pos + EPS < self.end(k) < end - EPS})
        out = []
        for a, b in zip(edges, edges[1:]):
            on = [self.where[k] for k in notes if self.end(k) > a + EPS]
            out.append((a, b - a, on, bool(on) and a > self.pos + EPS))
        return out


def _at(t: float) -> str:
    return f"@{t % 4 + 1:g}"   # 小节内第几拍（1 起）


def _strings(it: Placed) -> set[int]:
    return {w[0] for w in it.where if w is not None}


def _cut_note(it: Placed, k: int, t: float) -> None:
    """只让 where[k] 在 t 止音；剩下的音都比它先停时整体截短。"""
    it.stop[k] = t
    it.dur = max(it.end(j) for j, w in enumerate(it.where) if w is not None) - it.pos
    it.stop = {j: e for j, e in it.stop.items() if e < it.pos + it.dur - EPS}


def cut_same_string(v1: list[Placed], v2: list[Placed]) -> list[tuple[float, str]]:
    """同一根弦不能一边按住一边再弹：低音 / 填充持续期间有旋律音弹在它的弦上 → 只截这根弦上的音（"fill cut" /
    "bass cut"），另一个音接着响（写成连音线延续）；旋律音持续期间有低音 / 填充弹在它的弦上 → 旋律截止到该处。
    返回 (记谱位置, 说明)。"""
    out = []
    for it in v2:
        for k, name in ((0, "fill cut"), (1, "bass cut")):
            w = it.where[k]
            hit = None if w is None else min((o.pos for o in v1 if k in it.sounding(o.pos) and w[0] in _strings(o)),
                                             default=None)
            if hit is not None:
                _cut_note(it, k, hit)
                out.append((hit, f"{name} {_at(hit)} (same string)"))
    for it in v1:
        hit = min((o.pos for o in v2 if it.holds(o.pos) and _strings(it) & _strings(o)), default=None)
        if hit is not None:
            it.cut(hit)
            out.append((hit, f"melody cut {_at(hit)} (same string)"))
    return sorted(out)


def _pitch(w: tuple[int, int]) -> int:
    return OPEN_STRINGS[w[0] - 1] + w[1]


def cut_crossings(v1: list[Placed], v2: list[Placed]) -> list[tuple[float, str]]:
    """填充本来按整段持续期间的最低旋律音放低至少 3 个半音；但指法降级（如 melody-12）会把后面的旋律压下来。
    低音 / 填充持续期间有旋律音不比它高出 3 个半音 → 截止到这个旋律音。返回 (记谱位置, 说明)。"""
    out = []
    for it in v2:
        hit = min((o.pos for o in v1 if o.where[0] is not None
                   and any(_pitch(o.where[0]) - _pitch(it.where[k]) < FILL_GAP for k in it.sounding(o.pos))),
                  default=None)
        if hit is not None:
            it.cut(hit)
            out.append((hit, f"bass cut {_at(hit)} (crossing)"))
    return out


def _fretted(w: tuple[int, int] | None) -> bool:
    return w is not None and w[1] > 0


def _attacked(items: list[Placed], t: float) -> list[Placed]:
    return [it for it in items if abs(it.pos - t) < EPS and any(w is not None for w in it.where)]


def hand(v1: list[Placed], v2: list[Placed], t: float) -> list[int]:
    """t 时手上按着的品（记谱时间），与校验的 _hand_frets 同一模型：t 时新拨的有品音 + 之前起音、仍在发声的有品音，
    新拨的有品音所在的弦上旧音让位。空表示没有把位。"""
    fresh = {w for it in _attacked(v1 + v2, t) for w in it.where if _fretted(w)}
    strings = {s for s, _ in fresh}
    held = {it.where[k] for it in v1 + v2 for k in it.sounding(t)
            if _fretted(it.where[k]) and it.where[k][0] not in strings}
    return sorted(f for _, f in fresh | held)


def previous_hand(v1: list[Placed], v2: list[Placed], t: float) -> float | None:
    """t 之前最近一个手上有按弦音的起音：校验的换把规则拿 t 的把位和它比。"""
    onsets = {it.pos for it in v1 + v2 if it.pos < t - EPS and _attacked([it], it.pos)}
    return next((o for o in sorted(onsets, reverse=True) if hand(v1, v2, o)), None)


def jump(v1: list[Placed], v2: list[Placed], t: float) -> int:
    """模型里 t 的换把幅度：t 的把位（最低按弦品）与 1 拍以内前一个把位之差；t 没有把位、或 1 拍内没有前一个把位则为 0。
    记谱时间与演奏时间对"相隔是否不到 1 拍"的判断一致（起音都在八分格上，Shuffle 的反拍成对后移）。"""
    here, prev = hand(v1, v2, t), previous_hand(v1, v2, t)
    if not here or prev is None or t - prev > 1 - EPS:
        return 0
    return abs(here[0] - hand(v1, v2, prev)[0])


def _cut_held(items: list[Placed], t: float) -> bool:
    """截短第一个 t 之前起音、t 时仍有按弦音在响的音 / 和弦。"""
    for it in items:
        if any(_fretted(it.where[k]) for k in it.sounding(t)):
            it.cut(t)
            return True
    return False


def _drop(v2: list[Placed], t: float, k: int) -> bool:
    """去掉 t 时起音的按弦填充（k = 0）或按弦低音（k = 1）。"""
    for it in v2:
        if abs(it.pos - t) < EPS and _fretted(it.where[k]):
            it.where[k] = None
            return True
    return False


Step = tuple[float, str, Callable[[list[Placed], list[Placed]], bool]]   # (修补位置, 说明, 动作(v1, v2) → 是否做了)


def _steps_at(t: float) -> list[Step]:
    """t 时的修补顺序：截短仍按着的低音 / 填充 → 截短仍按着的旋律 → 去掉 t 时起音的填充 → 去掉 t 时起音的低音。"""
    return [(t, "bass cut", lambda v1, v2: _cut_held(v2, t)), (t, "melody cut", lambda v1, v2: _cut_held(v1, t)),
            (t, "drop fill", lambda v1, v2: _drop(v2, t, 0)), (t, "drop bass", lambda v1, v2: _drop(v2, t, 1))]


def _shift_steps(t: float, prev: float | None) -> list[Step]:
    """换把的修补顺序：先截短（t 时的两步，再是前一个手上有音的起音 prev 时仍按着的低音 / 填充），
    再去音（t 起音的填充、低音，然后 prev 起音的填充、低音）。"""
    at_t = _steps_at(t)
    if prev is None:
        return at_t
    return (at_t[:2] + [(prev, "bass cut", lambda v1, v2: _cut_held(v2, prev))] + at_t[2:]
            + [(prev, "drop fill", lambda v1, v2: _drop(v2, prev, 0)),
               (prev, "drop bass", lambda v1, v2: _drop(v2, prev, 1))])


def _repair_at(v1: list[Placed], v2: list[Placed], t: float) -> str | None:
    return next((name for _, name, act in _steps_at(t) if act(v1, v2)), None)


def _melody_room(it: Placed, w: tuple[int, int], v2: list[Placed]) -> bool:
    """旋律音换到 w 后，它发声期间响着的低音 / 填充不和它同弦、且都比它低至少 FILL_GAP 个半音。"""
    start, end = it.pos, it.pos + it.dur
    for o in v2:
        for k, x in enumerate(o.where):
            if x is not None and o.pos < end - EPS and o.end(k) > start + EPS:
                if x[0] == w[0] or _pitch(w) - _pitch(x) < FILL_GAP:
                    return False
    return True


def _move_octave(v1: list[Placed], v2: list[Placed], t: float, prev: float,
                 max_fret: int) -> tuple[float, str] | None:
    """(b) 最后一招：t 与 prev 之间的换把只涉及两个旋律音（没有低音 / 填充）时，把其中一个移八度（±12），
    选让跳动缩得最多的那个（落到空弦则没有把位），品位须在 max_fret 以内。每个音最多移一次。返回 (位置, 说明)。"""
    here, there = hand(v1, v2, t), hand(v1, v2, prev)
    b = next((it for it in _attacked(v1, t) if _fretted(it.where[0])), None)
    a = next((it for it in v1 if _fretted(it.where[0]) and (abs(it.pos - prev) < EPS or it.holds(prev))), None)
    if a is None or b is None or here != [b.where[0][1]] or there != [a.where[0][1]]:
        return None
    old = abs(here[0] - there[0])
    best = None
    for it, other in ((b, a), (a, b)):
        if it.moved:
            continue
        for shift in (12, -12):
            pitch = _pitch(it.where[0]) + shift
            for s in range(1, 7):
                w = (s, pitch - OPEN_STRINGS[s - 1])
                if not 0 <= w[1] <= max_fret or not _melody_room(it, w, v2):
                    continue
                dist = abs(w[1] - other.where[0][1]) if w[1] > 0 else 0
                key = (dist, max(0, s - 3), w[1])
                if dist < old and (best is None or key < best[0]):
                    best = (key, it, w, shift)
    if best is None:
        return None
    _, it, w, shift = best
    it.where[0], it.moved = w, True
    return it.pos, f"melody{shift:+d}"


def _repair_shift(v1: list[Placed], v2: list[Placed], t: float, max_fret: int,
                  touched: set[float]) -> tuple[float, str] | None:
    """按 _shift_steps 的顺序，先取第一个能让模型里的跳动（jump）变小的一步；都不能时取第一个做得了的
    （为后面的步骤让路）；什么都做不了时 (b) 移旋律八度。前一个起音本轮已经改过的留到下一轮。"""
    prev = previous_hand(v1, v2, t)
    if prev is not None and prev in touched:
        return None
    old, first = jump(v1, v2, t), None
    for at, name, act in _shift_steps(t, prev):
        trial = copy.deepcopy((v1, v2))
        if not act(*trial):
            continue
        if jump(*trial, t) < old:
            act(v1, v2)
            return at, name
        first = first or (at, name, act)
    if first is not None:
        first[2](v1, v2)
        return first[0], first[1]
    return None if prev is None else _move_octave(v1, v2, t, prev, max_fret)


def repair(v1: list[Placed], v2: list[Placed], violations: Iterable[Violation], feel: str,
           max_fret: int, max_shift: int | None = None) -> list[tuple[float, str]]:
    """按自检违规修补（就地修改 v1 / v2）。Violation.start 是演奏时间，先换回记谱位置 t。每个 t 每轮只修一处：
    - 跨度 / 手指数：依次 1. 截短 t 之前起音、t 时仍按着的低音 / 填充；2. 截短 t 时仍按着的旋律音；
      3. 去掉 t 时起音的填充；4. 再去掉低音。
    - 换把：同样的顺序，但截短之后先截短前一个手上有音的起音 t′ 时仍按着的低音 / 填充（宁可截短也不丢新起音），
      去音之后再去掉 t′ 起音的填充、低音；优先取能缩小跳动的一步（见 _repair_shift），
      都做不了时 (b) 把 t 或 t′ 的旋律音移八度（melody±12）。
    只动按弦音（空弦不占手型），只修手型类规则。违规来自本轮修补之前的文本：给了 max_shift 时，
    本轮前面的修补已让 t 的跳动 ≤ max_shift 的换把不再修（免得白白去音）。返回 (修补位置, 说明)；空表示无可修。"""
    done, seen, touched = [], set(), set()
    for v in violations:
        t = written_position(v.start, feel)
        if v.rule not in HAND_RULES or t in seen:
            continue
        seen.add(t)
        if v.rule == "shift":
            if max_shift is not None and jump(v1, v2, t) <= max_shift:
                continue
            found = _repair_shift(v1, v2, t, max_fret, touched)
        else:
            what = _repair_at(v1, v2, t)
            found = (t, what) if what else None
        if found:
            touched.add(found[0])
            done.append((found[0], f"{found[1]} {_at(found[0])} ({v.rule})"))
    return done


def annotate(lines: list[str], remarks: dict[int, list[str]]) -> list[str]:
    """把降级 / 修补说明接到对应小节（0-based）的 `// bar N ...` 注释行末尾。"""
    out = []
    for i, line in enumerate(lines):
        if remarks.get(i):
            head, body = line.split("\n", 1)
            line = head + "".join(f" · {r}" for r in remarks[i]) + "\n" + body
        out.append(line)
    return out


def generate_finger(leadsheet: dict, capo: int, profile: Profile, title: str,
                    check: Callable[[str], list[Violation]] | None = None, octave: OctavePolicy = "song") -> str:
    """check：自检（alphaTex → 违规列表）。给出时渲染后按违规修补、重新渲染，最多 10 轮。
    octave：旋律八度策略（song.yaml 的 overrides.melody_octave，见 capo.octave_plan）。
    谱上在低音声部标出该按的和弦指型（附和弦图）；旋律 f、伴奏 mf：试听时旋律稍突出（伴奏再轻会让与原曲器乐分轨的和声相似度下降）。"""
    profile = effective_profile(profile, capo)
    feel = leadsheet.get("feel", "straight")
    n_bars = len(leadsheet["bars"])
    mel = melody_line(leadsheet, capo, profile, octave)
    items = []
    for j, (pos, p) in enumerate(mel):
        nxt = mel[j + 1][0] if j + 1 < len(mel) else pos + MAX_MELODY_NOTE
        items.append((pos, min(nxt, pos + MAX_MELODY_NOTE, (int(pos // 4) + 1) * 4.0) - pos, p))
    bass = bass_events(leadsheet, capo, profile)
    m_at = {pos: p for pos, _, p in items}
    b_at = {pos: (p, place_fill(fill_pc, p, melody_floor(items, pos, dur))) for pos, dur, p, fill_pc in bass}
    starts = sorted(set(m_at) | set(b_at))
    events = assign_frets([FretEvent(s, m_at.get(s), b_at.get(s, (None, None))[0], fill=b_at.get(s, (None, None))[1])
                           for s in starts], profile)
    at = {e.start: e for e in events}
    v1 = [Placed(pos, d, [at[pos].melody_pos]) for pos, d, _ in items]
    v2 = [Placed(pos, d, [at[pos].fill_pos, at[pos].bass_pos]) for pos, d, _, _ in bass]
    remarks: dict[int, list[str]] = {}

    def remark(found: Iterable[tuple[float, str]]) -> None:
        for t, text in found:
            remarks.setdefault(int(t // 4), []).append(text)

    remark((e.start, f"{e.fallback} {_at(e.start)}") for e in events if e.fallback)
    remark(cut_same_string(v1, v2))
    remark(cut_crossings(v1, v2))
    sections = [b.get("section") for b in leadsheet["bars"]]
    marks, diagrams = shape_marks(leadsheet, capo, profile)
    head = atex.header(title, leadsheet["tempo_bpm"], capo, feel, "score tabs", diagrams, key=atex.leadsheet_key(leadsheet))

    def render() -> str:
        pieces1 = [p for it in v1 for p in it.pieces()]
        effects1: dict[float, list[str]] = {0.0: ["dy f"]}
        effects2: dict[float, list[str]] = {0.0: ["dy mf"]}
        for pos, name in marks.items():   # 指型名尽量挂在旋律声部（黑色、醒目）；那里正被连音线延续占着、
            # 或是段落第一拍（段落名与旋律声部的和弦名画在同一行会重叠）时挂在低音声部（另起一行）
            section_start = pos % 4 == 0 and bool(sections[int(pos // 4)])
            target = effects1 if _starts_beat(pieces1, pos) and not section_start else effects2
            target.setdefault(pos, []).insert(0, f"ch {atex.quote(name)}")
        lines1 = _voice_lines(n_bars, pieces1, sections, feel, effects1)
        lines2 = _voice_lines(n_bars, [p for it in v2 for p in it.pieces()], [None] * n_bars, feel, effects2)
        return atex.document(head, [annotate(lines1, remarks), lines2])

    text = render()
    for _ in range(REPAIR_ROUNDS if check else 0):
        found = check(text)
        if not found:
            break
        done = repair(v1, v2, found, feel, profile.max_fret, profile.max_shift)
        if not done:
            break
        remark(done)
        text = render()
    return text
