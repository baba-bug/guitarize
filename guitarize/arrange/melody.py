"""旋律谱草稿（arr = melody）：只弹人声旋律的单声部谱。用户的流程是"先保证旋律没问题，再加伴奏"：
这份谱逐音对照人声（verify 按全部参考音评旋律、按人声分轨评相似度），核对准了再编指弹。

- 音：leadsheet 的每一个旋律音（不取骨架、每拍不限个数）；八度按 capo.octave_plan，与指弹同一策略
  （默认整曲一个移位，保持人声的高低关系；overrides.melody_octave 同样生效）。
- 节奏：起音、时值都按原唱落到记谱格上；休止照旧是休止（不把音延长到下一个音）；跨小节线写成连音线延续。
  Shuffle：三连音格。某拍有 +1/3 起音时整拍写成三连音（{tu 3}，记谱 = 演奏）；其余拍写成摇摆八分对
  （+2/3 记在反拍八分 +1/2，alphaTab 照样在 +2/3 演奏；落在 +1/3 的音尾延到反拍八分，即一个摇摆八分）。
  直拍：十六分格；某拍有 +1/3 / +2/3 起音（overrides.melody 里写的三连音）时整拍写成三连音。
- 指法：fretting.assign_frets 的 Viterbi（只有旋律；1–4 弦不加罚、低把位优先）。自检报换把（shift）时
  给该音（不行再给前一个按弦音）换一个同音高的位置：只换弦和品，不改音高。
- 谱面：五线谱 + 六线谱、调号、力度 f；每小节第 1 拍标和弦指型（finger.shape_marks，附和弦图），
  小节中间换指型再标；变调夹与指弹相同（arrange 按同样的旋律音域选）。段落名写成段落第一拍的文字（txt），
  不用 \\section：alphaTab 把段落标记和同一拍的和弦名排进同一行、叠在一起（EffectBandSlot.canBeUsed 允许
  "上一个效果的最后一拍 == 这个效果的第一拍" 共用一行）；指弹把那里的指型挂到低音声部，单声部没有别处可挂，
  而文字效果总是自己占一行。源码里 `// bar N [段落]` 的注释照旧。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .. import theory
from ..config import Profile
from ..verify.playability import Violation
from . import atex
from .capo import OctavePolicy, effective_profile, octave_plan
from .finger import _at, _voice_lines, annotate, section_bars, shape_marks
from .fretting import FretEvent, assign_frets
from .shapes import OPEN_STRINGS

Q = 12               # 内部以 1/12 拍为单位：三连音格（4）与十六分格（3）都是整数
REPAIR_ROUNDS = 10
TOP_STRINGS = 4      # 旋律放在 1–4 弦不加罚
EPS = 1e-6


@dataclass
class Sung:
    """谱上的一个旋律音。perf：演奏位置（alphaTab 播放的时刻）；pos / dur：记谱位置与时值；单位都是拍。
    pitch：指法调音高（八度移位之后）；where：(弦, 品)，None = 弹不了（已丢）。"""
    perf: float
    pos: float
    dur: float
    pitch: int
    where: tuple[int, int] | None = None


def _q(x: float) -> int:
    """拍 → 1/12 拍（四舍五入）。"""
    return int(np.floor(x * Q + 0.5))


def _snap(k: int, step: int) -> int:
    """k（1/12 拍）落到 step 的整数倍：取最近的，正中间取后一个（Shuffle 的 +1/2 归 +2/3）。"""
    return (k + step // 2) // step * step


def triplet_beats(starts: Iterable[int], feel: str) -> set[int]:
    """要整拍写成三连音的拍（0 起）。starts：起音（1/12 拍）。Shuffle（起音已落到三连音格）：有 +1/3 起音的拍；
    直拍：有 +1/3 或 +2/3 起音（不在十六分格上）的拍。"""
    thirds = (4,) if feel == "shuffle" else (4, 8)
    return {k // Q for k in starts if k % Q in thirds}


def _step(k: int, feel: str, triplets: set[int]) -> int:
    """k 所在拍的格（1/12 拍）：Shuffle 与三连音拍是三连音格，直拍其余拍是十六分格。"""
    return 4 if feel == "shuffle" or k // Q in triplets else 3


def _snap_point(k: int, feel: str, triplets: set[int]) -> int:
    """演奏位置的一个边界（起音或音尾，1/12 拍）落到它所在拍的格上（可能落到下一拍的拍点）。"""
    return k // Q * Q + _snap(k % Q, _step(k, feel, triplets))


def _written(k: int, feel: str, triplets: set[int]) -> int:
    """演奏位置 → 记谱位置（1/12 拍）。Shuffle 非三连音拍里的 +1/3、+2/3 记在反拍八分 +1/2（alphaTab 按八分对
    在 +2/3 演奏；+1/3 只会是音尾：延到反拍八分，即一个摇摆八分）；其余照记。"""
    beat, off = divmod(k, Q)
    if feel == "shuffle" and beat not in triplets and off:
        return beat * Q + Q // 2
    return k


def rhythm(notes: Iterable[tuple[float, float, int]], feel: str,
           n_bars: int) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, str]]]:
    """[(起音, 时值, 音高)]（演奏时间，拍）→ 单声部 [(演奏位置, 记谱位置, 记谱时值, 音高)]（1/12 拍），以及丢掉的音
    [(记谱位置, 说明)]：起音落到同一格上的后一个音（只有手写的 overrides.melody 才会这样）。
    起音落格见 triplet_beats；音尾 = 原唱音尾落格，截到下一个音的起音（单声部，不重叠）和末小节线；
    落格后不足一格的补足一格。"""
    end_all = n_bars * 4 * Q
    raw = sorted((_q(s), _q(s + d), p) for s, d, p in notes if 0 <= _q(s) < end_all)
    if feel == "shuffle":   # 先把起音落到三连音格，再看哪些拍有 +1/3 起音
        raw = [(_snap(a, 4), b, p) for a, b, p in raw]
    triplets = triplet_beats([a for a, _, _ in raw], feel)
    kept: list[tuple[int, int, int]] = []
    dropped: list[tuple[int, str]] = []
    for a, b, p in raw:
        a = _snap_point(a, feel, triplets)
        if a >= end_all:
            continue
        if kept and kept[-1][0] == a:
            dropped.append((_written(a, feel, triplets), f"drop {theory.midi_name(p)}"))
            continue
        kept.append((a, b, p))
    out = []
    for i, (a, b, p) in enumerate(kept):
        nxt = kept[i + 1][0] if i + 1 < len(kept) else end_all
        end = min(_snap_point(b, feel, triplets), nxt)
        if end <= a:
            end = min(a + _step(a, feel, triplets), nxt)
        pos = _written(a, feel, triplets)
        out.append((a, pos, _written(end, feel, triplets) - pos, p))
    return out, dropped


def melody_line(leadsheet: dict, capo: int,
                octave: OctavePolicy = "song") -> tuple[list[Sung], list[tuple[float, str]]]:
    """leadsheet 的全部旋律音 → (谱上的旋律音 [Sung]（未分配指法）, 丢掉的音 [(记谱位置, 说明)])。
    八度移位见 capo.octave_plan（默认整曲一个移位，与指弹 finger.melody_line 同一策略）。"""
    notes = [(m["start_beats"], m["dur_beats"], m["midi"]) for b in leadsheet["bars"] for m in b["melody"]]
    timed, dropped = rhythm(notes, leadsheet.get("feel", "straight"), len(leadsheet["bars"]))
    line = [Sung(a / Q, pos / Q, dur / Q, p - capo) for a, pos, dur, p in timed]
    for idx, shift, _ in octave_plan([(n.pos, n.pitch) for n in line], section_bars(leadsheet), octave):
        for i in idx:
            line[i].pitch += shift
    return line, [(pos / Q, text) for pos, text in dropped]


def _pitch(w: tuple[int, int]) -> int:
    return OPEN_STRINGS[w[0] - 1] + w[1]


def positions(pitch: int, max_fret: int) -> list[tuple[int, int]]:
    """能弹出这个音高的全部 (弦, 品)。"""
    return [(s, pitch - OPEN_STRINGS[s - 1]) for s in range(1, 7) if 0 <= pitch - OPEN_STRINGS[s - 1] <= max_fret]


def _fretted(n: Sung) -> bool:
    return n.where is not None and n.where[1] > 0


def shift_breaks(line: list[Sung], max_shift: int) -> list[int]:
    """与校验的换把规则同一模型：按弦音与上一个按弦音（演奏时间）相隔不到 1 拍、品位差超过 max_shift → 该音的下标。
    单声部的音互不重叠，手上只有当前这个音；空弦不占把位（上一个把位跨过它保留）。"""
    out, last = [], None
    for i, n in enumerate(line):
        if not _fretted(n):
            continue
        if last is not None and n.perf - last[0] < 1 - EPS and abs(n.where[1] - last[1]) > max_shift:
            out.append(i)
        last = (n.perf, n.where[1])
    return out


def _reach(line: list[Sung], j: int) -> int:
    """音 j 离前后 1 拍以内的按弦音有几品（j 是空弦时 0）。"""
    if not _fretted(line[j]):
        return 0
    near = [n for n in line[:j][-1:] + line[j + 1:j + 2] if _fretted(n) and abs(n.perf - line[j].perf) < 1 - EPS]
    return sum(abs(n.where[1] - line[j].where[1]) for n in near)


def refret(line: list[Sung], i: int, profile: Profile,
           fixed: set[int] = frozenset()) -> tuple[int, tuple[int, int], tuple[int, int]] | None:
    """换把违规的音 i：给它（不行再给前一个按弦音）换一个同音高的位置（就地修改）。依次比较：全曲换把违规数、
    先换 i、离前后按弦音的品数、1–4 弦、品位低。模型里 i 也违规、却没有哪个位置能减少违规时不换（免得白换）。
    fixed：已经换过的音（每个音最多换一次）。返回 (换了的音, 旧位置, 新位置)；没换时 None。"""
    before = shift_breaks(line, profile.max_shift)
    prev = next((j for j in range(i - 1, -1, -1) if _fretted(line[j])), None)
    best = None
    for rank, j in enumerate(k for k in (i, prev) if k is not None and k not in fixed):
        old = line[j].where
        for w in positions(line[j].pitch, profile.max_fret):
            if w == old:
                continue
            line[j].where = w
            key = (len(shift_breaks(line, profile.max_shift)), rank, _reach(line, j), max(0, w[0] - TOP_STRINGS), w[1])
            line[j].where = old
            if best is None or key < best[0]:
                best = (key, j, w)
    if best is None or best[0][0] > len(before) or (i in before and best[0][0] == len(before)):
        return None
    _, j, w = best
    old, line[j].where = line[j].where, w
    return j, old, w


def repair(line: list[Sung], violations: Iterable[Violation], profile: Profile,
           fixed: set[int]) -> list[tuple[float, str]]:
    """按自检的换把违规（Violation.start 是演奏时间）给对应的音换位置（refret）。返回 (记谱位置, 说明)；空表示无可修。"""
    done = []
    for v in violations:
        if v.rule != "shift":
            continue
        i = next((k for k, n in enumerate(line) if n.where is not None and abs(n.perf - v.start) < 1e-3), None)
        if i is None or i in fixed:
            continue
        found = refret(line, i, profile, fixed)
        if found:
            j, old, new = found
            fixed.add(j)
            done.append((line[j].pos, f"refret {_at(line[j].pos)} {old[1]}.{old[0]}→{new[1]}.{new[0]} (shift)"))
    return done


def split_at(pieces: list[tuple[float, float, list, bool]], cuts: Iterable[float]) -> list[tuple[float, float, list, bool]]:
    """在 cuts（记谱位置）处切开发声中的音，后段写成连音线延续：指型名要挂在从这里开始的一拍上。"""
    cuts = sorted(cuts)
    out = []
    for pos, dur, notes, tie in pieces:
        edges = [pos, *(c for c in cuts if notes and pos + EPS < c < pos + dur - EPS), pos + dur]
        out += [(a, b - a, notes, tie or k > 0) for k, (a, b) in enumerate(zip(edges, edges[1:]))]
    return out


def _label_sections(lines: list[str], sections: list[str | None]) -> list[str]:
    """小节注释 `// bar N` → `// bar N [段落]`（与 atex.bar_line 的注释一致；段落标记本身见模块说明）。"""
    return [line.replace(f"// bar {i + 1}\n", f"// bar {i + 1} [{' '.join(str(sec).splitlines())}]\n", 1) if sec else line
            for i, (line, sec) in enumerate(zip(lines, sections))]


def generate_melody(leadsheet: dict, capo: int, profile: Profile, title: str,
                    check: Callable[[str], list[Violation]] | None = None, octave: OctavePolicy = "song") -> str:
    """check：自检（alphaTex → 违规列表，与 verify 同一套可弹性检查）。给出时按换把违规换指法、重新渲染，最多 10 轮。
    octave：旋律八度策略（song.yaml 的 overrides.melody_octave，见 capo.octave_plan）。"""
    profile = effective_profile(profile, capo)
    feel = leadsheet.get("feel", "straight")
    n_bars = len(leadsheet["bars"])
    line, dropped = melody_line(leadsheet, capo, octave)
    events = assign_frets([FretEvent(n.perf, n.pitch, None) for n in line], profile, top_strings=TOP_STRINGS)
    remarks: dict[int, list[str]] = {}

    def remark(found: Iterable[tuple[float, str]]) -> None:
        for t, text in found:
            remarks.setdefault(int(t // 4), []).append(text)

    for n, e in zip(line, events):
        n.where = e.melody_pos
        if n.where is not None:
            n.pitch = _pitch(n.where)   # 降级（melody±12）后的实际指法调音高
    remark(dropped)
    remark((n.pos, f"{e.fallback} {_at(n.pos)}") for n, e in zip(line, events) if e.fallback)
    sections = [b.get("section") for b in leadsheet["bars"]]
    marks, diagrams = shape_marks(leadsheet, capo, profile)
    head = atex.header(title, leadsheet["tempo_bpm"], capo, feel, "score tabs", diagrams,
                       key=atex.leadsheet_key(leadsheet))
    starts = {i * 4.0: name for i, name in enumerate(sections) if name}
    effects = {pos: ([f"ch {atex.quote(marks[pos])}"] if pos in marks else [])
               + ([f"txt {atex.quote(starts[pos])}"] if pos in starts else []) + (["dy f"] if pos == 0.0 else [])
               for pos in sorted({0.0} | set(marks) | set(starts))}

    def render() -> str:
        pieces = split_at([(n.pos, n.dur, [n.where] if n.where else [], False) for n in line], effects)
        lines = _voice_lines(n_bars, pieces, [None] * n_bars, feel, effects, tie_effects=True)
        return atex.document(head, [annotate(_label_sections(lines, sections), remarks)])

    text = render()
    fixed: set[int] = set()
    for _ in range(REPAIR_ROUNDS if check else 0):
        found = check(text)
        if not found:
            break
        done = repair(line, found, profile, fixed)
        if not done:
            break
        remark(done)
        text = render()
    return text
