"""指弹编配草稿：旋律骨架（voice 1）+ 低音与填充（voice 2），指法由 Viterbi 分配。
给出自检（与 verify 同一套可弹性检查）时按违规逐轮修补：截短仍按着的音、去掉填充 / 低音。
每处指法降级与修补都写进该小节的 `// bar N` 注释，方便审阅。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .. import theory
from ..config import Profile
from ..skeleton import skeleton_mask
from ..verify.playability import Violation
from . import atex
from .capo import effective_profile, octave_plan
from .fretting import FretEvent, assign_frets
from .shapes import OPEN_STRINGS
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


def melody_line(leadsheet: dict, capo: int, profile: Profile) -> list[tuple[float, int]]:
    """骨架旋律 → (记谱位置, 指法调音高)；按段落（无标记时按乐句）分组各自选八度（见 capo.octave_plan）。"""
    mel = [(pos, p - capo) for pos, p in skeleton_notes(leadsheet, profile)]
    out = list(mel)
    for idx, shift, _ in octave_plan(mel, section_bars(leadsheet)):
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


def _voice_lines(n_bars: int, items: list[tuple[float, float, list]], sections: list[str | None],
                 feel: str) -> list[str]:
    """items：(记谱位置, 时值, [(弦, 品) 或 None, ...])；按小节铺满（空隙补休止），Shuffle 下按八分对切分。"""
    by_bar: dict[int, list] = {}
    for pos, dur, where in sorted(items, key=lambda x: x[0]):
        by_bar.setdefault(int(pos // 4), []).append((pos, dur, where))
    lines = []
    for i in range(n_bars):
        t, end, bar_items = i * 4.0, i * 4.0 + 4.0, []
        for pos, dur, where in by_bar.get(i, []):
            if pos > t + 1e-6:
                bar_items.append(atex.Item(t - i * 4.0, pos - t, []))
            dur = min(dur, end - pos)
            bar_items.append(atex.Item(pos - i * 4.0, dur, [w for w in where if w is not None]))
            t = pos + dur
        if t < end - 1e-6:
            bar_items.append(atex.Item(t - i * 4.0, end - t, []))
        lines.append(atex.bar_line(atex.voice_bar_tokens(bar_items, feel), i + 1, sections[i]))
    return lines


@dataclass
class Placed:
    """声部里的一个音 / 和弦（记谱位置、时值，拍）。where：voice 1 为 [旋律]，voice 2 为 [填充, 低音]；None = 不弹。"""
    pos: float
    dur: float
    where: list

    def holds(self, t: float) -> bool:
        """起音早于 t、t 时仍在发声。"""
        return self.pos < t - EPS and t < self.pos + self.dur - EPS


def _at(t: float) -> str:
    return f"@{t % 4 + 1:g}"   # 小节内第几拍（1 起）


def _strings(it: Placed) -> set[int]:
    return {w[0] for w in it.where if w is not None}


def cut_same_string(v1: list[Placed], v2: list[Placed]) -> list[tuple[float, str]]:
    """同一根弦不能一边按住一边再弹：低音 / 填充持续期间有旋律音弹在它的弦上 → 截止到这个旋律音；
    旋律音持续期间有低音 / 填充弹在它的弦上 → 旋律截止到该处。返回 (记谱位置, 说明)。"""
    out = []
    for held, other, name in ((v2, v1, "bass cut"), (v1, v2, "melody cut")):
        for it in held:
            hit = min((o.pos for o in other if it.holds(o.pos) and _strings(it) & _strings(o)), default=None)
            if hit is not None:
                it.dur = hit - it.pos
                out.append((hit, f"{name} {_at(hit)} (same string)"))
    return out


def _pitch(w: tuple[int, int]) -> int:
    return OPEN_STRINGS[w[0] - 1] + w[1]


def cut_crossings(v1: list[Placed], v2: list[Placed]) -> list[tuple[float, str]]:
    """填充本来按整段持续期间的最低旋律音放低至少 3 个半音；但指法降级（如 melody-12）会把后面的旋律压下来。
    低音 / 填充持续期间有旋律音不比它高出 3 个半音 → 截止到这个旋律音。返回 (记谱位置, 说明)。"""
    out = []
    for it in v2:
        top = max((_pitch(w) for w in it.where if w is not None), default=None)
        if top is None:
            continue
        hit = min((o.pos for o in v1 if it.holds(o.pos) and o.where[0] is not None
                   and _pitch(o.where[0]) - top < FILL_GAP), default=None)
        if hit is not None:
            it.dur = hit - it.pos
            out.append((hit, f"bass cut {_at(hit)} (crossing)"))
    return out


def _fretted(w: tuple[int, int] | None) -> bool:
    return w is not None and w[1] > 0


def _repair_at(v1: list[Placed], v2: list[Placed], t: float) -> str | None:
    for it in v2:
        if it.holds(t) and any(_fretted(w) for w in it.where):
            it.dur = t - it.pos
            return "bass cut"
    for it in v2:
        if abs(it.pos - t) < EPS:
            for k, name in ((0, "drop fill"), (1, "drop bass")):
                if _fretted(it.where[k]):
                    it.where[k] = None
                    return name
    for it in v1:
        if it.holds(t) and _fretted(it.where[0]):
            it.dur = t - it.pos
            return "melody cut"
    return None


def repair(v1: list[Placed], v2: list[Placed], violations: Iterable[Violation], feel: str) -> list[tuple[float, str]]:
    """按自检违规修补（就地修改 v1 / v2）。Violation.start 是演奏时间，先换回记谱位置 t。
    每个 t 每轮只修一处，依次尝试：1. 截短 t 之前起音、t 时仍按着的低音 / 填充；2. 去掉 t 时起音的填充，没有则去掉低音；
    3. 截短 t 时仍按着的旋律音。只动按弦音（空弦不占手型），只修手型类规则。返回 (t, 说明)；空表示无可修。"""
    done, seen = [], set()
    for v in violations:
        t = written_position(v.start, feel)
        if v.rule not in HAND_RULES or t in seen:
            continue
        seen.add(t)
        what = _repair_at(v1, v2, t)
        if what:
            done.append((t, f"{what} {_at(t)} ({v.rule})"))
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
                    check: Callable[[str], list[Violation]] | None = None) -> str:
    """check：自检（alphaTex → 违规列表）。给出时渲染后按违规修补、重新渲染，最多 10 轮。"""
    profile = effective_profile(profile, capo)
    feel = leadsheet.get("feel", "straight")
    n_bars = len(leadsheet["bars"])
    mel = melody_line(leadsheet, capo, profile)
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
    head = atex.header(title, leadsheet["tempo_bpm"], capo, feel, "score tabs")

    def render() -> str:
        lines1 = _voice_lines(n_bars, [(it.pos, it.dur, it.where) for it in v1], sections, feel)
        lines2 = _voice_lines(n_bars, [(it.pos, it.dur, it.where) for it in v2], [None] * n_bars, feel)
        return atex.document(head, [annotate(lines1, remarks), lines2])

    text = render()
    for _ in range(REPAIR_ROUNDS if check else 0):
        found = check(text)
        if not found:
            break
        done = repair(v1, v2, found, feel)
        if not done:
            break
        remark(done)
        text = render()
    return text
