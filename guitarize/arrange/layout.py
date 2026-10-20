"""谱面排版（指弹、旋律谱、卡祖笛谱共用）：声部的音铺成小节行（空隙补休止、跨小节线写连音线）、每小节的和弦指型标记、
单声部谱的标记拍效果、小节注释里的位置写法与降级 / 修补说明。"""
from __future__ import annotations

from typing import Sequence

from .. import theory
from ..config import Profile
from . import atex
from .shapes import voicings
from .strum import beat_chords

EPS = 1e-6


def section_bars(leadsheet: dict) -> list[int]:
    """带段落标记的小节（0-based）。"""
    return [i for i, b in enumerate(leadsheet["bars"]) if b.get("section")]


def _split_at_bars(items: list[tuple]) -> list[tuple]:
    """跨小节线的音在小节线处切开，后半段写成连音线延续（有音时）或休止。"""
    out = []
    for pos, dur, notes, tie in items:
        while dur > EPS:
            bar_end = (int(round(pos, 6) // 4) + 1) * 4.0
            part = min(dur, bar_end - pos)
            out.append((pos, part, notes, tie))
            pos, dur, tie = bar_end, dur - part, bool(notes)
    return out


def _voice_lines(n_bars: int, items: list[tuple], sections: list[str | None],
                 feel: str, effects: dict[float, list[str]] | None = None, tie_effects: bool = False,
                 note_effects: dict[float, dict[int, list[str]]] | None = None,
                 head_effects: dict[float, dict[int, list[str]]] | None = None) -> list[str]:
    """items：(记谱位置, 时值, [(弦, 品)], 是否连音线延续)（见 Placed.pieces；连音线也可以是接着响的那几根弦，
    frozenset，见 finger.merged_pieces）；按小节铺满（空隙补休止），
    Shuffle 下按八分对切分。effects：{记谱位置: 拍效果}（和弦指型、力度），挂在该位置起音的音 / 休止上
    （休止在这些位置切开，保证有一拍从这里开始）；tie_effects 时也挂在从该位置开始的连音线延续上
    （单声部的旋律谱：小节以连音线开头时指型名没有别的地方挂）。
    note_effects：{记谱音尾: {弦: 音符效果}}（击勾弦、滑音，见 atex 模块说明），挂在到这里结束的那一段上：
    切开的音（跨小节线、换指型处）只挂在最后一段。
    head_effects：{记谱起音: {弦（音高谱表：音高）: 音符效果}}（连线 slur 的起止），挂在从这里起音的那一段上
    （连音线延续不挂）。items 里的音也可以是 atex.Pitch（音高谱表，见 atex 模块说明）。"""
    marks = {round(k, 6): v for k, v in (effects or {}).items()}
    tails = {round(k, 6): v for k, v in (note_effects or {}).items()}
    heads = {round(k, 6): v for k, v in (head_effects or {}).items()}
    by_bar: dict[int, list] = {}
    for item in sorted(_split_at_bars(items), key=lambda x: x[0]):
        by_bar.setdefault(int(round(item[0], 6) // 4), []).append(item)
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
            held = tie if isinstance(tie, frozenset) else frozenset()   # 同一拍里只有这些弦接着响
            tie = tie is True
            mark = marks.get(round(pos, 6), ()) if tie_effects or not tie else ()
            bar_items.append(atex.Item(pos - i * 4.0, dur, notes, mark, tie=tie, held=held,
                                       note_effects=tails.get(round(pos + dur, 6)) if notes else None,
                                       head_effects=heads.get(round(pos, 6)) if notes and not tie else None))
            t = pos + dur
        if t < end - 1e-6:
            rests(t, end)
        lines.append(atex.bar_line(atex.voice_bar_tokens(bar_items, feel), i + 1, sections[i]))
    return lines


def mark_effects(marks: dict[float, str], sections: Sequence[str | None]) -> dict[float, list[str]]:
    """单声部谱（旋律谱、卡祖笛谱）每个标记位置的拍效果 → {记谱位置: 拍效果}（_voice_lines 的 effects）：
    和弦名 / 指型名 ch、段落名 txt（写在段落第一拍，不用 \\section，见 melody 模块说明）、第一拍的力度 dy f。"""
    starts = {i * 4.0: name for i, name in enumerate(sections) if name}
    return {pos: ([f"ch {atex.quote(marks[pos])}"] if pos in marks else [])
            + ([f"txt {atex.quote(starts[pos])}"] if pos in starts else []) + (["dy f"] if pos == 0.0 else [])
            for pos in sorted({0.0} | set(marks) | set(starts))}


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


def _at(t: float) -> str:
    """修补说明里的位置：小节内第几拍（1 起），八分格写 @2.5，三连音位置写 @4+⅓ / @4+⅔。"""
    x = round(t % 4, 6)
    beat = int(x)
    for k, mark in ((1, "+⅓"), (2, "+⅔")):
        if abs(x - beat - k / 3) < 1e-4:
            return f"@{beat + 1}{mark}"
    return f"@{x + 1:g}"


def annotate(lines: list[str], remarks: dict[int, list[str]]) -> list[str]:
    """把降级 / 修补说明接到对应小节（0-based）的 `// bar N ...` 注释行末尾。"""
    out = []
    for i, line in enumerate(lines):
        if remarks.get(i):
            head, body = line.split("\n", 1)
            line = head + "".join(f" · {r}" for r in remarks[i]) + "\n" + body
        out.append(line)
    return out
