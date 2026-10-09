"""文本六线谱：每行若干小节，带小节号、段落、和弦名、扫弦方向。"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import theory
from ..verify.model import Bar, Score

CHARS_PER_BEAT = 4


@dataclass
class _Col:
    start: float
    frets: dict[int, str] = field(default_factory=dict)
    chord: str = ""
    strum: str = ""


def _columns(score: Score, bar: Bar) -> list[_Col]:
    cols: dict[float, _Col] = {}
    for e in score.events:
        if e.bar != bar.index:
            continue
        c = cols.setdefault(round(e.written_start, 6), _Col(e.written_start))
        for n in e.notes:
            if not n.tie:
                c.frets[n.string] = "x" if n.dead else str(n.fret)
        if e.chord:
            c.chord = e.chord
        direction = e.pick or e.brush
        if direction:
            c.strum = "↓" if direction == "down" else "↑"
    return [cols[k] for k in sorted(cols)]


def _bar_block(score: Score, bar: Bar) -> dict[str, str]:
    n = len(score.tuning)
    rows = {"chord": "", "strum": "", **{f"s{s}": "" for s in range(1, n + 1)}}
    cols = _columns(score, bar)
    if not cols:
        width = round(bar.length * CHARS_PER_BEAT)
        return {k: ("-" * width if k.startswith("s") else " " * width) for k in rows}
    end = bar.start + bar.length
    for i, c in enumerate(cols):
        nxt = cols[i + 1].start if i + 1 < len(cols) else end
        width = max(
            round((nxt - c.start) * CHARS_PER_BEAT),
            max((len(t) for t in c.frets.values()), default=1) + 1,
            len(c.chord) + 1,
            2,
        )
        rows["chord"] += c.chord.ljust(width)
        rows["strum"] += c.strum.ljust(width)
        for s in range(1, n + 1):
            rows[f"s{s}"] += c.frets.get(s, "").ljust(width, "-")
    return rows


def render_ascii(score: Score, bars_per_line: int = 4) -> str:
    n = len(score.tuning)
    names = [theory.pc_name(m) for m in score.tuning]  # tuning[0] = 1 弦
    names[0] = names[0].lower()
    label_w = max(len(x) for x in names)
    info = f"♩ = {score.tempo:g}"
    if score.capo:
        info += f" · Capo {score.capo}"
    info += " · 调弦 " + " ".join(reversed([theory.pc_name(m) for m in score.tuning]))
    out = [score.title, info]
    if any(b.feel == "triplet8th" for b in score.bars):
        out.append("Shuffle：八分音符按三连音感觉弹（♪♪ = ♩♪³）")
    out.append("")
    pad = " " * (label_w + 1)
    for i in range(0, len(score.bars), bars_per_line):
        group = score.bars[i:i + bars_per_line]
        top, chords, strum = pad, pad, pad
        strings = [names[s - 1].rjust(label_w) + "|" for s in range(1, n + 1)]
        for bar in group:
            blk = _bar_block(score, bar)
            width = len(blk["s1"])
            label = f"{bar.index + 1}" + (f" [{bar.section}]" if bar.section else "")
            top += label.ljust(width + 1)
            chords += blk["chord"] + " "
            strum += blk["strum"] + " "
            for s in range(1, n + 1):
                strings[s - 1] += blk[f"s{s}"] + "|"
        out.append(top.rstrip())
        if chords.strip():
            out.append(chords.rstrip())
        out += strings
        if strum.strip():
            out.append(strum.rstrip())
        out.append("")
    return "\n".join(out)
