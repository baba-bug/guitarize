"""alphaTex 文本生成：时值拆分、拍子记号、Shuffle 八分对切分、文件头、小节行、多声部文档。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

# (拍数, alphaTex 时值码, 是否附点)，从大到小
DURATIONS = [(4.0, 1, False), (3.0, 2, True), (2.0, 2, False), (1.5, 4, True), (1.0, 4, False),
             (0.75, 8, True), (0.5, 8, False), (0.25, 16, False)]
STANDARD_TUNING = "(E4 B3 G3 D3 A2 E2)"


def split_beats(length: float) -> list[tuple[int, bool]]:
    """把时值（拍）贪心拆成 alphaTex 时值码；16 分音符精度表示不了时报错。"""
    out: list[tuple[int, bool]] = []
    rest = length
    while rest > 1e-6:
        for beats, code, dotted in DURATIONS:
            if beats <= rest + 1e-6:
                out.append((code, dotted))
                rest -= beats
                break
        else:
            raise ValueError(f"cannot express {length} beats at 16th-note resolution")
    return out


def notes_token(notes: Sequence[tuple[int, int]], tie: bool = False, dead: bool = False) -> str:
    """[(弦, 品)] → '3.1' / '(3.6 2.5)'；空 → 'r'。tie 写成连音线延续 '-.弦'，dead 写成闷音 'x.弦'。"""
    if not notes:
        return "r"
    parts = [f"{'-' if tie else ('x' if dead else fret)}.{string}" for string, fret in notes]
    return parts[0] if len(parts) == 1 else "(" + " ".join(parts) + ")"


def beat_tokens(notes: Sequence[tuple[int, int]], length: float, effects: Iterable[str] = (),
                dead: bool = False, tie: bool = False) -> list[str]:
    """一个音 / 和弦 / 休止持续 length 拍 → 拍子记号列表。需要多个时值时，后续部分用连音线接上（休止、闷音直接拆分）。
    tie=True 表示整段都是前一个音的延续（第一个记号也写成连音线）。"""
    out = []
    first = list(effects)
    for i, (code, dotted) in enumerate(split_beats(length)):
        eff = (first if i == 0 else []) + (["d"] if dotted else [])
        tok = notes_token(notes, tie=(tie or i > 0) and bool(notes) and not dead, dead=dead)
        out.append(f"{tok}.{code}" + (" {" + " ".join(eff) + "}" if eff else ""))
    return out


@dataclass
class Item:
    """一个声部一小节里的一个音 / 和弦 / 休止（小节内记谱位置，拍）。"""
    start: float
    length: float
    notes: list
    effects: Sequence[str] = ()
    dead: bool = False


def split_for_swing(items: Sequence[Item], feel: str) -> list[tuple[Item, bool]]:
    """三连音感觉下 alphaTab 只把"拍点八分 + 紧接的八分"按 shuffle 演奏。
    对每个发声的反拍起音 p，在 p−0.5、p、p+0.5 处切分覆盖这些位置的音符/休止，使其成为八分对的第二个八分。
    返回 (片段, 是否为前一片段的延续)；直拍原样返回。"""
    if feel != "shuffle":
        return [(it, False) for it in items]
    cuts: set[float] = set()
    for it in items:
        if it.notes and abs(it.start % 1 - 0.5) < 1e-6:
            cuts |= {it.start - 0.5, it.start, it.start + 0.5}
    out: list[tuple[Item, bool]] = []
    for it in items:
        inner = sorted(c for c in cuts if it.start + 1e-6 < c < it.start + it.length - 1e-6)
        edges = [it.start, *inner, it.start + it.length]
        for k, (a, b) in enumerate(zip(edges, edges[1:])):
            out.append((Item(a, b - a, it.notes, list(it.effects) if k == 0 else [], it.dead), k > 0))
    return out


def voice_bar_tokens(items: Sequence[Item], feel: str) -> list[str]:
    """一个声部一小节（items 须首尾相接铺满小节）→ 拍子记号；延续片段写成连音线，闷音 / 休止的延续写成休止。"""
    tokens: list[str] = []
    for piece, cont in split_for_swing(items, feel):
        if cont and (piece.dead or not piece.notes):
            tokens += beat_tokens([], piece.length)
        else:
            tokens += beat_tokens(piece.notes, piece.length, piece.effects, dead=piece.dead, tie=cont)
    return tokens


def header(title: str, tempo: float, capo: int, feel: str, staff: str,
           chords: dict[str, Sequence[int | None]] | None = None) -> str:
    lines = [f'\\title "{title.replace(chr(34), chr(39))}"', f"\\tempo {int(round(tempo))}", '\\track "Guitar"',
             "\\instrument 25", f"\\staff {{{staff}}}", f"\\tuning {STANDARD_TUNING}"]
    if capo:
        lines.append(f"\\capo {capo}")
    for name, frets in (chords or {}).items():
        lines.append(f'\\chord "{name}" ' + " ".join("x" if f is None else str(f) for f in frets))
    lines.append("\\ts 4 4")
    if feel == "shuffle":
        lines.append("\\tf triplet-8th")
    return "\n".join(lines) + "\n"


def bar_line(tokens: Sequence[str], number: int, section: str | None = None) -> str:
    label = f"// bar {number}" + (f" [{section}]" if section else "")
    prefix = f'\\section "{section}" ' if section else ""
    return f"{label}\n{prefix}{' '.join(tokens)} |"


def document(head: str, voices: Sequence[Sequence[str]]) -> str:
    if len(voices) == 1:
        return head + "\n".join(voices[0]) + "\n"
    return head + "".join("\\voice\n" + "\n".join(v) + "\n" for v in voices)
