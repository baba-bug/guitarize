"""alphaTex 文本生成：时值拆分、拍子记号、三连音拍、Shuffle 八分对切分、文件头、小节行、多声部文档。

音符效果（alphaTab 1.8.4 已验证）：紧跟在 品.弦 后面、时值前面，'5.3{h}.4'；和弦里跟在各自的音后面，
'(5.3{h} 3.2).4'；连音线延续同样能挂，'-.3{h}.4'。拍效果照旧写在时值后面（'5.3{h}.4 {d tu 3}'）；
'5.3.4{h}' 会被当成拍效果，报 Unrecognized property。h = 击弦 / 勾弦、sl = 连奏滑音（连到同弦的下一个音，
alphaTab 不为下一个音重新起音）、sod / sou = 音尾向下 / 向上滑出。这些效果都属于音尾：一个音被拆成几个记号
（跨小节线、三连音拍线、八分对切分、时值拆分）时只挂在最后一个（紧挨着下一个音的那一段）上。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Collection, Iterable, Mapping, Sequence

from .. import theory

# (拍数, alphaTex 时值码, 是否附点)，从大到小
DURATIONS = [(4.0, 1, False), (3.0, 2, True), (2.0, 2, False), (1.5, 4, True), (1.0, 4, False),
             (0.75, 8, True), (0.5, 8, False), (0.25, 16, False)]
STANDARD_TUNING = "(E4 B3 G3 D3 A2 E2)"
TUPLET = "tu 3"   # 三连音：alphaTab 1.8.4 的拍效果 {tu 3}（.8 = 1/3 拍、.4 = 2/3 拍，起音正好在 k/3）


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


def split_triplet(length: float) -> list[tuple[int, bool]]:
    """三连音拍里的一段（1/3 的倍数，至多 1 拍）→ [(时值码, 是否三连音)]：1/3 = 三连音八分、2/3 = 三连音四分、
    整拍 = 普通四分。"""
    thirds = round(length * 3)
    if abs(length * 3 - thirds) > 1e-4 or not 1 <= thirds <= 3:
        raise ValueError(f"cannot express {length} beats inside a triplet beat")
    return [{1: (8, True), 2: (4, True), 3: (4, False)}[thirds]]


NoteEffects = Mapping[int, Sequence[str]]   # {弦: 音符效果}，如 {3: ["h"]}


def notes_token(notes: Sequence[tuple[int, int]], tie: bool = False, dead: bool = False,
                effects: NoteEffects | None = None, held: Collection[int] = ()) -> str:
    """[(弦, 品)] → '3.1' / '(3.6 2.5)'；空 → 'r'。tie 写成连音线延续 '-.弦'，dead 写成闷音 'x.弦'。
    held：只有这些弦上的音是连音线延续（同一拍里别的音照常起音，'(0.2 0.3 -.6)'：低音接着响、新拨两个合音）。
    effects：{弦: 音符效果}，写在该弦的音后面（'5.3{h}'、'(5.3{h} 3.2)'），见模块说明。"""
    if not notes:
        return "r"
    effects = effects or {}
    parts = [f"{'-' if tie or string in held else ('x' if dead else fret)}.{string}"
             + ("{" + " ".join(effects[string]) + "}" if effects.get(string) else "") for string, fret in notes]
    return parts[0] if len(parts) == 1 else "(" + " ".join(parts) + ")"


def beat_tokens(notes: Sequence[tuple[int, int]], length: float, effects: Iterable[str] = (),
                dead: bool = False, tie: bool = False, tuplet: bool = False,
                note_effects: NoteEffects | None = None, held: Collection[int] = ()) -> list[str]:
    """一个音 / 和弦 / 休止持续 length 拍 → 拍子记号列表。需要多个时值时，后续部分用连音线接上（休止、闷音直接拆分）。
    tie=True 表示整段都是前一个音的延续（第一个记号也写成连音线）；held 只让这些弦的音在第一个记号里写成连音线。
    tuplet=True：这一段在三连音拍里（见 split_triplet），记号带 {tu 3}。
    effects 是拍效果，挂在第一个记号上；note_effects（{弦: 音符效果}）属于音尾，挂在最后一个记号上。"""
    out = []
    first = list(effects)
    parts = ([(code, False, tu) for code, tu in split_triplet(length)] if tuplet
             else [(code, dotted, False) for code, dotted in split_beats(length)])
    for i, (code, dotted, tu) in enumerate(parts):
        eff = (first if i == 0 else []) + (["d"] if dotted else []) + ([TUPLET] if tu else [])
        tok = notes_token(notes, tie=(tie or i > 0) and bool(notes) and not dead, dead=dead,
                          effects=note_effects if i == len(parts) - 1 else None, held=held)
        out.append(f"{tok}.{code}" + (" {" + " ".join(eff) + "}" if eff else ""))
    return out


@dataclass
class Item:
    """一个声部一小节里的一个音 / 和弦 / 休止（小节内记谱位置，拍）。
    tie=True：没有新的起音，notes 是前面某个和弦里接着响的那几个音（连音线延续，可以只是和弦的一部分）。
    held：tie=False 时，notes 里这些弦上的音是接着响的（连音线延续），其余照常起音；切分后的后续片段整个是连音线。
    tuplet=True：在三连音拍里（时值 1/3 或 2/3 拍，写成 {tu 3}），见 split_triplets。
    effects：拍效果，切分时留在第一段；note_effects：{弦: 音符效果}（音尾），切分时只留在最后一段。"""
    start: float
    length: float
    notes: list
    effects: Sequence[str] = ()
    dead: bool = False
    tie: bool = False
    tuplet: bool = False
    note_effects: NoteEffects | None = None
    held: frozenset = frozenset()


def _on_third(x: float) -> bool:
    """x 落在某拍的 +1/3 或 +2/3 上。"""
    frac = x % 1
    return abs(frac - 1 / 3) < 1e-6 or abs(frac - 2 / 3) < 1e-6


def triplet_beats(items: Sequence[Item]) -> set[int]:
    """三连音拍：有音 / 休止的边界（起点或终点）落在 +1/3、+2/3 上的拍（小节内第几拍，0 起）。"""
    return {int(x) for it in items for x in (it.start, it.start + it.length) if _on_third(x)}


def split_triplets(items: Sequence[Item]) -> list[Item]:
    """在三连音拍的拍线处切开跨拍的音 / 休止，三连音拍里的片段标 tuplet。切出的后续片段没有新起音：
    有音的写成连音线延续，闷音 / 休止写成休止。拍效果留在第一段、音符效果（音尾）只留在最后一段。
    没有三连音拍时原样返回。"""
    beats = triplet_beats(items)
    if not beats:
        return list(items)
    lines = sorted({float(b) for b in beats} | {b + 1.0 for b in beats})
    out: list[Item] = []
    for it in items:
        end = it.start + it.length
        edges = [it.start, *(x for x in lines if it.start + 1e-6 < x < end - 1e-6), end]
        last = len(edges) - 2
        for k, (a, b) in enumerate(zip(edges, edges[1:])):
            tuplet = int(a + 1e-6) in beats
            tail = it.note_effects if k == last else None
            if k == 0:
                out.append(Item(a, b - a, it.notes, it.effects, it.dead, it.tie, tuplet, tail, it.held))
            else:
                notes = [] if it.dead else it.notes
                out.append(Item(a, b - a, notes, (), False, bool(notes), tuplet, tail if notes else None))
    return out


def rest_cuts(start: float, end: float) -> list[float]:
    """Shuffle 下一段休止 [start, end) 的切点：从拍中间开始的，在下一条拍线处切开（本拍余下的部分单独成一个休止，
    拍点八分后面的反拍休止就是八分，两者成对，前面的音才按 shuffle 演奏）；结束在拍中间的，在最后一条拍线处切开。
    中间从拍点开始的整拍照旧合并（r.2、r.1）：除此之外没有休止跨过拍线。"""
    head, tail = math.ceil(start - 1e-6), math.floor(end + 1e-6)
    return sorted({float(x) for x in (head, tail) if start + 1e-6 < x < end - 1e-6})


def split_for_swing(items: Sequence[Item], feel: str) -> list[tuple[Item, bool]]:
    """三连音感觉下 alphaTab 只把"拍点八分 + 紧接的八分"按 shuffle 演奏。
    对每个发声的反拍起音 p（连音线延续不是起音），在 p−0.5、p、p+0.5 处切分覆盖这些位置的音符/休止，
    使其成为八分对的第二个八分；休止另在 rest_cuts 处切开（拍点八分后面跨拍的反拍休止先写成一个八分，与它成对）。
    三连音拍里的片段（tuplet）既不是反拍起音、也不切：alphaTab 只让普通八分成对，
    三连音拍照记谱演奏、也不会与前后的八分凑成一对（切点都在拍线或普通拍里，碰不到三连音拍）。
    拍效果留在第一段、音符效果（音尾）只留在最后一段。返回 (片段, 是否为前一片段的延续)；直拍原样返回。"""
    if feel != "shuffle":
        return [(it, False) for it in items]
    cuts: set[float] = set()
    for it in items:
        if it.notes and not it.tie and not it.tuplet and abs(it.start % 1 - 0.5) < 1e-6:
            cuts |= {it.start - 0.5, it.start, it.start + 0.5}
    out: list[tuple[Item, bool]] = []
    for it in items:
        if it.tuplet:
            out.append((it, False))
            continue
        own = rest_cuts(it.start, it.start + it.length) if not it.notes else []
        inner = sorted(c for c in cuts.union(own) if it.start + 1e-6 < c < it.start + it.length - 1e-6)
        edges = [it.start, *inner, it.start + it.length]
        for k, (a, b) in enumerate(zip(edges, edges[1:])):
            out.append((Item(a, b - a, it.notes, list(it.effects) if k == 0 else [], it.dead, it.tie,
                             note_effects=it.note_effects if k == len(edges) - 2 else None,
                             held=it.held if k == 0 else frozenset()), k > 0))
    return out


def voice_bar_tokens(items: Sequence[Item], feel: str) -> list[str]:
    """一个声部一小节（items 须首尾相接铺满小节）→ 拍子记号；延续片段与 tie 项写成连音线，闷音 / 休止的延续写成休止。
    有边界落在 +1/3、+2/3 的拍整拍写成三连音（split_triplets），其余拍按 feel 处理。"""
    tokens: list[str] = []
    for piece, cont in split_for_swing(split_triplets(items), feel):
        if cont and (piece.dead or not piece.notes):
            tokens += beat_tokens([], piece.length)
        else:
            tokens += beat_tokens(piece.notes, piece.length, piece.effects, dead=piece.dead, tie=cont or piece.tie,
                                  tuplet=piece.tuplet, note_effects=piece.note_effects, held=piece.held)
    return tokens


def _one_line(text: str) -> str:
    """换行 / 制表符换成空格：名字只占一行（小节注释和字符串都不会被截断）。"""
    return " ".join(str(text).replace("\t", " ").splitlines())


def quote(text: str) -> str:
    """alphaTex 字符串字面量：反斜杠写成两个、双引号前加反斜杠（alphaTab 1.8.4 已验证这两种转义）。"""
    return '"' + _one_line(text).replace("\\", "\\\\").replace('"', '\\"') + '"'


# 调号名：按惯用写法（五度圈里升降号更少的一边）；alphaTab 不认 A#、D# 大调这类名字
_MAJOR_KS = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
_MINOR_KS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B")


def key_signature(tonic: str, mode: str) -> str:
    """原调（实际音高，五线谱按实际音高记谱，与变调夹无关）→ alphaTex \\ks 的调名。"""
    pc = theory.pc(tonic)
    return _MINOR_KS[pc] + "minor" if mode == "minor" else _MAJOR_KS[pc]


def leadsheet_key(leadsheet: dict) -> tuple[str, str] | None:
    """leadsheet 的原调 (主音, 大 / 小调)；没有记调时不写调号。"""
    return (leadsheet["key"], leadsheet.get("mode", "major")) if leadsheet.get("key") else None


def header(title: str, tempo: float, capo: int, feel: str, staff: str,
           chords: dict[str, Sequence[int | None]] | None = None, key: tuple[str, str] | None = None) -> str:
    lines = [f"\\title {quote(title)}", f"\\tempo {int(round(tempo))}", '\\track "Guitar"',
             "\\instrument 25", f"\\staff {{{staff}}}", f"\\tuning {STANDARD_TUNING}"]
    if capo:
        lines.append(f"\\capo {capo}")
    for name, frets in (chords or {}).items():
        lines.append(f"\\chord {quote(name)} " + " ".join("x" if f is None else str(f) for f in frets))
    if chords:
        lines.append("\\chordDiagramsInScore")   # 每处指型标记画成和弦图（谱首仍有总表）
    lines.append("\\ts 4 4")
    if key is not None:
        lines.append(f"\\ks {key_signature(*key)}")
    if feel == "shuffle":
        lines.append("\\tf triplet-8th")
    return "\n".join(lines) + "\n"


def bar_line(tokens: Sequence[str], number: int, section: str | None = None) -> str:
    label = f"// bar {number}" + (f" [{_one_line(section)}]" if section else "")
    prefix = f"\\section {quote(section)} " if section else ""
    return f"{label}\n{prefix}{' '.join(tokens)} |"


def document(head: str, voices: Sequence[Sequence[str]]) -> str:
    if len(voices) == 1:
        return head + "\n".join(voices[0]) + "\n"
    return head + "".join("\\voice\n" + "\n".join(v) + "\n" for v in voices)
