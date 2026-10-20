"""卡祖笛独奏谱（arr = kazoo）：对着卡祖笛哼唱人声旋律。卡祖笛靠哼唱发声，没有指法，谱上只有音高和节奏。

- 音：leadsheet 的每一个旋律音（用户确认过的旋律，与旋律谱同一份），按实际音高写：不加变调夹、不移八度，
  就是歌手自己唱的那个八度。节奏与旋律谱完全相同（melody.melody_line：起音、时值、三连音、休止、跨小节线的连音线）。
- 吐音：连奏的音（leadsheet 的 legato：同一音节里滑过去）跟着人声滑过去、不重新吐音，写成从前一个音起的连线
  （slur，见 legato_slurs）；其余的音都重新吐音（"du"）。
- 谱面：五线谱 + 简谱（\\staff {score numbered}：中国的卡祖笛学习者多看简谱）、调号、力度 f；音色 GM 59
  Muted Trumpet（KAZOO_PROGRAM：GM 音色里最接近卡祖笛的嗡嗡鼻音）。简谱不加点的八度放在旋律最集中的那个八度
  （display_ottava；男声旋律用 8vb 谱号）。
- 和弦名：每小节第 1 拍标和弦名（实际音高，不画和弦图：给伴奏的吉他手、也让吹的人看到和声），小节中间换和弦再标，
  转入无和弦标 N.C.（chord_marks）。段落名写成段落第一拍的文字（txt），原因与旋律谱相同：\\section 和同一拍的
  和弦名叠在一起（见 melody 模块说明）；音高谱表上 `\\section "名" C#4.4` 还会把音名当成 \\section 的参数。
  源码里 `// bar N [段落]` 的注释照旧。
- 校验（verify --arr kazoo）不查可弹性，只查记谱；报旋律 F1（与旋律谱同一套评法）、音域、换气（连续太久没有休止）。"""
from __future__ import annotations

from typing import Sequence

from . import atex
from .layout import _voice_lines, annotate, mark_effects
from .melody import Sung, _label_sections, melody_line, split_at
from .strum import beat_chords

KAZOO_PROGRAM = 59   # GM 59 Muted Trumpet：GM 音色里最接近卡祖笛的嗡嗡鼻音
TRACK = "Kazoo"
NO_CHORD = "N.C."


def chord_marks(leadsheet: dict) -> dict[float, str]:
    """→ {记谱位置: 和弦名}（实际音高）：每小节第 1 拍都标（和弦没变也标），小节中间换了再标；从有和弦转入
    无和弦（N）标 N.C.，一直没有和弦的小节不重复标。"""
    marks: dict[float, str] = {}
    last = None
    for i, b in enumerate(leadsheet["bars"]):
        for k, label in enumerate(beat_chords(b, leadsheet.get("beats_per_bar", 4))):
            if label == "N":
                if last != "N":
                    marks[i * 4.0 + k] = NO_CHORD
            elif k == 0 or label != last:
                marks[i * 4.0 + k] = label
            last = label
    return marks


def legato_slurs(line: list[Sung]) -> dict[float, dict[int, list[str]]]:
    """连奏的音不重新吐音：从前一个音起画一条连线，连奏的几个音连成一串时一条连线罩住整串。
    → {记谱起音: {音高: ["slur sN"]}}（layout._voice_lines 的 head_effects）：连线挂在头尾两个音的起音上；
    id 各不相同（alphaTab 按 id 配对，配过的 id 不清掉，不能再用）。"""
    runs: list[list[Sung]] = []
    for prev, n in zip(line, line[1:]):
        if not n.legato:
            continue
        if runs and runs[-1][-1] is prev:
            runs[-1].append(n)
        else:
            runs.append([prev, n])
    out: dict[float, dict[int, list[str]]] = {}
    for k, run in enumerate(runs, 1):
        for n in (run[0], run[-1]):
            out[n.pos] = {n.pitch: [f"slur s{k}"]}
    return out


def display_ottava(pitches: Sequence[int], fifths: int, minor: bool) -> str | None:
    """简谱不加点的八度（从主音起）放在装下最多旋律音的那个八度 → 谱号八度记号（atex.OTTAVA_SHIFT 的名字），
    None = 不用。alphaTab 默认从第 4 组的主音起（atex.numbered_one：B 大调 B4），男声唱的旋律大多在它下面、
    满谱下加点；用 8vb 谱号（记谱高八度，男声旋律的惯用写法）把五线谱和简谱的显示一起上移，播放的音高不变。
    装下的音一样多时取移得少的。文本简谱（render/jianpu.py）按同一个谱号定八度。"""
    one = atex.numbered_one(fifths, minor)
    shift = max([0, *atex.OTTAVA_SHIFT.values()],
                key=lambda s: (sum(0 <= p + s - one < 12 for p in pitches), -abs(s)))
    return next((name for name, s in atex.OTTAVA_SHIFT.items() if s == shift), None)


def generate_kazoo(leadsheet: dict, title: str) -> str:
    feel = leadsheet.get("feel", "straight")
    n_bars = len(leadsheet["bars"])
    key = atex.leadsheet_key(leadsheet)
    fifths, minor = (atex.key_fifths(*key), key[1] == "minor") if key else (0, False)
    line, dropped = melody_line(leadsheet, 0, octave=0)   # 实际音高：不加变调夹、不移八度
    head = atex.pitched_header(title, leadsheet["tempo_bpm"], feel, TRACK, KAZOO_PROGRAM, key=key,
                               ottava=display_ottava([n.pitch for n in line], fifths, minor))
    sections = [b.get("section") for b in leadsheet["bars"]]
    effects = mark_effects(chord_marks(leadsheet), sections)
    flat = atex.flats(key)
    pieces = split_at([(n.pos, n.dur, [atex.Pitch(n.pitch, flat)], False) for n in line], effects)
    # alphaTab 1.8.4 的简谱谱表画不了全休止（r.1：NumberedBarRenderer 报 Cannot read properties of null，网页、PDF
    # 都出不来）：休止在每小节第 3 拍切开（r.2 r.2；4/4 的休止本来也不跨第 3 拍）。空的拍效果只切休止、不切音
    halves = {i * 4.0 + 2: [] for i in range(n_bars)}
    lines = _voice_lines(n_bars, pieces, [None] * n_bars, feel, {**halves, **effects}, tie_effects=True,
                         head_effects=legato_slurs(line))
    remarks: dict[int, list[str]] = {}
    for t, text in dropped:
        remarks.setdefault(int(t // 4), []).append(text)
    return atex.document(head, [annotate(_label_sections(lines, sections), remarks)])
