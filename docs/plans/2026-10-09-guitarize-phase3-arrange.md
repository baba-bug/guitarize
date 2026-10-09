# Guitarize Phase 3：编配草稿生成 + 获取 + Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从 leadsheet.json 自动生成弹唱与指弹的 alphaTex 草稿（v1），可经 Phase 1 校验器直接打分；补齐音源获取与命令行；把整套流程写成可复用的 Claude Code skill。

**Architecture:** `arrange/` 下按职责拆分：alphaTex 文本工具（atex）→ 和弦指型库（shapes）→ 变调夹选择（capo）→ 节奏型（patterns）→ 弹唱生成（strum）；指法 Viterbi（fretting）→ 指弹生成（finger）；编排（pipeline）把 leadsheet、分轨起音、难度档串起来写出 `arrangements/<arr>.atex`。生成器只出草稿，音乐性取舍由主会话（Claude）按校验报告逐小节修改。

**Tech Stack:** Python 3.11、numpy；Phase 1 的 theory / skeleton / playability / metrics / alphatab 构建；yt-dlp、ffmpeg。

**Spec:** `docs/specs/2026-10-09-guitarize-design.md`（§4.2 获取、§4.4 编配、§4.7 迭代协议、§6 目录、§10 复用）

## 全局约束

- Python：`PY=~/anaconda3/envs/guitarize/bin/python`；测试在仓库根目录 `$PY -m pytest`。
- 弦号 1 = 最高音弦；指型 `frets` 元组按 1 弦 → 6 弦，`None` = 不弹。标准调弦空弦音高 `(64, 59, 55, 50, 45, 40)`。
- 指法调音高 = 实际音高 − 变调夹；和弦名在谱面上一律写**指法调名字**（如 capo 4 下实际 B 写作 G）。
- alphaTex 写法（已在 1.8.4 验证）：行内时值 `3.1.4`、`(0.1 1.2).8 {sd}`、休止 `r.2`、附点 `{d}`、和弦连音线 `(-.1 -.2).4`、闷音 `(x.6 x.5).4`、`\section "名" ` 写在小节第一个拍子前、`\voice` 分声部、`// 注释`、`\tuning (E4 B3 G3 D3 A2 E2)`；**不写** `.` 分隔符。
- 每个小节行前写 `// bar N [段落]` 注释，方便逐小节修改。
- 难度档参数只从 `profiles.yaml` 读（intermediate：max_span 3、max_fingers 4、max_fret 12、max_shift 4、allow_barre true、melody_max_per_beat 2、bass_on_beat3 true）。
- 生成器产出的草稿必须 0 可弹性违规（用 Phase 1 `check_playability` 测；含 Phase 1 收尾新增的 `straight_offbeat` 规则）。
- **Shuffle 记谱约定（Phase 1 收尾裁决）**：alphaTab 只把"拍点上的八分 + 紧接的八分"这一对按 2/3 + 1/3 演奏。所以三连音感觉的小节里，每个发声的反拍起音都必须是这样一对里的第二个八分：用 `atex.split_for_swing` 在 p−0.5、p、p+0.5 处切分，余下时值用连音线（休止直接拆分）。
- **品位上限（Phase 1 收尾裁决）**：有效上限 = `min(profile.max_fret, NECK_JOIN_FRET − capo)`，`NECK_JOIN_FRET = 14` 从 `guitarize.verify.playability` 导入；选变调夹和生成指型、指法都用这个有效上限。
- 获取阶段：只下载用户确认过的来源；不破解任何加密格式（.ncm 等）；音频只存在 `songs/<slug>/source/`（不入 git）。

## 文件结构（Phase 3）

```
guitarize/arrange/
  __init__.py   （空）
  atex.py       DURATIONS / split_beats / notes_token / beat_tokens / header / bar_line / document
  shapes.py     OPEN_STRINGS / OPEN_SHAPES / Voicing / sounding / playable / with_bass / voicings
  capo.py       CapoOption / melody_octave / choose_capo
  patterns.py   PATTERNS / DEFAULT / slots_per_beat / accent_vector / choose_pattern
  strum.py      beat_chords / bar_tokens / generate_strum
  fretting.py   FretEvent / assign_frets
  finger.py     written_position / skeleton_notes / bass_events / generate_finger
  pipeline.py   chord_beats / section_patterns / arrange
guitarize/acquire.py   download / normalize / acquire
guitarize/cli.py       （修改）新增 new / acquire / arrange
.claude/skills/guitarize/SKILL.md
tests/test_atex.py test_shapes.py test_strum.py test_fretting.py test_finger.py test_acquire.py test_arrange_cli.py
```

---

### Task 1: alphaTex 文本工具

**Files:**
- Create: `guitarize/arrange/__init__.py`（空）, `guitarize/arrange/atex.py`
- Test: `tests/test_atex.py`

**Interfaces:**
- Produces: `DURATIONS`、`STANDARD_TUNING`、`split_beats(length) -> list[(code, dotted)]`、`notes_token(notes, tie=False, dead=False) -> str`、`beat_tokens(notes, length, effects=(), dead=False, tie=False) -> list[str]`、`Item(start, length, notes, effects=(), dead=False)`（小节内记谱位置，notes 空 = 休止）、`split_for_swing(items, feel) -> list[(Item, is_continuation)]`、`voice_bar_tokens(items, feel) -> list[str]`、`header(title, tempo, capo, feel, staff, chords=None) -> str`、`bar_line(tokens, number, section=None) -> str`、`document(head, voices) -> str`

- [ ] **Step 1: 写失败测试 `tests/test_atex.py`**

```python
import pytest

from guitarize.arrange import atex
from guitarize.render import alphatab
from guitarize.verify.model import load_score


def test_split_beats():
    assert atex.split_beats(4) == [(1, False)]
    assert atex.split_beats(1.5) == [(4, True)]
    assert atex.split_beats(2.5) == [(2, False), (8, False)]
    assert atex.split_beats(0.75) == [(8, True)]
    with pytest.raises(ValueError):
        atex.split_beats(0.1)


def test_tokens():
    assert atex.notes_token([]) == "r"
    assert atex.notes_token([(1, 3)]) == "3.1"
    assert atex.notes_token([(6, 3), (5, 2)]) == "(3.6 2.5)"
    assert atex.notes_token([(1, 3)], tie=True) == "-.1"
    assert atex.notes_token([(6, 0), (5, 0)], dead=True) == "(x.6 x.5)"
    assert atex.beat_tokens([(1, 3)], 2.5, ['ch "G"', "sd"]) == ['3.1.2 {ch "G" sd}', "-.1.8"]
    assert atex.beat_tokens([], 1.5) == ["r.4 {d}"]


def test_split_for_swing_pairs_offbeat_onsets(tmp_path):
    items = [atex.Item(0, 1, [(1, 3)]), atex.Item(1, 0.5, []), atex.Item(1.5, 2.5, [(1, 5)], ["sd"])]
    pieces = atex.split_for_swing(items, "shuffle")
    assert [(p.start, p.length, cont) for p, cont in pieces] == [(0, 1, False), (1, 0.5, False), (1.5, 0.5, False), (2.0, 2.0, True)]
    assert pieces[2][0].effects == ["sd"] and pieces[3][0].effects == []
    assert atex.split_for_swing(items, "straight") == [(it, False) for it in items]
    tokens = atex.voice_bar_tokens(items, "shuffle")
    assert tokens == ["3.1.4", "r.8", "5.1.8 {sd}", "-.1.2"]
    head = atex.header("T", 120, 0, "shuffle", "tabs")
    src = tmp_path / "s.atex"
    src.write_text(atex.document(head, [[atex.bar_line(tokens, 1)]]), encoding="utf-8")
    score = load_score(alphatab.build(src, tmp_path / "b", "s")["json"])
    from guitarize.config import load_profile
    from guitarize.verify.playability import check_playability
    assert check_playability(score, load_profile("intermediate"), "finger") == []
    assert [round(a.start, 3) for a in score.attacks()] == [0.0, 1.667]


def test_document_builds_with_alphatab(tmp_path):
    head = atex.header('My "Song"', 168.4, 4, "shuffle", "score tabs", {"G": (3, 0, 0, 0, 2, 3)})
    v1 = [atex.bar_line(atex.beat_tokens([(1, 3)], 4, ['ch "G"']), 1, "サビ")]
    v2 = [atex.bar_line(atex.beat_tokens([(6, 3)], 4), 1)]
    text = atex.document(head, [v1, v2])
    assert "// bar 1 [サビ]" in text and '\\section "サビ"' in text and text.count("\\voice") == 2
    src = tmp_path / "t.atex"
    src.write_text(text, encoding="utf-8")
    score = load_score(alphatab.build(src, tmp_path / "b", "t")["json"])
    assert score.capo == 4 and score.tempo == 168 and score.bars[0].section == "サビ"
    assert score.bars[0].feel == "triplet8th"
    assert [n.midi for n in score.melody(0)] == [71]
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest tests/test_atex.py -v`
Expected: FAIL（`ModuleNotFoundError: guitarize.arrange`）

- [ ] **Step 3: 实现 `guitarize/arrange/atex.py`**

```python
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
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest tests/test_atex.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add guitarize/arrange tests/test_atex.py
git commit -m "feat: alphaTex writing helpers"
```

---

### Task 2: 和弦指型库与变调夹选择

**Files:**
- Create: `guitarize/arrange/shapes.py`, `guitarize/arrange/capo.py`
- Test: `tests/test_shapes.py`

**Interfaces:**
- Consumes: `theory.parse_chord / pc_name / transpose_label`；`playability.fingers_needed`；`Profile`
- Produces:
  - `shapes.OPEN_STRINGS`、`OPEN_SHAPES: dict[str, tuple]`、`Voicing(name, frets, cost, barre=False)`、`sounding(frets) -> list[int]`、`playable(frets, profile) -> bool`、`with_bass(v, bass_pc, profile) -> Voicing | None`、`voicings(label, profile) -> list[Voicing]`（按代价升序；`'N'` → `[]`；没有可弹指型抛 `ValueError`；`'B5'` 按 B 大三处理）
  - `capo.CapoOption(capo, cost, shapes: dict[实际和弦, Voicing], melody_shift)`、`melody_octave(pitches) -> (shift, out_ratio)`、`effective_profile(profile, capo) -> Profile`、`choose_capo(chord_beats, profile, melody=(), max_capo=7) -> list[CapoOption]`（按 cost 升序）

- [ ] **Step 1: 写失败测试 `tests/test_shapes.py`**

```python
import pytest

from guitarize.arrange.capo import choose_capo, melody_octave
from guitarize.arrange.shapes import playable, voicings, with_bass
from guitarize.config import load_profile

MID = load_profile("intermediate")
BEG = load_profile("beginner")


def test_open_shape_first():
    v = voicings("G", MID)[0]
    assert v.name == "G" and v.frets == (3, 0, 0, 0, 2, 3) and v.cost == 0.0


def test_substitute_seventh_for_missing_open_major():
    v = voicings("B", MID)[0]
    assert v.name == "B7" and v.frets == (2, 0, 2, 1, 2, None)


def test_slash_chords():
    assert voicings("G/B", MID)[0].frets == (3, 0, 0, 0, 2, None)
    assert voicings("D/F#", MID)[0].frets == (2, 3, 2, 0, None, 2)
    assert voicings("C/E", MID)[0].frets == (0, 1, 0, 2, 3, 0)
    v = voicings("B/D#", MID)[0]
    assert v.name == "B7/D#" and v.frets == (2, 0, 2, 1, None, None)


def test_barre_only_when_allowed():
    assert voicings("F", MID)[0].barre is True
    with pytest.raises(ValueError):
        voicings("F#m7-5", BEG)
    assert all(not v.barre for v in voicings("A", BEG))


def test_power_chord_and_no_chord():
    assert voicings("E5", MID)[0].name == "E"
    assert voicings("N", MID) == []


def test_with_bass_moves_to_fifth_string_when_sixth_is_too_far():
    v = with_bass(voicings("G", MID)[0], 1, MID)   # G/C#：6 弦需要 9 品（跨度超限）→ 改用 5 弦 4 品
    assert v.name == "G/C#" and v.frets == (3, 0, 0, 0, 4, None) and playable(v.frets, MID)


def test_choose_capo_prefers_g_shapes_for_b_major_song():
    beats = {"B": 16, "E": 16, "D#/G": 8, "G#m": 8, "C#": 8, "D#": 8}
    best = choose_capo(beats, MID)[0]
    assert best.capo == 4
    assert best.shapes["D#/G"].name == "B7/D#"
    assert best.shapes["G#m"].name == "Em" and best.shapes["E"].name == "C"


def test_melody_octave():
    assert melody_octave([71, 73, 75, 76]) == (-12, 0.0)
    assert melody_octave([]) == (0, 0.0)
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest tests/test_shapes.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `guitarize/arrange/shapes.py`**

```python
"""和弦指型库：开放和弦、E/A 型横按、替代指型、斜杠低音调整；按难度档过滤，代价升序。"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

from .. import theory
from ..config import Profile
from ..verify.playability import fingers_needed

OPEN_STRINGS = (64, 59, 55, 50, 45, 40)  # 1 弦 → 6 弦

OPEN_SHAPES: dict[str, tuple] = {
    "C": (0, 1, 0, 2, 3, None), "A": (0, 2, 2, 2, 0, None), "G": (3, 0, 0, 0, 2, 3),
    "E": (0, 0, 1, 2, 2, 0), "D": (2, 3, 2, 0, None, None),
    "Am": (0, 1, 2, 2, 0, None), "Em": (0, 0, 0, 2, 2, 0), "Dm": (1, 3, 2, 0, None, None),
    "C7": (0, 1, 3, 2, 3, None), "A7": (0, 2, 0, 2, 0, None), "G7": (1, 0, 0, 0, 2, 3),
    "E7": (0, 0, 1, 0, 2, 0), "D7": (2, 1, 2, 0, None, None), "B7": (2, 0, 2, 1, 2, None),
    "Am7": (0, 1, 0, 2, 0, None), "Em7": (0, 0, 0, 0, 2, 0), "Dm7": (1, 1, 2, 0, None, None),
    "Cmaj7": (0, 0, 0, 2, 3, None), "Fmaj7": (0, 1, 2, 3, None, None),
    "Asus4": (0, 3, 2, 2, 0, None), "Dsus4": (3, 3, 2, 0, None, None), "Esus4": (0, 0, 2, 2, 2, 0),
}
# 横按形状：(根音所在空弦的 pc, {性质: 以空弦为根时的品位})
_BARRE = {
    "E": (4, {"maj": (0, 0, 1, 2, 2, 0), "min": (0, 0, 0, 2, 2, 0), "7": (0, 0, 1, 0, 2, 0), "min7": (0, 0, 0, 0, 2, 0)}),
    "A": (9, {"maj": (0, 2, 2, 2, 0, None), "min": (0, 1, 2, 2, 0, None), "7": (0, 2, 0, 2, 0, None), "min7": (0, 1, 0, 2, 0, None)}),
}
_SUFFIX = {"maj": "", "min": "m", "7": "7", "min7": "m7", "maj7": "maj7", "sus4": "sus4", "sus2": "sus2",
           "dim": "dim", "aug": "aug", "min7b5": "m7-5"}
# 找不到该性质的指型时可用的替代（性质, 额外代价）
_SUBSTITUTE = {"maj": [("7", 0.3)], "7": [("maj", 0.2)], "min7": [("min", 0.2)], "maj7": [("maj", 0.2)],
               "sus4": [("maj", 0.4)], "sus2": [("maj", 0.4)], "aug": [("maj", 0.6)], "dim": [("min", 0.6)],
               "min7b5": [("min", 0.6)]}
_POWER = re.compile(r"^([A-G][#b]?)5$")


@dataclass(frozen=True)
class Voicing:
    name: str      # 和弦图名称（指法调），如 "G"、"B7/D#"
    frets: tuple   # 1 弦 → 6 弦，None = 不弹
    cost: float
    barre: bool = False


def sounding(frets: tuple) -> list[int]:
    return [OPEN_STRINGS[i] + f for i, f in enumerate(frets) if f is not None]


def playable(frets: tuple, profile: Profile) -> bool:
    fretted = [f for f in frets if f is not None and f > 0]
    if not fretted:
        return True
    return (max(fretted) - min(fretted) <= profile.max_span
            and fingers_needed(fretted) <= profile.max_fingers
            and max(fretted) <= profile.max_fret)


def _shapes_for(root: int, quality: str, profile: Profile) -> list[Voicing]:
    name = theory.pc_name(root) + _SUFFIX[quality]
    out = []
    if name in OPEN_SHAPES:
        out.append(Voicing(name, OPEN_SHAPES[name], 0.0))
    if profile.allow_barre:
        for open_pc, table in _BARRE.values():
            k = (root - open_pc) % 12
            if quality in table and k:
                frets = tuple(None if f is None else f + k for f in table[quality])
                out.append(Voicing(name, frets, 1.0 + 0.05 * k, barre=True))
    return out


def with_bass(v: Voicing, bass_pc: int, profile: Profile) -> Voicing | None:
    """把最低音换成 bass_pc：依次尝试 6、5、4 弦（其下的弦不弹）；须仍为最低音且可弹。"""
    for idx in (5, 4, 3):
        fret = (bass_pc - OPEN_STRINGS[idx]) % 12
        frets = list(v.frets)
        frets[idx] = fret
        for j in range(idx + 1, 6):
            frets[j] = None
        frets = tuple(frets)
        if min(sounding(frets)) == OPEN_STRINGS[idx] + fret and playable(frets, profile):
            return Voicing(f"{v.name}/{theory.pc_name(bass_pc)}", frets, v.cost + 0.2, v.barre)
    return None


def voicings(label: str, profile: Profile) -> list[Voicing]:
    """指法调里的和弦名 → 可弹指型（代价升序）。"""
    m = _POWER.match(label)
    if m:
        label = m.group(1)
    parsed = theory.parse_chord(label)
    if parsed is None:
        return []
    root, quality, bass = parsed
    cands = _shapes_for(root, quality, profile)
    for alt, extra in _SUBSTITUTE.get(quality, []):
        cands += [replace(v, cost=v.cost + extra) for v in _shapes_for(root, alt, profile)]
    if bass is not None and bass != root:
        cands = [with_bass(v, bass, profile) or replace(v, cost=v.cost + 0.5) for v in cands]
    cands = [v for v in cands if playable(v.frets, profile)]
    if not cands:
        raise ValueError(f"no playable voicing for {label!r} at level {profile.name}")
    return sorted(cands, key=lambda v: v.cost)
```

- [ ] **Step 4: 实现 `guitarize/arrange/capo.py`**

```python
"""变调夹 / 指法调选择：对 0–7 品逐一打分（和弦指型代价 × 拍数 + 变调夹罚分 + 指弹旋律音域罚分）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from dataclasses import replace

from .. import theory
from ..config import Profile
from ..verify.playability import NECK_JOIN_FRET
from .shapes import Voicing, voicings

MELODY_LOW, MELODY_HIGH, MELODY_CENTER = 55, 74, 62   # G3–D5，中心 D4（指法调音高）


def effective_profile(profile: Profile, capo: int) -> Profile:
    """变调夹之上可用的品位上限不超过琴颈与琴身交界（14 品）。"""
    return replace(profile, max_fret=min(profile.max_fret, NECK_JOIN_FRET - capo))


@dataclass(frozen=True)
class CapoOption:
    capo: int
    cost: float
    shapes: dict          # 实际和弦名 → Voicing（指法调）
    melody_shift: int     # 指弹旋律整体移的半音数（八度倍数）


def melody_octave(pitches: Sequence[int]) -> tuple[int, float]:
    """在 −24 / −12 / 0 / +12 中选整体移位：先让越界音最少，再让中位数最接近 D4。"""
    if not len(pitches):
        return 0, 0.0
    best = None
    for shift in (-24, -12, 0, 12):
        p = np.asarray(pitches) + shift
        out = float(np.mean((p < MELODY_LOW) | (p > MELODY_HIGH)))
        key = (out, abs(float(np.median(p)) - MELODY_CENTER))
        if best is None or key < best[0]:
            best = (key, shift, out)
    return best[1], best[2]


def choose_capo(chord_beats: dict[str, float], profile: Profile, melody: Sequence[int] = (),
                max_capo: int = 7) -> list[CapoOption]:
    total = sum(chord_beats.values()) or 1.0
    options = []
    for capo in range(max_capo + 1):
        shapes: dict[str, Voicing] = {}
        cost = 0.0
        for label, beats in chord_beats.items():
            try:
                v = voicings(theory.transpose_label(label, -capo), effective_profile(profile, capo))[0]
            except ValueError:
                cost += beats * 5.0
                continue
            shapes[label] = v
            cost += beats * v.cost
        cost += 0.02 * capo * total
        shift = 0
        if len(melody):
            shift, out = melody_octave([m - capo for m in melody])
            cost += 2.0 * out * total
        options.append(CapoOption(capo, round(cost, 3), shapes, shift))
    return sorted(options, key=lambda o: o.cost)
```

- [ ] **Step 5: 运行，确认通过**

Run: `$PY -m pytest tests/test_shapes.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add guitarize/arrange/shapes.py guitarize/arrange/capo.py tests/test_shapes.py
git commit -m "feat: chord voicing library and capo selection"
```

---

### Task 3: 节奏型与弹唱草稿

**Files:**
- Create: `guitarize/arrange/patterns.py`, `guitarize/arrange/strum.py`
- Test: `tests/test_strum.py`

**Interfaces:**
- Consumes: `atex`（Task 1）、`shapes.voicings / Voicing`（Task 2）、`theory.transpose_label`
- Produces:
  - `patterns.PATTERNS: dict[str, str]`（8 个八分格；`D` 下扫、`U` 上扫、`x` 闷音、`-` 延续）、`DEFAULT = "push"`、`slots_per_beat(feel)`、`accent_vector(pattern, feel) -> np.ndarray`（长度 4×slots_per_beat）、`choose_pattern(onset, feel) -> str`
  - `strum.beat_chords(lbar, bpb=4) -> list[str]`、`strum.bar_tokens(chords, pattern, shapes, prev, feel="straight") -> (tokens, last_label)`、`strum.generate_strum(leadsheet, capo, profile, patterns, title) -> str`（`patterns` 为逐小节节奏型名；内部用 `effective_profile(profile, capo)`）

- [ ] **Step 1: 写失败测试 `tests/test_strum.py`**

```python
import numpy as np

from guitarize.arrange.patterns import accent_vector, choose_pattern
from guitarize.arrange.strum import generate_strum
from guitarize.config import Thresholds, load_profile
from guitarize.render import alphatab
from guitarize.verify import metrics
from guitarize.verify.model import load_score
from guitarize.verify.playability import check_playability

MID = load_profile("intermediate")


def leadsheet(bars, feel="straight", tempo=120.0):
    out = []
    for i, chords in enumerate(bars):
        out.append({"bar": i + 1, "section": "A" if i == 0 else None, "t_start": i * 2.0, "t_end": (i + 1) * 2.0,
                    "chords": [{"beat": b, "dur_beats": d, "label": l, "confidence": "high"} for l, b, d in chords],
                    "melody": []})
    return {"key": "B", "mode": "major", "tempo_bpm": tempo, "feel": feel, "beats_per_bar": 4, "bars": out}


def build(text, tmp_path, name="s"):
    src = tmp_path / f"{name}.atex"
    src.write_text(text, encoding="utf-8")
    return load_score(alphatab.build(src, tmp_path / "b", name)["json"])


def test_accent_vector_and_choice():
    v = accent_vector("D-DU-UDU", "shuffle")
    assert v.shape == (12,) and list(np.flatnonzero(v)) == [0, 3, 5, 8, 9, 11]
    assert choose_pattern(accent_vector("D-DU-UDU", "shuffle"), "shuffle") == "pop"
    assert choose_pattern(None, "shuffle") == "push"
    assert choose_pattern(np.zeros(12), "shuffle") == "push"


def test_generate_strum_capo4_b_major(tmp_path):
    ls = leadsheet([[("B", 1, 4)], [("E", 1, 2), ("D#/G", 3, 2)], [("G#m", 1, 4)], [("N", 1, 4)]], feel="shuffle")
    text = generate_strum(ls, 4, MID, ["push", "pop", "push", "push"], "Test")
    assert '\\chord "G"' in text and '\\chord "B7/D#"' in text and "\\capo 4" in text
    assert "// bar 1 [A]" in text
    score = build(text, tmp_path)
    assert check_playability(score, MID, "strum") == []
    bars = metrics.score_bars(score, ls, "strum")
    s = metrics.summarize(bars, [], Thresholds(), "strum")
    assert s["chord_agreement"] == 1.0 and s["chord_mark_issues"] == 0
    attacks = [a for a in score.attacks() if a.bar == 0]
    assert len(attacks) == 6 and attacks[0].direction == "down"
    # 第 2 小节第 3 拍换和弦：pop 型第 5 格原是延续，须改为下扫
    bar2 = [a for a in score.attacks() if a.bar == 1]
    assert any(abs(a.start - 6.0) < 1e-6 and a.direction == "down" for a in bar2)
    assert not [a for a in score.attacks() if a.bar == 3]
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest tests/test_strum.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `guitarize/arrange/patterns.py`**

```python
"""弹唱节奏型：8 个八分格。D = 下扫，U = 上扫，x = 闷音下扫，- = 延续。按与原曲起音的相关度选型。"""
from __future__ import annotations

import numpy as np

PATTERNS = {
    "whole": "D-------",
    "half": "D---D---",
    "quarter": "D-D-D-D-",
    "push": "D-DUD-DU",
    "pop": "D-DU-UDU",
    "eighth": "DUDUDUDU",
}
DEFAULT = "push"
_WEIGHT = {"D": 1.0, "x": 0.8, "U": 0.6, "-": 0.0}


def slots_per_beat(feel: str) -> int:
    return 3 if feel == "shuffle" else 2


def accent_vector(pattern: str, feel: str) -> np.ndarray:
    """节奏型 → 每拍 slots_per_beat 格的重音向量；反拍八分落在 swing 位置（shuffle 为第 3 格）。"""
    spb = slots_per_beat(feel)
    v = np.zeros(4 * spb)
    for k, ch in enumerate(pattern):
        pos = (k // 2) * spb + (0 if k % 2 == 0 else spb - 1)
        v[pos] = max(v[pos], _WEIGHT[ch])
    return v


def choose_pattern(onset: np.ndarray | None, feel: str) -> str:
    """与原曲起音向量相关度最高的节奏型；相关度相同取扫弦次数少的；无数据时用默认型。"""
    if onset is None or np.std(onset) == 0:
        return DEFAULT
    best = None
    for name, pattern in PATTERNS.items():
        r = float(np.corrcoef(accent_vector(pattern, feel), onset)[0, 1])
        key = (round(r, 3), -sum(c != "-" for c in pattern))
        if best is None or key > best[0]:
            best = (key, name)
    return best[1]
```

- [ ] **Step 4: 实现 `guitarize/arrange/strum.py`**

```python
"""弹唱编配草稿：逐拍和弦 → 指法调指型 → 每小节套节奏型 → alphaTex（单声部 + 和弦图）。"""
from __future__ import annotations

from .. import theory
from ..config import Profile
from . import atex
from .capo import effective_profile
from .patterns import PATTERNS
from .shapes import Voicing, voicings

UP_STRINGS = 3  # 上扫只扫最细的 3 根弦


def beat_chords(lbar: dict, bpb: int = 4) -> list[str]:
    """leadsheet 小节 → 逐拍和弦标签（无和弦为 'N'）。"""
    out = ["N"] * bpb
    for seg in lbar.get("chords", []):
        first = int(round(seg["beat"])) - 1
        for k in range(max(0, first), min(bpb, first + int(round(seg["dur_beats"])))):
            out[k] = seg["label"]
    return out


def _notes(v: Voicing, stroke: str) -> list[tuple[int, int]]:
    notes = [(i + 1, f) for i, f in enumerate(v.frets) if f is not None]
    return [n for n in notes if n[0] <= UP_STRINGS] if stroke == "U" else notes


def bar_tokens(chords: list[str], pattern: str, shapes: dict[str, Voicing],
               prev: str | None, feel: str = "straight") -> tuple[list[str], str | None]:
    """chords：指法调逐拍和弦（4 个）。换和弦或休止之后的延续格改为下扫；无和弦的格为休止。
    Shuffle 下反拍扫弦按八分对切分（atex.voice_bar_tokens）。"""
    slot_chord = [chords[k // 2] for k in range(8)]
    slots = list(pattern)
    for k in range(8):
        if slot_chord[k] == "N":
            slots[k] = "r"
        elif slots[k] == "-" and (k == 0 or slot_chord[k] != slot_chord[k - 1] or slots[k - 1] == "r"):
            slots[k] = "D"
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
            effects = ([f'ch "{v.name}"'] if label != last else []) + ["su" if slots[k] == "U" else "sd"]
            items.append(atex.Item(k * 0.5, n * 0.5, _notes(v, slots[k]), effects, dead=slots[k] == "x"))
            last = label
        k += n
    return atex.voice_bar_tokens(items, feel), last


def generate_strum(leadsheet: dict, capo: int, profile: Profile, patterns: list[str], title: str) -> str:
    profile = effective_profile(profile, capo)
    bpb = leadsheet.get("beats_per_bar", 4)
    shapes: dict[str, Voicing] = {}
    lines: list[str] = []
    prev = None
    for i, b in enumerate(leadsheet["bars"]):
        chords = [c if c == "N" else theory.transpose_label(c, -capo) for c in beat_chords(b, bpb)]
        for c in chords:
            if c != "N" and c not in shapes:
                shapes[c] = voicings(c, profile)[0]
        tokens, prev = bar_tokens(chords, PATTERNS[patterns[i]], shapes, prev, leadsheet.get("feel", "straight"))
        lines.append(atex.bar_line(tokens, i + 1, b.get("section")))
    diagrams = {v.name: v.frets for v in shapes.values()}
    head = atex.header(title, leadsheet["tempo_bpm"], capo, leadsheet.get("feel", "straight"), "tabs", diagrams)
    return atex.document(head, [lines])
```

- [ ] **Step 5: 运行，确认通过**

Run: `$PY -m pytest tests/test_strum.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add guitarize/arrange/patterns.py guitarize/arrange/strum.py tests/test_strum.py
git commit -m "feat: strum patterns and strum arrangement generator"
```

---

### Task 4: 指弹指法分配（Viterbi）

**Files:**
- Create: `guitarize/arrange/fretting.py`
- Test: `tests/test_fretting.py`

**Interfaces:**
- Consumes: `Profile`；`shapes.OPEN_STRINGS`
- Produces: `FretEvent(start, melody, bass, fill=None, melody_pos=None, fill_pos=None, bass_pos=None, fallback=None)`（音高为指法调音高；`fill` 为低音之上的一个和弦内音；结果写回 `*_pos` = `(弦, 品)`，`fallback` 记录降级方案）、`assign_frets(events, profile) -> list[FretEvent]`（原地填写并返回）

- [ ] **Step 1: 写失败测试 `tests/test_fretting.py`**

```python
from guitarize.arrange.fretting import FretEvent, assign_frets
from guitarize.config import load_profile

MID = load_profile("intermediate")


def test_first_position_melody_with_bass():
    evs = assign_frets([FretEvent(0, 59, 43), FretEvent(1, 60, 48), FretEvent(2, 62, None)], MID)
    assert [(e.melody_pos, e.bass_pos) for e in evs] == [((2, 0), (6, 3)), ((2, 1), (5, 3)), ((2, 3), None)]
    assert all(e.fallback is None for e in evs)


def test_unplayable_event_falls_back_without_crashing():
    ev = assign_frets([FretEvent(0, 71, 43)], MID)[0]
    assert ev.fallback == "bass+12" and ev.melody_pos == (1, 7) and ev.bass_pos == (3, 0)


def test_bass_above_melody_is_dropped_an_octave():
    ev = assign_frets([FretEvent(0, 57, 60)], MID)[0]
    assert ev.melody_pos is not None and ev.bass_pos is not None
    m_string, b_string = ev.melody_pos[0], ev.bass_pos[0]
    assert m_string < b_string


def test_fill_sits_between_melody_and_bass():
    ev = assign_frets([FretEvent(0, 66, 43, fill=59)], MID)[0]
    assert ev.fallback is None
    assert ev.melody_pos == (1, 2) and ev.fill_pos == (2, 0) and ev.bass_pos == (6, 3)


def test_fill_is_kept_when_bass_moves_up_an_octave():
    ev = assign_frets([FretEvent(0, 71, 43, fill=59)], MID)[0]
    assert ev.fallback == "bass+12"
    assert ev.melody_pos == (1, 7) and ev.fill_pos == (2, 0) and ev.bass_pos == (3, 0)


def test_melody_prefers_top_three_strings():
    evs = assign_frets([FretEvent(t, p, None) for t, p in enumerate([55, 57, 59, 60])], MID)
    assert all(e.melody_pos[0] <= 3 for e in evs)
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest tests/test_fretting.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `guitarize/arrange/fretting.py`**

```python
"""指弹指法分配：每个起音时刻枚举可弹的（弦, 品）组合，Viterbi 选出整体最顺手的把位。

组合规则：音高从高到低依次放在编号严格递增的弦上（旋律在上、低音在下）；按弦跨度 ≤ max_span，品位 ≤ max_fret。
静态代价：把位越高越贵、跨度越大越贵、7 品以上加罚、空弦减分、旋律不在 1–3 弦加罚。
转移代价：把位移动的品数 ×（1 + 1/时间间隔）。
音高从高到低必须是 旋律 > 填充 > 低音。某时刻无解时按顺序降级：低音降八度 → 低音升八度 → 去掉填充 →
（去掉填充后）低音降 / 升八度 → 去掉低音 → 旋律升八度 → 旋律降八度，并记录在 fallback。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from ..config import Profile
from .shapes import OPEN_STRINGS


@dataclass
class FretEvent:
    start: float               # 记谱位置（拍）
    melody: int | None         # 指法调音高
    bass: int | None
    fill: int | None = None    # 低音之上的和弦内音
    melody_pos: tuple[int, int] | None = None
    fill_pos: tuple[int, int] | None = None
    bass_pos: tuple[int, int] | None = None
    fallback: str | None = None


def _candidates(pitches: list[int], profile: Profile) -> list[tuple]:
    per = [[(s + 1, p - OPEN_STRINGS[s]) for s in range(6) if 0 <= p - OPEN_STRINGS[s] <= profile.max_fret]
           for p in pitches]
    out = []
    for combo in itertools.product(*per):
        strings = [c[0] for c in combo]
        if any(b <= a for a, b in zip(strings, strings[1:])):
            continue
        fretted = [f for _, f in combo if f > 0]
        if fretted and max(fretted) - min(fretted) > profile.max_span:
            continue
        out.append(combo)
    return out


def _static(combo: tuple, has_melody: bool) -> float:
    fretted = [f for _, f in combo if f > 0]
    cost = 0.0
    if fretted:
        cost += 0.15 * min(fretted) + 0.3 * (max(fretted) - min(fretted)) + 0.5 * max(0, max(fretted) - 7)
    cost -= 0.3 * sum(1 for _, f in combo if f == 0)
    if has_melody and combo:
        cost += 0.5 * max(0, combo[0][0] - 3)
    return cost


def _position(combo: tuple) -> int | None:
    fretted = [f for _, f in combo if f > 0]
    return min(fretted) if fretted else None


def _transition(a: tuple, b: tuple, gap: float) -> float:
    pa, pb = _position(a), _position(b)
    if pa is None or pb is None:
        return 0.0
    return abs(pa - pb) * (1.0 + 1.0 / max(gap, 0.25))


def _ordered(m: int | None, f: int | None, b: int | None) -> tuple[int | None, int | None, int | None]:
    """保证 旋律 > 填充 > 低音：低音不低于旋律时降八度；填充不在两者之间时丢弃。"""
    if m is not None and b is not None and b >= m:
        b -= 12 * ((b - m) // 12 + 1)
    if f is not None:
        if m is not None:
            while f >= m:
                f -= 12
        if b is not None and f <= b:
            f = None
    return m, f, b


def _variants(m: int | None, f: int | None, b: int | None) -> list[tuple]:
    """(旋律, 填充, 低音, 降级说明)，按尝试顺序排列。"""
    m, f, b = _ordered(m, f, b)
    out = [(m, f, b, None)]
    if b is not None:
        out.append(_ordered(m, f, b - 12) + ("bass-12",))
        if (m is None or b + 12 < m) and (f is None or b + 12 < f):
            out.append((m, f, b + 12, "bass+12"))
    if f is not None:
        out.append((m, None, b, "drop fill"))
        if b is not None:
            out.append((m, None, b - 12, "drop fill, bass-12"))
            if m is None or b + 12 < m:
                out.append((m, None, b + 12, "drop fill, bass+12"))
    if b is not None and m is not None:
        out.append((m, None, None, "drop bass"))
    if m is not None:
        out += [(m + 12, None, None, "melody+12"), (m - 12, None, None, "melody-12")]
    return out


def assign_frets(events: list[FretEvent], profile: Profile) -> list[FretEvent]:
    if not events:
        return events
    layers = []  # 每个事件：(候选组合列表, (旋律, 填充, 低音), 降级说明)
    for ev in events:
        for m, f, b, label in _variants(ev.melody, ev.fill, ev.bass):
            pitches = [p for p in (m, f, b) if p is not None]
            cands = _candidates(pitches, profile) if pitches else [()]
            if cands:
                layers.append((cands, (m, f, b), label))
                break
        else:
            layers.append(([()], (None, None, None), "dropped"))
    cost = [_static(c, layers[0][1][0] is not None) for c in layers[0][0]]
    back = []
    for i in range(1, len(layers)):
        gap = events[i].start - events[i - 1].start
        prev = layers[i - 1][0]
        new, bp = [], []
        for c in layers[i][0]:
            k = min(range(len(prev)), key=lambda j: cost[j] + _transition(prev[j], c, gap))
            new.append(cost[k] + _transition(prev[k], c, gap) + _static(c, layers[i][1][0] is not None))
            bp.append(k)
        cost = new
        back.append(bp)
    k = min(range(len(cost)), key=cost.__getitem__)
    path = [k]
    for bp in reversed(back):
        path.append(bp[path[-1]])
    path.reverse()
    for ev, (cands, used, label), k in zip(events, layers, path):
        positions = iter(cands[k])
        ev.melody_pos, ev.fill_pos, ev.bass_pos = (next(positions) if p is not None else None for p in used)
        ev.fallback = label
    return events
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest tests/test_fretting.py -v`
Expected: 6 passed。若 `test_first_position_melody_with_bass` 的期望与 Viterbi 结果不同：先打印两种方案的代价，确认差异是否由代价函数合理导致；只有在测试期望本身不合理（例如有同样顺手的等价把位）时才改测试，并在报告中写明。

- [ ] **Step 5: Commit**

```bash
git add guitarize/arrange/fretting.py tests/test_fretting.py
git commit -m "feat: Viterbi fret assignment for fingerstyle"
```

---

### Task 5: 指弹草稿生成

**Files:**
- Create: `guitarize/arrange/finger.py`
- Test: `tests/test_finger.py`

**Interfaces:**
- Consumes: `skeleton_mask`（Phase 1）、`atex`、`capo.melody_octave`、`fretting`、`strum.beat_chords`、`theory`
- Produces: `written_position(perf, feel) -> float`、`skeleton_notes(leadsheet, profile) -> list[(记谱位置, midi)]`、`bass_events(leadsheet, capo, profile) -> list[(记谱位置, 时值, 低音指法调音高, 填充音 pc)]`、`place_fill(pc, bass, top) -> int | None`、`generate_finger(leadsheet, capo, profile, title) -> str`（两声部：voice 1 旋律、voice 2 低音 + 一个填充和弦音；`\staff {score tabs}`）
- 设计说明：只有旋律 + 低音时，逐拍和声常被推断成别的和弦（例如五音在低音时）。所以每个低音都配一个填充音：和弦三音（三音已是斜杠低音时用根音；强力和弦用五音），放在低音之上、旋律之下，优先 D3–E4（50–64）；第 3 拍和弦不变时重复根音（不用五音）。

- [ ] **Step 1: 写失败测试 `tests/test_finger.py`**

```python
import pytest

from guitarize.arrange.finger import bass_events, generate_finger, place_fill, skeleton_notes, written_position
from guitarize.config import Thresholds, load_profile
from guitarize.render import alphatab
from guitarize.verify import metrics
from guitarize.verify.model import load_score
from guitarize.verify.playability import check_playability

MID = load_profile("intermediate")


def leadsheet(feel="shuffle"):
    # B 大调两小节：B（4 拍）、E（2 拍）+ D#/G（2 拍）。旋律在演奏时间轴（Shuffle 反拍在 2/3）。
    mel1 = [(0, 1, 75), (1, 2 / 3, 73), (5 / 3, 1 / 3, 71), (2, 2, 70)]
    mel2 = [(4, 1, 68), (5, 1, 71), (6, 2, 70)]
    bars = [
        {"bar": 1, "section": "サビ", "t_start": 0.0, "t_end": 1.44,
         "chords": [{"beat": 1, "dur_beats": 4, "label": "B", "confidence": "high"}],
         "melody": [{"start_beats": s, "dur_beats": d, "midi": m, "conf": 1.0} for s, d, m in mel1]},
        {"bar": 2, "section": None, "t_start": 1.44, "t_end": 2.88,
         "chords": [{"beat": 1, "dur_beats": 2, "label": "E", "confidence": "high"},
                    {"beat": 3, "dur_beats": 2, "label": "D#/G", "confidence": "high"}],
         "melody": [{"start_beats": s, "dur_beats": d, "midi": m, "conf": 1.0} for s, d, m in mel2]},
    ]
    return {"key": "B", "mode": "major", "tempo_bpm": 167.0, "feel": feel, "beats_per_bar": 4, "bars": bars}


def test_written_position():
    assert written_position(2 / 3, "shuffle") == 0.5
    assert written_position(1 / 3, "shuffle") == 0.0
    assert written_position(1.0, "shuffle") == 1.0
    assert written_position(1.5, "straight") == 1.5
    assert written_position(1.2, "straight") == 1.0


def test_skeleton_notes_drop_short_offbeats():
    notes = skeleton_notes(leadsheet(), MID)
    assert (1.5, 71) not in notes          # 5/3 拍的短反拍音不是骨架音
    assert notes[0] == (0.0, 75) and (2.0, 70) in notes


def test_bass_events_follow_chords_and_slash():
    ev = bass_events(leadsheet(), 4, MID)
    # 实际 B / E / D#/G 在 capo 4 下为指法调 G / C / B/D#
    assert ev == [(0.0, 2.0, 43, 11), (2.0, 2.0, 43, 11),   # G：根音 G2，填充三音 B；第 3 拍重复根音
                  (4.0, 2.0, 48, 4),                        # C：根音 C3，填充三音 E
                  (6.0, 2.0, 51, 11)]                       # B/D#：低音 D#3；三音就是低音 → 填充根音 B


def test_place_fill():
    assert place_fill(11, 43, 66) == 59      # B3：在 G2 之上、F#4 之下，且落在 50–64
    assert place_fill(4, 48, 64) == 52       # E3（E4 = 64 不低于旋律）
    assert place_fill(4, 60, 62) is None     # 两者之间放不下 E


def test_generate_finger_is_playable_and_matches_reference(tmp_path):
    ls = leadsheet()
    text = generate_finger(ls, 4, MID, "Test")
    assert text.count("\\voice") == 2 and "\\capo 4" in text and "// bar 1 [サビ]" in text
    src = tmp_path / "f.atex"
    src.write_text(text, encoding="utf-8")
    score = load_score(alphatab.build(src, tmp_path / "b", "f")["json"])
    assert check_playability(score, MID, "finger") == []
    s = metrics.summarize(metrics.score_bars(score, ls, "finger"), [], Thresholds(), "finger")
    assert s["melody_f1"] == pytest.approx(1.0)
    assert s["chord_agreement"] >= 0.85
    v2 = [e for e in score.events if e.voice == 1 and not e.rest]
    assert all(len(e.notes) == 2 for e in v2)    # 每个低音都带一个填充音
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest tests/test_finger.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `guitarize/arrange/finger.py`**

```python
"""指弹编配草稿：旋律骨架（voice 1）+ 低音（voice 2），指法由 Viterbi 分配。"""
from __future__ import annotations

import re

import numpy as np

from .. import theory
from ..config import Profile
from ..skeleton import skeleton_mask
from . import atex
from .capo import effective_profile, melody_octave
from .fretting import FretEvent, assign_frets
from .strum import beat_chords

BASS_LOW = 40        # E2：低音区间 E2–D#3（指法调）
MAX_MELODY_NOTE = 2.0


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
    """填充音：在 bass 与 top（旋律，无旋律时为 65）之间取该音名最高的音，优先落在 50–64。"""
    limit = 65 if top is None else top
    inside = [p for p in range(bass + 1, limit) if p % 12 == pc]
    preferred = [p for p in inside if FILL_LOW <= p <= FILL_HIGH]
    if preferred:
        return max(preferred)
    return max(inside) if inside else None


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


def generate_finger(leadsheet: dict, capo: int, profile: Profile, title: str) -> str:
    profile = effective_profile(profile, capo)
    feel = leadsheet.get("feel", "straight")
    n_bars = len(leadsheet["bars"])
    mel = skeleton_notes(leadsheet, profile)
    shift, _ = melody_octave([p - capo for _, p in mel])
    mel = [(pos, p - capo + shift) for pos, p in mel]
    items = []
    for j, (pos, p) in enumerate(mel):
        nxt = mel[j + 1][0] if j + 1 < len(mel) else pos + MAX_MELODY_NOTE
        items.append((pos, min(nxt, pos + MAX_MELODY_NOTE, (int(pos // 4) + 1) * 4.0) - pos, p))
    bass = bass_events(leadsheet, capo, profile)

    def melody_at(t: float) -> int | None:
        for pos, dur, p in items:
            if pos - 1e-6 <= t < pos + dur - 1e-6:
                return p
        return None

    m_at = {pos: p for pos, _, p in items}
    b_at = {pos: (p, place_fill(fill_pc, p, melody_at(pos))) for pos, _, p, fill_pc in bass}
    starts = sorted(set(m_at) | set(b_at))
    events = assign_frets([FretEvent(s, m_at.get(s), b_at.get(s, (None, None))[0], fill=b_at.get(s, (None, None))[1])
                           for s in starts], profile)
    at = {e.start: e for e in events}
    sections = [b.get("section") for b in leadsheet["bars"]]
    v1 = _voice_lines(n_bars, [(pos, d, [at[pos].melody_pos]) for pos, d, _ in items], sections, feel)
    v2 = _voice_lines(n_bars, [(pos, d, [at[pos].fill_pos, at[pos].bass_pos]) for pos, d, _, _ in bass],
                      [None] * n_bars, feel)
    return atex.document(atex.header(title, leadsheet["tempo_bpm"], capo, feel, "score tabs"), [v1, v2])
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest tests/test_finger.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add guitarize/arrange/finger.py tests/test_finger.py
git commit -m "feat: fingerstyle arrangement generator (melody skeleton + bass)"
```

---

### Task 6: 编配编排、音源获取与命令行

**Files:**
- Create: `guitarize/arrange/pipeline.py`, `guitarize/acquire.py`
- Modify: `guitarize/cli.py`（新增 `new`、`acquire`、`arrange`）
- Test: `tests/test_acquire.py`, `tests/test_arrange_cli.py`

**Interfaces:**
- Consumes: Task 1–5；Phase 1 `metrics.onset_slots`、`load_profile`、`SongPaths`、`load_song_config`
- Produces:
  - `pipeline.chord_beats(leadsheet) -> dict[str, float]`、`pipeline.section_patterns(paths, leadsheet) -> list[str]`、`pipeline.arrange(paths, cfg, arr, capo=None, force=False) -> tuple[Path, CapoOption]`（`capo` 也可由 `song.yaml` 的 `overrides.capo` 指定；目标文件已存在且未 `force` 时抛 `FileExistsError`）
  - `acquire.download(url, raw_dir) -> Path`、`acquire.normalize(raw, wav, trim=None) -> Path`、`acquire.acquire(paths, url=None, raw=None, trim=None) -> Path`（写 `source/original.wav` 与 `source/meta.json`）
  - CLI：`guitarize new <slug> --title T --artist A`、`guitarize acquire <slug> (--url URL | --raw FILE) [--trim A B]`、`guitarize arrange <slug> --arr {strum,finger} [--capo N] [--force]`

- [ ] **Step 1: 写失败测试**

`tests/test_acquire.py`：
```python
import json
import subprocess

import soundfile as sf

from guitarize.acquire import acquire, normalize
from guitarize.paths import SongPaths


def sine(path, seconds=3):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
                    "-ar", "48000", str(path)], check=True)
    return path


def test_normalize_with_trim(tmp_path):
    out = normalize(sine(tmp_path / "a.m4a"), tmp_path / "o.wav", trim=(0.5, 2.0))
    info = sf.info(str(out))
    assert info.samplerate == 44100 and info.channels == 2
    assert abs(info.duration - 1.5) < 0.05


def test_acquire_from_local_raw(tmp_path):
    paths = SongPaths(tmp_path / "song")
    wav = acquire(paths, raw=sine(tmp_path / "x.m4a"))
    assert wav == paths.original_wav and wav.exists()
    meta = json.loads((paths.source_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["kind"] == "local" and len(meta["sha256"]) == 64 and meta["trim"] is None
```

`tests/test_arrange_cli.py`：
```python
import json

import pytest

from conftest import demo_leadsheet
from guitarize.cli import main
from guitarize.config import load_song_config
from guitarize.paths import SongPaths


def test_new_creates_song_yaml(tmp_path):
    assert main(["--songs-dir", str(tmp_path), "new", "x", "--title", "T", "--artist", "A"]) == 0
    cfg = load_song_config(SongPaths.for_slug("x", tmp_path))
    assert cfg.title == "T" and cfg.level == "intermediate" and cfg.arrangements == ("strum", "finger")


def test_arrange_then_verify(demo_song, capsys):
    songs = str(demo_song.root.parent)
    demo_song.atex("strum").unlink()
    demo_song.atex("finger").unlink()
    for arr in ("strum", "finger"):
        assert main(["--songs-dir", songs, "arrange", "demo", "--arr", arr]) == 0
        assert demo_song.atex(arr).exists()
        assert main(["--songs-dir", songs, "verify", "demo", "--arr", arr]) == 0
        s = json.loads((demo_song.report_root(arr) / "iter-01" / "scores.json").read_text(encoding="utf-8"))["summary"]
        assert s["violations"] == 0 and s["chord_agreement"] >= 0.85
    out = capsys.readouterr().out
    assert "capo 5" in out   # 《小星星》C 调：capo 5 用 G / C / D 开放和弦比 capo 0 的 F 横按更省力


def test_arrange_refuses_to_overwrite(demo_song):
    with pytest.raises(FileExistsError):
        main(["--songs-dir", str(demo_song.root.parent), "arrange", "demo", "--arr", "strum"])
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest tests/test_acquire.py tests/test_arrange_cli.py -v`
Expected: FAIL（`ModuleNotFoundError: guitarize.acquire`）

- [ ] **Step 3: 实现 `guitarize/acquire.py`**

```python
"""获取音源：B 站等公开音频流用 yt-dlp 下载；或使用用户自己放进来的本地文件。
统一用 ffmpeg 转成 44.1kHz 立体声 wav（可截取区间），记录来源到 source/meta.json。
不处理任何加密格式（如 .ncm）。下载前必须已获得用户确认（由调用方 / skill 负责）。"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from .paths import SongPaths


def download(url: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "-f", "bestaudio", "--no-playlist",
         "-o", str(raw_dir / "%(id)s.%(ext)s"), "--print", "after_move:filepath", url],
        capture_output=True, text=True, check=True,
    )
    return Path(proc.stdout.strip().splitlines()[-1])


def normalize(raw: Path, wav: Path, trim: tuple[float, float] | None = None) -> Path:
    wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(raw)]
    if trim:
        cmd += ["-ss", str(trim[0]), "-to", str(trim[1])]
    cmd += ["-ac", "2", "-ar", "44100", str(wav)]
    subprocess.run(cmd, check=True)
    return wav


def acquire(paths: SongPaths, url: str | None = None, raw: Path | None = None,
            trim: tuple[float, float] | None = None) -> Path:
    if (url is None) == (raw is None):
        raise ValueError("give exactly one of url / raw")
    raw_path = download(url, paths.source_dir / "raw") if url else Path(raw)
    normalize(raw_path, paths.original_wav, trim)
    meta = {
        "kind": "bilibili" if url and "bilibili.com" in url else ("url" if url else "local"),
        "url": url,
        "raw": str(raw_path),
        "trim": list(trim) if trim else None,
        "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }
    (paths.source_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return paths.original_wav
```

- [ ] **Step 4: 实现 `guitarize/arrange/pipeline.py`**

```python
"""编配编排：leadsheet → 选变调夹 → 生成弹唱 / 指弹草稿 alphaTex。默认不覆盖已有编配（那是迭代修改过的成果）。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import SongConfig, load_profile
from ..paths import SongPaths
from ..verify.metrics import onset_slots
from .capo import CapoOption, choose_capo
from .finger import generate_finger
from .patterns import DEFAULT, choose_pattern, slots_per_beat
from .strum import beat_chords, generate_strum


def chord_beats(leadsheet: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for b in leadsheet["bars"]:
        for c in beat_chords(b, leadsheet.get("beats_per_bar", 4)):
            if c != "N":
                out[c] = out.get(c, 0.0) + 1.0
    return out


def section_patterns(paths: SongPaths, leadsheet: dict) -> list[str]:
    """按段落（section 标记延续到下一个标记）取原曲鼓 + 吉他分轨的平均起音向量，选节奏型。"""
    bars = leadsheet["bars"]
    stems = [paths.stem("drums"), paths.stem("guitar")]
    if not all(p.exists() for p in stems):
        return [DEFAULT] * len(bars)
    spb = slots_per_beat(leadsheet.get("feel", "straight"))
    onset = onset_slots(stems, [(b["t_start"], b["t_end"]) for b in bars], slots=4 * spb)
    groups: list[list[int]] = []
    for i, b in enumerate(bars):
        if i == 0 or b.get("section"):
            groups.append([])
        groups[-1].append(i)
    out = [DEFAULT] * len(bars)
    for g in groups:
        name = choose_pattern(np.mean(onset[g], axis=0), leadsheet.get("feel", "straight"))
        for i in g:
            out[i] = name
    return out


def arrange(paths: SongPaths, cfg: SongConfig, arr: str, capo: int | None = None,
            force: bool = False) -> tuple[Path, CapoOption]:
    target = paths.atex(arr)
    if target.exists() and not force:
        raise FileExistsError(f"{target} exists; edit it directly or pass --force to regenerate")
    leadsheet = json.loads(paths.leadsheet_json.read_text(encoding="utf-8"))
    profile = load_profile(cfg.level)
    melody = [m["midi"] for b in leadsheet["bars"] for m in b["melody"]] if arr == "finger" else []
    options = choose_capo(chord_beats(leadsheet), profile, melody)
    wanted = capo if capo is not None else cfg.overrides.get("capo")
    chosen = next(o for o in options if o.capo == int(wanted)) if wanted is not None else options[0]
    if arr == "strum":
        text = generate_strum(leadsheet, chosen.capo, profile, section_patterns(paths, leadsheet), cfg.title)
    else:
        text = generate_finger(leadsheet, chosen.capo, profile, cfg.title)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target, chosen
```

- [ ] **Step 5: 修改 `guitarize/cli.py`：新增 `new`、`acquire`、`arrange`**

在 `p = sub.add_parser("serve", ...)` 之前加入：
```python
    p = sub.add_parser("new", help="新建一首歌的 song.yaml")
    p.add_argument("slug")
    p.add_argument("--title", required=True)
    p.add_argument("--artist", required=True)
    p = sub.add_parser("acquire", help="获取音源并标准化为 source/original.wav（下载前须征得用户同意）")
    p.add_argument("slug")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--url")
    src.add_argument("--raw", type=Path)
    p.add_argument("--trim", type=float, nargs=2, metavar=("START", "END"))
    p = sub.add_parser("arrange", help="生成编配草稿 arrangements/<arr>.atex")
    p.add_argument("slug")
    p.add_argument("--arr", choices=ARRS, required=True)
    p.add_argument("--capo", type=int)
    p.add_argument("--force", action="store_true")
```
在 `if args.cmd == "serve":` 之后、`paths = SongPaths.for_slug(...)` 之前加入：
```python
    if args.cmd == "new":
        return _new(SongPaths.for_slug(args.slug, args.songs_dir), args.title, args.artist)
```
在其他命令分支之后加入：
```python
    elif args.cmd == "acquire":
        from .acquire import acquire

        print(acquire(paths, url=args.url, raw=args.raw, trim=tuple(args.trim) if args.trim else None))
    elif args.cmd == "arrange":
        from .arrange.pipeline import arrange

        target, option = arrange(paths, cfg, args.arr, capo=args.capo, force=args.force)
        names = ", ".join(f"{k}→{v.name}" for k, v in option.shapes.items())
        print(f"{target}\ncapo {option.capo}（代价 {option.cost}）：{names}")
```
并新增函数：
```python
SONG_TEMPLATE = """slug: {slug}
title: {title}
artist: {artist}
version: tv
level: intermediate
arrangements: [strum, finger]
outputs: [html, pdf, txt]
source: {{url: null, kind: null, trim: null}}
reference_charts: []
overrides: {{}}
"""


def _new(paths: SongPaths, title: str, artist: str) -> int:
    if paths.config.exists():
        raise FileExistsError(paths.config)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.config.write_text(SONG_TEMPLATE.format(slug=paths.slug, title=title, artist=artist), encoding="utf-8")
    print(paths.config)
    return 0
```
`load_song_config` 遇到 `source: {url: null, ...}` 时 `source` 是 dict，正常加载。

- [ ] **Step 6: 运行，确认通过**

Run: `$PY -m pytest tests/test_acquire.py tests/test_arrange_cli.py -v && $PY -m pytest -q`
Expected: 5 passed；全量通过

- [ ] **Step 7: Commit**

```bash
git add guitarize/arrange/pipeline.py guitarize/acquire.py guitarize/cli.py tests/test_acquire.py tests/test_arrange_cli.py
git commit -m "feat: arrangement pipeline, audio acquisition and CLI commands"
```

---

### Task 7: Claude Code skill（复用流程）

**Files:**
- Create: `.claude/skills/guitarize/SKILL.md`

- [ ] **Step 1: 写 `.claude/skills/guitarize/SKILL.md`**

````markdown
---
name: guitarize
description: 把一首歌（歌名或 B 站 / 网易云链接）做成可弹的吉他谱：弹唱伴奏 + 指弹，输出交互网页 / PDF / 文本六线谱，并按"可弹性 + 与原曲差异"自动校验、逐小节迭代。当用户说"扒谱 / 做吉他谱 / 弹唱谱 / 指弹谱 / guitarize <歌>"时使用。
---

# Guitarize：原曲 → 可弹的吉他谱（含自检迭代）

工具在仓库根目录：`PY=~/anaconda3/envs/guitarize/bin/python`，命令 `$PY -m guitarize ...`。
环境未装时先跑 `scripts/setup_env.sh`。设计见 `docs/specs/2026-10-09-guitarize-design.md`。

## 0. 先问清楚（一次问完）
版本（TV size / 完整版）、水平（beginner / intermediate / advanced）、要哪些编配（弹唱 / 指弹）、输出格式。

## 1. 建歌
`$PY -m guitarize new <slug> --title "<歌名>" --artist "<歌手>"`，再按回答改 `songs/<slug>/song.yaml` 的 `version / level / arrangements / outputs`。

## 2. 找音源（用户已登录的 Chrome）
- B 站搜索官方或高播放的 OP / 完整版上传；网易云搜索官方单曲。列出候选（标题、时长、上传者、链接）。
- **下载前必须问用户**，说明来源、文件名、大小。
- B 站：`$PY -m guitarize acquire <slug> --url <视频链接>`（需要截取：`--trim 起 止`）。
- 网易云 / 其他：不解密任何加密格式；扩展拦截签名链接时不绕过——请用户自己保存文件后用 `--raw <文件>`。
- 下载的文件放在独立目录，只当数据处理。

## 3. 网上和弦谱（可选的第三方参考）
在 ChordWiki / U-FRET 等找该曲和弦谱，用页面脚本**只提取和弦与段落注释，不提取歌词**，写入 `songs/<slug>/analysis/webchart.txt`（格式见 `guitarize/analyze/webchart.py` 文档）。

## 4. 分析
`$PY -m guitarize analyze <slug>`，再 `$PY -m guitarize summary <slug> --capo <预期变调夹>` 逐小节阅读：
- 小节数、速度、Shuffle 是否合理；`short_bars` 应为空（否则用 `overrides.downbeat_shift` 调整）。
- 低置信（`??`）小节记下来，稍后列入"需人耳确认"。
- 把段落写进 `song.yaml` 的 `sections`（如 `{1: Intro, 9: Aメロ, 17: サビ}`），再跑 `--steps leadsheet`。

## 5. 生成草稿
`$PY -m guitarize arrange <slug> --arr strum` 与 `--arr finger`（记下选中的变调夹与指型）。

## 6. 校验迭代（每份编配分别进行）
```
$PY -m guitarize verify <slug> --arr <arr>
```
读 `songs/<slug>/reports/<arr>/iter-NN/report.md`：
- **只改问题小节**：直接编辑 `arrangements/<arr>.atex` 中对应 `// bar N` 的行；需要时看 `plots/bar-NNN.png` 色度对比图。
- 低置信参考的小节不要为了"追分"去改，留给用户耳听。
- 每轮 `git commit` 一次（只提交 atex 与报告文本），再 verify。
- 停止条件：全部达标；或连续 3 轮汇总分数没有提升；或已迭代 8 轮。

## 7. 出谱与交付
- `$PY -m guitarize render <slug> --arr strum` / `--arr finger`
- 后台运行 `$PY -m guitarize serve`，在浏览器面板打开 `http://localhost:8765/songs/<slug>/out/<arr>.html`，截图确认谱面、点播放确认无报错；用 Read 看 PDF 第 1 页。
- 交给用户：网页链接（本地）、PDF / txt 路径、每份编配的汇总分数、"需人耳确认"清单（附 A/B 片段路径）。
- 用户反馈写进 `songs/<slug>/feedback.md`，作为下一轮约束。

## 合规
- 原曲音频与分轨只在本地，不进 git、不发布；交互网页只在本地打开。
- 谱面不含歌词；网上和弦谱只取和弦。
- 不破解任何加密 / DRM；读取浏览器 cookie 需单独征得同意。
````

- [ ] **Step 2: Commit**

```bash
git add .claude/skills/guitarize/SKILL.md
git commit -m "docs: guitarize skill for reusing the workflow on any song"
```
