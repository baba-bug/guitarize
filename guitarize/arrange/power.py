"""强力和弦弹唱（arr = power）：照原曲的两把失真吉他，用强力和弦按实际音高弹（不夹变调夹）。

- 和弦：按根音换成强力和弦 X5（斜杠和弦按和弦根音、不按低音：D#/G → D#5，B/D# → B5；小三、七和弦一样是 X5），
  无和弦（N）休止。谱上的和弦名就是 X5，每个指型画和弦图。
- 指型：两指的可移动指型，根音在 6 弦或 5 弦，加五音和高八度：6 弦 f 品是 (f.6, f+2.5, f+2.4)，5 弦是
  (f.5, f+2.4, f+2.3)；f = 0 就是开放的 E5 (0.6 2.5 2.4) / A5 (0.5 2.4 2.3)。根音在 0–9 品，每个强力和弦全曲
  一个指型，选使换和弦时把位移动总量最小的一套（choose_shapes）。
- 节奏型（patterns.POWER_PATTERNS）：按段落、只看原曲吉他分轨的起音选（rhythm.choose_patterns，与弹唱的 guitar
  节奏共用）。overrides.power_patterns 可以逐段指定。
- 记谱 / 播放：扫弦都写刷弦 bd / bu，力度逐下写（下扫 f、上扫和轻扫 mp），空弦重音加 ac、闷音写 pm（音符效果），
  见 patterns.STROKES。延音直接写长时值，跨拍、八分对切分处用连音线接上。"""
from __future__ import annotations

import itertools
from collections import Counter

from .. import theory
from ..config import Profile
from . import atex
from .capo import effective_profile
from .layout import annotate, pattern_remarks
from .patterns import POWER_DEFAULT, POWER_PATTERNS, stroke_effects
from .shapes import Voicing, playable
from .strum import bar_slots, beat_chords

ROOT_STRINGS = {6: 40, 5: 45}   # 根音所在的弦 → 空弦音高（E2、A2）
MAX_ROOT_FRET = 9               # 根音在 0–9 品
LOW_FRET = 0.01                 # 把位移动一样多时取低把位（每品的微小代价）


def power_name(label: str) -> str | None:
    """和弦标签 → 强力和弦名（按和弦根音，斜杠和弦不按低音）；无和弦是 None。"""
    parsed = theory.parse_chord(label)
    return None if parsed is None else theory.pc_name(parsed[0]) + "5"


def position(v: Voicing) -> int:
    """指型的把位：最低的按弦品位（与 verify 的换把规则同一个定义；开放 E5 / A5 是 2）。"""
    return min((f for f in v.frets if f), default=0)


def power_shapes(name: str, profile: Profile) -> list[Voicing]:
    """强力和弦名 → 可弹的指型（根音在 6 / 5 弦的 0–9 品，五音和高八度在相邻两根弦的同一品），低把位在前。
    难度档弹不了任何一个时 ValueError。"""
    root = theory.pc(name.removesuffix("5"))
    profile = effective_profile(profile, 0)
    out = []
    for string, open_midi in ROOT_STRINGS.items():
        fret = (root - open_midi) % 12
        frets = [None] * 6
        frets[string - 1], frets[string - 2], frets[string - 3] = fret, fret + 2, fret + 2
        if fret <= MAX_ROOT_FRET and playable(tuple(frets), profile):
            out.append(Voicing(name, tuple(frets), 0.0))
    if not out:
        raise ValueError(f"no playable power chord for {name!r} at level {profile.name}")
    return sorted(out, key=position)


def _moves(chords: list[list[str]]) -> Counter:
    """相邻两个不同强力和弦的换和弦次数 {(前, 后): 次数}；跨过休止（N）不算（休止时手有空换把）。"""
    moves: Counter = Counter()
    last = None
    for bar in chords:
        for c in bar:
            if c == "N":
                last = None
                continue
            if last is not None and c != last:
                moves[(last, c)] += 1
            last = c
    return moves


def movement(chords: list[list[str]], shapes: dict[str, Voicing]) -> int:
    """全曲换和弦时的把位移动总量（品）。"""
    return sum(k * abs(position(shapes[a]) - position(shapes[b])) for (a, b), k in _moves(chords).items())


def choose_shapes(chords: list[list[str]], cands: dict[str, list[Voicing]]) -> dict[str, Voicing]:
    """每个强力和弦定一个指型（全曲一致，和弦图只出现一次），使换和弦时的把位移动总量最小；一样多时取低把位。
    chords：逐小节逐拍的强力和弦名（N = 无和弦）。每个根音在 6、5 弦的 0–9 品上至多各一个指型，最多 2^12 种组合：
    直接穷举（顺序确定）。"""
    moves = _moves(chords)
    names = list(cands)
    best = None
    for pick in itertools.product(*(cands[n] for n in names)):
        chosen = dict(zip(names, pick))
        cost = (sum(k * abs(position(chosen[a]) - position(chosen[b])) for (a, b), k in moves.items())
                + LOW_FRET * sum(position(v) for v in pick))
        if best is None or cost < best[0] - 1e-9:
            best = (cost, chosen)
    return best[1] if best else {}


# ---------- 记谱 ----------

def _notes(v: Voicing) -> list[tuple[int, int]]:
    return [(i + 1, f) for i, f in enumerate(v.frets) if f is not None]


def bar_tokens(chords: list[str], pattern: str, shapes: dict[str, Voicing], prev: str | None,
               feel: str = "straight") -> tuple[list[str], str | None]:
    """一小节：chords 是逐拍强力和弦名（4 个，N = 无和弦）。节奏型的格见 strum.bar_slots（无和弦的格休止，换和弦或休止
    之后的延续格改为下扫 D）；每下扫弦写刷弦方向、力度和音符效果（patterns.stroke_effects），延续格并进前一下（让它接着响）。
    指型名标在每小节第一下扫弦和每次换和弦处。Shuffle 下反拍扫弦按八分对切分（atex.voice_bar_tokens）。"""
    slot_chord = [chords[k // 2] for k in range(8)]
    slots = bar_slots(chords, pattern)
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
            name = slot_chord[k]
            notes = _notes(shapes[name])
            beat, effect = stroke_effects(slots[k])
            mark = name != last or not any(it.notes for it in items)   # 每小节第一下都标（和弦没变也标）
            effects = ([f"ch {atex.quote(name)}"] if mark else []) + beat
            items.append(atex.Item(k * 0.5, n * 0.5, notes, effects,
                                   head_effects={s: [effect] for s, _ in notes} if effect else None))
            last = name
        k += n
    return atex.voice_bar_tokens(items, feel), last


def generate_power(leadsheet: dict, profile: Profile, patterns: list[str | None], title: str) -> str:
    """patterns：逐小节的节奏型名（rhythm.choose_patterns；整小节无和弦是 None）。每个乐句（节奏型变了、或新段落）的
    第一小节在 `// bar N` 注释里写上节奏型名（layout.pattern_remarks）。"""
    feel = leadsheet.get("feel", "straight")
    bars = leadsheet["bars"]
    chords = [[power_name(c) or "N" for c in beat_chords(b, leadsheet.get("beats_per_bar", 4))] for b in bars]
    names = list(dict.fromkeys(c for bar in chords for c in bar if c != "N"))   # 按首次出现排序
    shapes = choose_shapes(chords, {n: power_shapes(n, profile) for n in names})
    lines: list[str] = []
    prev = None
    for i, b in enumerate(bars):
        pattern = patterns[i] or POWER_DEFAULT   # 整小节无和弦：节奏型无所谓，全是休止
        tokens, prev = bar_tokens(chords[i], POWER_PATTERNS[pattern], shapes, prev, feel)
        lines.append(atex.bar_line(tokens, i + 1, b.get("section")))
    diagrams = {v.name: v.frets for v in shapes.values()}
    head = atex.header(title, leadsheet["tempo_bpm"], 0, feel, "tabs", diagrams, key=atex.leadsheet_key(leadsheet))
    return atex.document(head, [annotate(lines, pattern_remarks(leadsheet, patterns))])
