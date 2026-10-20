"""强力和弦弹唱（arr = power）：照原曲的两把失真吉他，用强力和弦按实际音高弹（不夹变调夹）。

- 和弦：按根音换成强力和弦 X5（斜杠和弦按和弦根音、不按低音：D#/G → D#5，B/D# → B5；小三、七和弦一样是 X5），
  无和弦（N）休止。谱上的和弦名就是 X5，每个指型画和弦图。
- 指型：两指的可移动指型，根音在 6 弦或 5 弦，加五音和高八度：6 弦 f 品是 (f.6, f+2.5, f+2.4)，5 弦是
  (f.5, f+2.4, f+2.3)；f = 0 就是开放的 E5 (0.6 2.5 2.4) / A5 (0.5 2.4 2.3)。根音在 0–9 品，每个强力和弦全曲
  一个指型，选使换和弦时把位移动总量最小的一套（choose_shapes）。
- 节奏型（patterns.POWER_PATTERNS）：按段落、只看原曲吉他分轨的起音选（鼓的起音强得多，混在一起选出来的是鼓的节奏型）；
  段落在无和弦小节处断开，段落内前后两半差得多时切一刀（なんもねえ 的 Bメロ），起音没有可辨认节奏的段落是一击
  （choose_patterns）。overrides.power_patterns 可以逐段指定。
- 记谱 / 播放：扫弦都写刷弦 bd / bu（alphaTab 只有刷弦才把各弦错开起音，sd / su 只是箭头），力度逐下写（下扫 f、
  上扫和轻扫 mp），空弦重音加 ac（音符效果），闷音写 pm（音符效果；写成拍效果 {pm} 是语法错误）——alphaTab 按
  80/60 拍的固定短时值播放闷音。延音直接写长时值，跨拍、八分对切分处用连音线接上。"""
from __future__ import annotations

import itertools
from collections import Counter

import numpy as np

from .. import theory
from ..config import Profile
from . import atex
from .capo import effective_profile
from .layout import annotate
from .patterns import POWER_DEFAULT, POWER_FLAT, POWER_PATTERNS, choose_pattern, correlations
from .shapes import Voicing, playable
from .strum import bar_slots, beat_chords

ROOT_STRINGS = {6: 40, 5: 45}   # 根音所在的弦 → 空弦音高（E2、A2）
MAX_ROOT_FRET = 9               # 根音在 0–9 品
LOW_FRET = 0.01                 # 把位移动一样多时取低把位（每品的微小代价）
BRUSH = 60                      # 刷弦把各弦错开的总时长（tick，♩ = 960；167 BPM 下 3 根弦相隔约 11 毫秒）
# 节奏型的记号 → (刷弦方向, 力度, 音符效果)
STROKES = {"A": ("bd", "f", "ac"), "D": ("bd", "f", None), "M": ("bd", "f", "pm"),
           "d": ("bd", "mp", None), "U": ("bu", "mp", None)}
MIN_CORR = 0.5      # 最像的节奏型与段落起音的相关度也不到它：没有可辨认的扫弦节奏（持续的音墙 / 余音）→ 一击
MIN_PART = 3        # 段落内切一刀时每段至少几小节
SPLIT_GAIN = 0.05   # 切开后（按小节数加权的）相关度至少比不切高这么多才切


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


# ---------- 节奏型 ----------

def phrases(leadsheet: dict) -> list[list[int]]:
    """选节奏型的乐句（小节下标，0 起）：段落（标记延续到下一个标记），再在整小节无和弦处断开——无和弦小节只有休止，
    不在任何一组里（なんもねえ 間奏的 riff 前的停顿把一击和 riff 分开）。"""
    groups: list[list[int]] = []
    current: list[int] = []
    for i, b in enumerate(leadsheet["bars"]):
        if b.get("section") and current:
            groups.append(current)
            current = []
        if all(power_name(c) is None for c in beat_chords(b, leadsheet.get("beats_per_bar", 4))):
            if current:
                groups.append(current)
            current = []
            continue
        current.append(i)
    return groups + ([current] if current else [])


def _best(rows: np.ndarray, feel: str) -> tuple[str, float]:
    """一组小节的平均起音向量 → (节奏型, 最像的型的相关度)；相关度不到 MIN_CORR 时是一击（POWER_FLAT）。"""
    mean = rows.mean(axis=0)
    cors = correlations(mean, feel, POWER_PATTERNS)
    if not cors:
        return POWER_FLAT, 0.0
    name = choose_pattern(mean, feel, POWER_PATTERNS, POWER_DEFAULT)
    return (name if cors[name] >= MIN_CORR else POWER_FLAT), cors[name]


def _split(group: list[int], onset: np.ndarray, feel: str) -> list[tuple[list[int], str]]:
    """一个乐句选型：前后两半（各至少 MIN_PART 小节）选出的型不同、且按小节数加权的相关度比不切至少高 SPLIT_GAIN 时
    切成两段（取提高最多的切点，只切一刀）。"""
    whole, r = _best(onset[group], feel)
    best = None
    for k in range(MIN_PART, len(group) - MIN_PART + 1):
        (head, ra), (tail, rb) = _best(onset[group[:k]], feel), _best(onset[group[k:]], feel)
        gain = (k * ra + (len(group) - k) * rb) / len(group) - r
        if head != tail and gain >= SPLIT_GAIN - 1e-9 and (best is None or gain > best[0]):
            best = (gain, k, head, tail)
    if best is None:
        return [(group, whole)]
    _, k, head, tail = best
    return [(group[:k], head), (group[k:], tail)]


def choose_patterns(onset: np.ndarray | None, leadsheet: dict,
                    overrides: dict[int, str] | None = None) -> list[str | None]:
    """逐小节的节奏型（整小节无和弦是 None）。onset：原曲吉他分轨逐小节的起音向量（metrics.onset_slots，每拍
    slots_per_beat 格），None = 没有分轨，全用 POWER_DEFAULT。overrides（config.power_patterns）：{小节(1 起): 型}，
    从该小节起用到它所在乐句的末尾或下一个覆盖起点。"""
    feel = leadsheet.get("feel", "straight")
    out: list[str | None] = [None] * len(leadsheet["bars"])
    groups = phrases(leadsheet)
    for group in groups:
        for idx, name in ([(group, POWER_DEFAULT)] if onset is None else _split(group, onset, feel)):
            for i in idx:
                out[i] = name
    starts = {bar - 1: name for bar, name in (overrides or {}).items()}
    for start, name in sorted(starts.items()):
        group = next((g for g in groups if start in g), None)
        if group is None:
            raise ValueError(f"overrides.power_patterns: bar {start + 1} has no chord (or is out of range)")
        for i in group[group.index(start):]:
            if i != start and i in starts:
                break
            out[i] = name
    return out


# ---------- 记谱 ----------

def _notes(v: Voicing) -> list[tuple[int, int]]:
    return [(i + 1, f) for i, f in enumerate(v.frets) if f is not None]


def bar_tokens(chords: list[str], pattern: str, shapes: dict[str, Voicing], prev: str | None,
               feel: str = "straight") -> tuple[list[str], str | None]:
    """一小节：chords 是逐拍强力和弦名（4 个，N = 无和弦）。节奏型的格见 strum.bar_slots（无和弦的格休止，换和弦或休止
    之后的延续格改为下扫 D）；每下扫弦写刷弦方向、力度和音符效果（STROKES），延续格并进前一下（让它接着响）。
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
            brush, dynamic, effect = STROKES[slots[k]]
            mark = name != last or not any(it.notes for it in items)   # 每小节第一下都标（和弦没变也标）
            effects = ([f"ch {atex.quote(name)}"] if mark else []) + [f"{brush} {BRUSH}", f"dy {dynamic}"]
            items.append(atex.Item(k * 0.5, n * 0.5, notes, effects,
                                   head_effects={s: [effect] for s, _ in notes} if effect else None))
            last = name
        k += n
    return atex.voice_bar_tokens(items, feel), last


def generate_power(leadsheet: dict, profile: Profile, patterns: list[str | None], title: str) -> str:
    """patterns：逐小节的节奏型名（choose_patterns；整小节无和弦是 None）。每个乐句（节奏型变了、或新段落）的第一小节
    在 `// bar N` 注释里写上节奏型名。"""
    feel = leadsheet.get("feel", "straight")
    bars = leadsheet["bars"]
    chords = [[power_name(c) or "N" for c in beat_chords(b, leadsheet.get("beats_per_bar", 4))] for b in bars]
    names = list(dict.fromkeys(c for bar in chords for c in bar if c != "N"))   # 按首次出现排序
    shapes = choose_shapes(chords, {n: power_shapes(n, profile) for n in names})
    lines: list[str] = []
    remarks: dict[int, list[str]] = {}
    prev, shown = None, None
    for i, b in enumerate(bars):
        pattern = patterns[i] or POWER_DEFAULT   # 整小节无和弦：节奏型无所谓，全是休止
        tokens, prev = bar_tokens(chords[i], POWER_PATTERNS[pattern], shapes, prev, feel)
        lines.append(atex.bar_line(tokens, i + 1, b.get("section")))
        if patterns[i] is not None and (patterns[i] != shown or b.get("section")):
            remarks[i] = [f"节奏型 {patterns[i]}"]
        shown = patterns[i]
    diagrams = {v.name: v.frets for v in shapes.values()}
    head = atex.header(title, leadsheet["tempo_bpm"], 0, feel, "tabs", diagrams, key=atex.leadsheet_key(leadsheet))
    return atex.document(head, [annotate(lines, remarks)])
