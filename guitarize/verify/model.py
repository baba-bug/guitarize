"""谱面模型：把 Node 端导出的 JSON 变成带演奏时间（已应用 alphaTab 的成对摇摆）的事件序列。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

TPQ = 960  # alphaTab 每四分音符的 tick 数
# 力度 → MIDI 力度（alphaTab MidiUtils.dynamicToVelocity 同一张表）；重音（ac）+16，强重音（hac）+32
VELOCITY = {"PPP": 15, "PP": 31, "P": 47, "MP": 63, "MF": 79, "F": 95, "FF": 111, "FFF": 127}
ACCENT_STEP = 16


@dataclass(frozen=True)
class Note:
    string: int | None  # 1 = 最高音弦；音高谱表（卡祖笛等，没有弦）是 None
    fret: int | None    # 相对变调夹；音高谱表是 None
    midi: int    # 含变调夹的实际音高
    tie: bool
    dead: bool
    hammer: bool = False       # 击弦 / 勾弦的起点（alphaTab 找到了同弦的下一个音）
    slide: str | None = None   # 滑音：legato（连到同弦的下一个音）/ out_down / out_up；None = 没有
    slur_start: bool = False   # 连线（slur）的起点 / 终点
    slur_end: bool = False
    palm_mute: bool = False    # 闷音（pm）
    accent: int = 0            # 0 无，1 重音（ac），2 强重音（hac），3 保持音（不改力度）


@dataclass(frozen=True)
class Event:
    """某声部里的一拍（alphaTab 的 Beat）。"""
    bar: int              # 0-based
    voice: int
    start: float          # 演奏时间（拍，已应用摇摆）
    dur: float
    written_start: float  # 记谱位置（拍）
    written_dur: float
    notes: tuple[Note, ...]
    chord: str | None
    pick: str | None
    brush: str | None
    rest: bool
    tuplet: int = 1       # 连音组数（1 = 非连音）
    text: str | None = None   # 拍上的文字（txt），旋律谱用它写段落名
    dynamics: str | None = None   # 力度（"F"、"MP"……，见 VELOCITY）；None = 旧的模型 JSON 没有导出


@dataclass(frozen=True)
class Bar:
    index: int     # 0-based
    start: float   # 拍
    length: float  # 拍
    section: str | None
    feel: str
    key: int = 0                # 调号：升号数为正、降号数为负
    minor: bool = False
    ottava: str | None = None   # 谱号八度记号（8vb 等，只改显示）；None = 没有


@dataclass(frozen=True)
class Attack:
    """同一时刻新拨响的所有音（跨声部合并，不含连音线延续）。"""
    start: float
    bar: int
    notes: tuple[Note, ...]
    strum: bool               # 带扫弦 / 刷弦标记
    direction: str | None     # "down" / "up" / None
    velocity: int = VELOCITY["F"]   # 新拨的音里最大的 MIDI 力度（力度 + 重音，同 alphaTab）
    palm_mute: bool = False         # 新拨的音全是闷音


@dataclass(frozen=True)
class Sounding:
    start: float
    end: float
    midi: int
    voice: int


def performed_times(beats: list[tuple[float, float, int]], feel: str) -> list[tuple[float, float]]:
    """一个声部、一个小节内各拍的演奏时间：(小节内演奏起点, 演奏时值)，单位拍。

    对应 alphaTab MidiFileGenerator 的三连音感觉：只有「成对」的八分音符才摇摆——
    前一个是恰好一个八分、起点落在四分音符边界、且紧跟着同声部的下一拍也恰好是一个八分
    （休止符也算一拍）。这一对播成 2/3 + 1/3 拍；其余拍（切分、连音、三连音等）按记谱播放。
    输入 (记谱起点, 记谱时值, 连音组数)，起点为小节内偏移（拍）。
    """
    out = [(s, d) for s, d, _ in beats]
    if feel != "triplet8th":
        return out
    tol = 1e-6

    def is_eighth(i: int) -> bool:
        return beats[i][2] == 1 and abs(beats[i][1] - 0.5) < tol

    for i in range(len(beats) - 1):
        s = beats[i][0]
        if not (is_eighth(i) and is_eighth(i + 1)):
            continue
        if abs(s - round(s)) > tol or abs(beats[i + 1][0] - (s + 0.5)) > tol:
            continue
        out[i] = (s, 2 / 3)
        out[i + 1] = (s + 2 / 3, 1 / 3)
    return out


@dataclass
class Score:
    title: str
    tempo: float
    capo: int
    tuning: list[int]
    bars: list[Bar]
    events: list[Event]

    def seconds(self, beats: float) -> float:
        return beats * 60.0 / self.tempo

    def attacks(self) -> list[Attack]:
        groups: dict[float, list[Event]] = {}
        for e in self.events:
            if not e.rest and any(not n.tie for n in e.notes):
                groups.setdefault(round(e.start, 6), []).append(e)
        out = []
        for start in sorted(groups):
            evs = groups[start]
            direction = next((e.pick or e.brush for e in evs if e.pick or e.brush), None)
            fresh = [(e, n) for e in evs for n in e.notes if not n.tie]
            out.append(Attack(
                start=start,
                bar=evs[0].bar,
                notes=tuple(n for _, n in fresh),
                strum=direction is not None,
                direction=direction,
                velocity=max(velocity(e, n) for e, n in fresh),
                palm_mute=all(n.palm_mute for _, n in fresh),
            ))
        return out

    def sounding(self) -> list[Sounding]:
        """每个音的实际发声区间；连音线把时值并入前一个同弦同声部的音（音高谱表没有弦：同音高同声部）。"""
        out: list[Sounding] = []
        open_notes: dict[tuple[int, int], int] = {}
        for e in sorted(self.events, key=lambda ev: (ev.start, ev.voice)):
            for n in e.notes:
                if n.dead:
                    continue
                key = (e.voice, n.midi if n.string is None else n.string)
                if n.tie and key in open_notes:
                    i = open_notes[key]
                    s = out[i]
                    out[i] = Sounding(s.start, e.start + e.dur, s.midi, s.voice)
                else:
                    open_notes[key] = len(out)
                    out.append(Sounding(e.start, e.start + e.dur, n.midi, e.voice))
        return out

    def melody(self, voice: int = 0) -> list[Sounding]:
        """旋律：指定声部中每个起音时刻的最高音。"""
        best: dict[float, Sounding] = {}
        for s in self.sounding():
            if s.voice != voice:
                continue
            k = round(s.start, 6)
            if k not in best or s.midi > best[k].midi:
                best[k] = s
        return [best[k] for k in sorted(best)]

    def voice_lengths(self) -> dict[tuple[int, int], float]:
        """(小节, 声部) → 记谱总时值（拍），用于发现小节拍数写错。"""
        out: dict[tuple[int, int], float] = {}
        for e in self.events:
            out[(e.bar, e.voice)] = out.get((e.bar, e.voice), 0.0) + e.written_dur
        return out


def velocity(event: Event, note: Note) -> int:
    """一个音的 MIDI 力度：拍的力度（没有导出时按 alphaTab 的默认 f）+ 重音。"""
    accent = {1: 1, 2: 2}.get(note.accent, 0)
    return min(127, VELOCITY.get(event.dynamics or "F", VELOCITY["F"]) + ACCENT_STEP * accent)


def load_score(path: Path) -> Score:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    bars: list[Bar] = []
    events: list[Event] = []
    for b in data["bars"]:
        bar_start = b["start"] / TPQ
        feel = b["triplet_feel"]
        bars.append(Bar(b["index"], bar_start, b["duration"] / TPQ, b["section"], feel,
                        b.get("key_signature", 0), b.get("key_minor", False), b.get("ottava")))
        for vi, voice in enumerate(b["voices"]):
            raw = [(bt["start"] / TPQ, bt["duration"] / TPQ, bt["tuplet"]) for bt in voice]
            for bt, (rs, rd, _), (ps, pd) in zip(voice, raw, performed_times(raw, feel)):
                notes = tuple(Note(n["string"], n["fret"], n["midi"], n["tie"], n["dead"], n.get("hammer", False),
                                   n.get("slide"), n.get("slur_start", False), n.get("slur_end", False),
                                   n.get("palm_mute", False), n.get("accent", 0))
                              for n in bt["notes"])
                events.append(Event(b["index"], vi, bar_start + ps, pd, bar_start + rs, rd, notes,
                                    bt["chord"], bt["pick"], bt["brush"], bt["rest"], bt["tuplet"], bt.get("text"),
                                    bt.get("dynamics")))
    return Score(data["title"], float(data["tempo"]), int(data["capo"]), list(data["tuning"]), bars, events)
