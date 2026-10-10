"""谱面模型：把 Node 端导出的 JSON 变成带演奏时间（已应用 alphaTab 的成对摇摆）的事件序列。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

TPQ = 960  # alphaTab 每四分音符的 tick 数


@dataclass(frozen=True)
class Note:
    string: int  # 1 = 最高音弦
    fret: int    # 相对变调夹
    midi: int    # 含变调夹的实际音高
    tie: bool
    dead: bool


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


@dataclass(frozen=True)
class Bar:
    index: int     # 0-based
    start: float   # 拍
    length: float  # 拍
    section: str | None
    feel: str


@dataclass(frozen=True)
class Attack:
    """同一时刻新拨响的所有音（跨声部合并，不含连音线延续）。"""
    start: float
    bar: int
    notes: tuple[Note, ...]
    strum: bool               # 带扫弦 / 刷弦标记
    direction: str | None     # "down" / "up" / None


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
            out.append(Attack(
                start=start,
                bar=evs[0].bar,
                notes=tuple(n for e in evs for n in e.notes if not n.tie),
                strum=direction is not None,
                direction=direction,
            ))
        return out

    def sounding(self) -> list[Sounding]:
        """每个音的实际发声区间；连音线把时值并入前一个同弦同声部的音。"""
        out: list[Sounding] = []
        open_notes: dict[tuple[int, int], int] = {}
        for e in sorted(self.events, key=lambda ev: (ev.start, ev.voice)):
            for n in e.notes:
                if n.dead:
                    continue
                key = (e.voice, n.string)
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


def load_score(path: Path) -> Score:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    bars: list[Bar] = []
    events: list[Event] = []
    for b in data["bars"]:
        bar_start = b["start"] / TPQ
        feel = b["triplet_feel"]
        bars.append(Bar(b["index"], bar_start, b["duration"] / TPQ, b["section"], feel))
        for vi, voice in enumerate(b["voices"]):
            raw = [(bt["start"] / TPQ, bt["duration"] / TPQ, bt["tuplet"]) for bt in voice]
            for bt, (rs, rd, _), (ps, pd) in zip(voice, raw, performed_times(raw, feel)):
                notes = tuple(Note(n["string"], n["fret"], n["midi"], n["tie"], n["dead"]) for n in bt["notes"])
                events.append(Event(b["index"], vi, bar_start + ps, pd, bar_start + rs, rd, notes,
                                    bt["chord"], bt["pick"], bt["brush"], bt["rest"], bt["tuplet"], bt.get("text")))
    return Score(data["title"], float(data["tempo"]), int(data["capo"]), list(data["tuning"]), bars, events)
