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


def section_bars(leadsheet: dict) -> list[int]:
    """带段落标记的小节（0-based）。"""
    return [i for i, b in enumerate(leadsheet["bars"]) if b.get("section")]


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
