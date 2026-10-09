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
            effects = ([f"ch {atex.quote(v.name)}"] if label != last else []) + ["su" if slots[k] == "U" else "sd"]
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
