"""音名、和弦标签、移调、大小三和弦推断。"""
from __future__ import annotations

import re

import numpy as np

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_ALIASES = {"DB": 1, "EB": 3, "GB": 6, "AB": 8, "BB": 10, "CB": 11, "FB": 4, "E#": 5, "B#": 0}
_CHORD_RE = re.compile(r"^([A-Ga-g])([#b♯♭]?)([^/]*)(?:/([A-Ga-g][#b♯♭]?))?$")
_NO_CHORD = {"", "N", "N.C.", "NC", "-"}

# (后缀正则, 规范化性质)，按顺序匹配
_QUALITIES = [
    (r"^(maj7|M7|Δ)", "maj7"),
    (r"^(m7-5|m7b5|ø)", "min7b5"),
    (r"^(dim|o)", "dim"),
    (r"^(aug|\+)", "aug"),
    (r"^(m7|min7|-7)", "min7"),
    (r"^(m(?!aj)|min|-)", "min"),
    (r"^sus2", "sus2"),
    (r"^sus", "sus4"),
    (r"^(7|9|11|13)", "7"),
    (r"", "maj"),
]
_INTERVALS = {
    "maj": (0, 4, 7), "min": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10), "dim": (0, 3, 6), "min7b5": (0, 3, 6, 10), "aug": (0, 4, 8),
    "sus4": (0, 5, 7), "sus2": (0, 2, 7),
}
_MAJ_LIKE = {"maj", "maj7", "7", "aug", "sus4", "sus2"}


def pc(name: str) -> int:
    n = name.strip().replace("♯", "#").replace("♭", "b")
    key = n[0].upper() + n[1:]
    if key in PITCH_CLASSES:
        return PITCH_CLASSES.index(key)
    if key.upper() in _ALIASES:
        return _ALIASES[key.upper()]
    raise ValueError(f"bad note name {name!r}")


def pc_name(p: int) -> str:
    return PITCH_CLASSES[p % 12]


def midi_name(m: int) -> str:
    return f"{pc_name(m)}{m // 12 - 1}"


_NOTE_RE = re.compile(r"^([A-Ga-g])([#b♯♭]?)(-?[0-9])$")
_NATURAL = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACCIDENTAL = {"": 0, "#": 1, "♯": 1, "b": -1, "♭": -1}


def note_midi(name: str) -> int:
    """带八度的音名 → MIDI 音高（科学音高记法，C4 = 60）：'F#4' → 66、'Bb3' → 58；B#3 = C4、Cb4 = B3。
    写错（没有八度、重升号、不是字符串）→ ValueError。"""
    m = _NOTE_RE.match(name.strip()) if isinstance(name, str) else None
    if not m:
        raise ValueError(f"bad note name {name!r}")
    return 12 * (int(m.group(3)) + 1) + _NATURAL[m.group(1).upper()] + _ACCIDENTAL[m.group(2)]


def parse_chord(label: str) -> tuple[int, str, int | None] | None:
    s = label.strip()
    if s in _NO_CHORD:
        return None
    m = _CHORD_RE.match(s)
    if not m:
        raise ValueError(f"bad chord label {label!r}")
    root = pc(m.group(1) + m.group(2))
    quality = next(q for pat, q in _QUALITIES if re.match(pat, m.group(3)))
    bass = pc(m.group(4)) if m.group(4) else None
    return root, quality, bass


def majmin(label: str | None) -> str:
    parsed = None if label is None else parse_chord(label)
    if parsed is None:
        return "N"
    root, quality, _ = parsed
    return f"{pc_name(root)}:{'maj' if quality in _MAJ_LIKE else 'min'}"


def transpose_label(label: str, semis: int) -> str:
    s = label.strip()
    m = _CHORD_RE.match(s)
    if s in _NO_CHORD or not m:
        return label
    out = pc_name(pc(m.group(1) + m.group(2)) + semis) + m.group(3)
    if m.group(4):
        out += "/" + pc_name(pc(m.group(4)) + semis)
    return out


def chord_tones(label: str) -> set[int]:
    parsed = parse_chord(label)
    if parsed is None:
        return set()
    root, quality, bass = parsed
    tones = {(root + i) % 12 for i in _INTERVALS[quality]}
    if bass is not None:
        tones.add(bass)
    return tones


MAJMIN_LABELS = [f"{n}:{q}" for q in ("maj", "min") for n in PITCH_CLASSES]


def _template(label: str) -> np.ndarray:
    root = pc(label.split(":")[0])
    v = np.zeros(12)
    v[[(root + i) % 12 for i in ((0, 4, 7) if label.endswith("maj") else (0, 3, 7))]] = 1
    return v


TEMPLATES = np.stack([_template(lbl) for lbl in MAJMIN_LABELS])  # (24, 12)


def infer_majmin(weights, bass_pc: int | None = None, tol: float = 1e-6) -> set[str]:
    """按音名权重推断大小三和弦。得分 = 和弦音权重占比 + 低音为根音时 0.25；并列最优全部返回。"""
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    if total <= 0:
        return {"N"}
    scores = TEMPLATES @ w / total
    if bass_pc is not None:
        for i in range(len(MAJMIN_LABELS)):
            if i % 12 == bass_pc:
                scores[i] += 0.25
    best = scores.max()
    return {MAJMIN_LABELS[i] for i in np.flatnonzero(scores >= best - tol)}
