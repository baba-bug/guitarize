"""调性估计：Krumhansl–Schmuckler 模板与平均色度做相关；再用和弦结果交叉确认（调内和弦拍数比例）。"""
from __future__ import annotations

import numpy as np

from .. import theory
from .webchart import POWER_RE

CHORD_KEY_MARGIN = 0.1  # 和弦推断的调比色度调的调内比例高出这么多才改用
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_STEPS = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10)}


def estimate_key(chroma_mean) -> tuple[str, str, float]:
    c = np.asarray(chroma_mean, dtype=float)
    best: tuple[str, str, float] | None = None
    for mode, profile in (("major", MAJOR), ("minor", MINOR)):
        for tonic in range(12):
            r = float(np.corrcoef(c, np.roll(profile, tonic))[0, 1])
            if best is None or r > best[2]:
                best = (theory.pc_name(tonic), mode, r)
    return best


def scale_pcs(tonic: str, mode: str) -> set[int]:
    root = theory.pc(tonic)
    return {(root + s) % 12 for s in _STEPS[mode]}


def parse_key(text: str) -> tuple[str, str]:
    """'B' → ('B', 'major')；'G#m' → ('G#', 'minor')。"""
    t = text.strip()
    if t.endswith("m"):
        return theory.pc_name(theory.pc(t[:-1])), "minor"
    return theory.pc_name(theory.pc(t)), "major"


def _diatonic(label: str, pcs: set[int]) -> bool | None:
    """N → None（不计）；强力和弦只看根音；其它取大 / 小三和弦（根、三、五音），全在音阶内才算调内。"""
    m = POWER_RE.match(label)
    if m:
        return theory.pc(m.group(1)) in pcs
    mm = theory.majmin(label)
    if mm == "N":
        return None
    root = theory.pc(mm.split(":")[0])
    third = 3 if mm.endswith(":min") else 4
    return {root, (root + third) % 12, (root + 7) % 12} <= pcs


def diatonic_fraction(segments: list[dict], tonic: str, mode: str) -> float:
    """和弦拍中（按 dur_beats 加权，N 不计）调内和弦所占比例；没有和弦拍时为 0。"""
    pcs = scale_pcs(tonic, mode)
    hit = total = 0
    for s in segments:
        d = _diatonic(s["label"], pcs)
        if d is not None:
            total += s["dur_beats"]
            hit += s["dur_beats"] * d
    return hit / total if total else 0.0


def _tonic_beats(segments: list[dict], tonic: str, mode: str) -> int:
    target = f"{tonic}:{'maj' if mode == 'major' else 'min'}"
    return sum(s["dur_beats"] for s in segments
               if not POWER_RE.match(s["label"]) and theory.majmin(s["label"]) == target)


def cross_check_key(chroma_key: tuple[str, str], segments: list[dict]) -> dict:
    """24 个大小调按 diatonic_fraction 打分，取最高者（关系大小调同分时取主和弦拍数多的）。
    比色度调的分数高 ≥ CHORD_KEY_MARGIN 时 used = True（旋律与 leadsheet 改用该调）。"""
    best = None
    for mode in ("major", "minor"):
        for p in range(12):
            tonic = theory.pc_name(p)
            rank = (diatonic_fraction(segments, tonic, mode), _tonic_beats(segments, tonic, mode))
            if best is None or rank > best[0]:
                best = (rank, tonic, mode)
    (score, _), tonic, mode = best
    chroma_score = diatonic_fraction(segments, *chroma_key)
    return {"tonic": tonic, "mode": mode, "score": round(score, 3), "chroma_key_score": round(chroma_score, 3),
            "used": bool(score >= chroma_score + CHORD_KEY_MARGIN - 1e-9)}


def chosen_key(chords: dict) -> tuple[str, str]:
    """旋律与 leadsheet 用的调：和弦交叉确认判定 used 时取和弦推断的调，否则取 chords.json 的 key（色度估计或 override）。"""
    kc = chords.get("key_from_chords")
    k = kc if kc and kc.get("used") else chords["key"]
    return k["tonic"], k["mode"]


def key_report(chords: dict, overridden: bool) -> str:
    k, kc = chords["key"], chords["key_from_chords"]
    chord_key = f"{kc['tonic']} {kc['mode']}"
    if overridden:
        return f"key {k['tonic']} {k['mode']} (overrides.key); chords suggest {chord_key} (diatonic {kc['score']:.2f})"
    chroma = f"{k['tonic']} {k['mode']} (chroma r={k['r']:.2f})"
    if kc["used"]:
        return (f"key {chord_key} from chords: diatonic {kc['score']:.2f} vs {kc['chroma_key_score']:.2f} for {chroma}"
                f" → used for melody and leadsheet")
    return (f"key {chroma}; chords' best key {chord_key} diatonic {kc['score']:.2f} vs {kc['chroma_key_score']:.2f}"
            f" (needs +{CHORD_KEY_MARGIN:g} to switch)")
