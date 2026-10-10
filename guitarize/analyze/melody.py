"""人声旋律：torchcrepe 基频 → 按节拍网格（Shuffle 用三连音格）量化成音符 → 调内吸附、八度纠错。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .beats import time_to_beat

SR = 16000
HOP = 160
VOICED_PERIODICITY = 0.45
SNAP_SEMITONES = 0.75
GLIDE_MAX_BEATS = 1 / 3 + 1e-3
GLIDE_MAX_SEMITONES = 2
OUT_OF_KEY_GUARD = 0.3   # 调外音比例超过它 → 调性多半错了，不做滑音合并


def vocal_pitch(vocals_wav: Path, decoder: str = "viterbi",
                voiced_periodicity: float = VOICED_PERIODICITY) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """人声基频（帧时间, MIDI 浮点音高或 nan, 周期性）。decoder："viterbi"（平滑，整体稳定，但会把很短的大跳
    抹成无声）或 "weighted_argmax"（逐帧，保留短促的音，抖动也多）。"""
    import librosa
    import torch
    import torchcrepe

    y, _ = librosa.load(str(vocals_wav), sr=SR, mono=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    f0, per = torchcrepe.predict(torch.tensor(y)[None], SR, HOP, 65.0, 1000.0, "full",
                                 decoder=getattr(torchcrepe.decode, decoder),
                                 return_periodicity=True, batch_size=1024, device=device)
    f0 = f0[0].cpu().numpy()
    per = torchcrepe.filter.median(per.cpu(), 3)[0].numpy()
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0][: len(f0)]
    rms = np.pad(rms, (0, len(f0) - len(rms)))
    voiced = (per > voiced_periodicity) & (rms > 0.02 * rms.max())
    midi = np.where(voiced, 69 + 12 * np.log2(np.maximum(f0, 1e-6) / 440.0), np.nan)
    return np.arange(len(f0)) * HOP / SR, midi, per


def snap_to_key(m: float, key_pcs: set[int]) -> tuple[int, bool]:
    """浮点音高 → 整数音高：离调内音 ≤ 0.75 半音则吸附；否则四舍五入并标记调外。"""
    cands = [p for p in range(int(np.floor(m)) - 1, int(np.ceil(m)) + 2) if p % 12 in key_pcs]
    if cands:
        best = min(cands, key=lambda p: abs(p - m))
        if abs(best - m) <= SNAP_SEMITONES:
            return best, False
    r = int(round(m))
    return r, r % 12 not in key_pcs


def quantize(beat_pos: np.ndarray, midi: np.ndarray, periodicity: np.ndarray, onset_beats: np.ndarray,
             key_pcs: set[int], slot: float) -> list[dict]:
    """逐格（slot 拍）取有声帧音高中位数；同音高且该格无起音时并入前一个音。"""
    beat_pos = np.asarray(beat_pos, dtype=float)
    onset_beats = np.asarray(onset_beats, dtype=float)
    if beat_pos.size == 0:
        return []
    k0 =int(np.floor(beat_pos.min() / slot))
    k1 = int(np.ceil(beat_pos.max() / slot))
    notes: list[dict] = []
    cur = None
    for k in range(k0, k1):
        s = k * slot
        m = (beat_pos >= s - 1e-9) & (beat_pos < s + slot - 1e-9)
        vals = midi[m]
        voiced = vals[~np.isnan(vals)]
        if m.sum() == 0 or len(voiced) < 0.5 * m.sum():
            cur = None
            continue
        p, out = snap_to_key(float(np.median(voiced)), key_pcs)
        onset = bool(np.any(np.abs(onset_beats - s) < slot / 2))
        conf = float(np.mean(periodicity[m]))
        if cur is not None and cur["midi"] == p and not onset and cur["_end"] == k:
            cur["_end"] = k + 1
            cur["_conf"].append(conf)
        else:
            cur = {"_start": k, "_end": k + 1, "midi": p, "out_of_key": out, "_conf": [conf]}
            notes.append(cur)
    # 起点、终点各自取 4 位小数，时值 = 终点 − 起点：三连音格上分别舍入起点和时值会让相邻音重叠 1e-4
    return [{"start_beats": round(n["_start"] * slot, 4), "dur_beats": _span(n["_start"] * slot, n["_end"] * slot),
             "midi": n["midi"], "out_of_key": n["out_of_key"], "conf": round(float(np.mean(n["_conf"])), 3)}
            for n in notes]


def _span(start: float, end: float) -> float:
    return round(round(end, 4) - round(start, 4), 4)


def fix_octaves(notes: list[dict], window: int = 3) -> list[dict]:
    """与前后各 window 个音的中位数相差 ≥ 11.5 半音的音（10–11 半音视为合法大跳，保留），按八度移向中位数。"""
    pitches = [n["midi"] for n in notes]
    out = []
    for i, n in enumerate(notes):
        neighbors = pitches[max(0, i - window):i] + pitches[i + 1:i + 1 + window]
        if neighbors:
            d = n["midi"] - float(np.median(neighbors))
            if abs(d) >= 11.5:
                n = {**n, "midi": n["midi"] - 12 * int(np.sign(d)) * int(round(abs(d) / 12))}
        out.append(n)
    return out


def merge_glides(notes: list[dict]) -> list[dict]:
    """调外短音（≤ 1/3 拍）若紧贴一个相距 ≤ 2 半音的调内音，视为滑音/倚音并入该音：
    优先并入紧随其后的音（起点提前到滑音起点），否则并入紧挨在前的音（时值延长）。
    其它调外音保留（可能是真正的经过音或蓝调音）。"""
    def ok(g: dict, t: dict) -> bool:
        return not t["out_of_key"] and abs(g["midi"] - t["midi"]) <= GLIDE_MAX_SEMITONES

    work = [dict(n) for n in notes]
    out: list[dict] = []
    for i, n in enumerate(work):
        if n["out_of_key"] and n["dur_beats"] <= GLIDE_MAX_BEATS:
            nxt = work[i + 1] if i + 1 < len(work) else None
            if nxt is not None and abs(n["start_beats"] + n["dur_beats"] - nxt["start_beats"]) <= 1e-3 and ok(n, nxt):
                nxt["dur_beats"] = _span(n["start_beats"], nxt["start_beats"] + nxt["dur_beats"])
                nxt["start_beats"] = round(n["start_beats"], 4)
                continue
            prev = out[-1] if out else None
            if prev is not None and abs(prev["start_beats"] + prev["dur_beats"] - n["start_beats"]) <= 1e-3 and ok(n, prev):
                prev["dur_beats"] = _span(prev["start_beats"], n["start_beats"] + n["dur_beats"])
                continue
        out.append(n)
    return out


def tidy_notes(notes: list[dict]) -> list[dict]:
    """八度纠错 → 去掉第 1 小节之前的音 → 合并滑音。滑音合并依赖调内 / 调外判断：
    合并前调外音超过 30% 说明调性多半错了，此时跳过合并并警告。"""
    notes = [n for n in fix_octaves(notes) if n["start_beats"] >= 0]
    out = sum(n["out_of_key"] for n in notes) / len(notes) if notes else 0.0
    if out > OUT_OF_KEY_GUARD:
        print(f"[melody] warning: {100 * out:.0f}% of notes are out of key (> {100 * OUT_OF_KEY_GUARD:.0f}%): "
              "the key is probably wrong; glide merging skipped", file=sys.stderr)
        return notes
    return merge_glides(notes)


FILL_PERIODICITY = 0.6    # 补空隙用的逐帧音轨的有声阈值
FILL_CONF = 0.7           # 补进来的音平均周期性至少这么高
FILL_MAX_STD = 0.5        # 且音高稳定（帧间标准差 < 半个半音；滑音不算）


def fill_gaps(main: list[dict], alt: list[dict], stable) -> list[dict]:
    """main：Viterbi 解码量化出的音（平滑可靠，但会把短促的大跳抹成无声，如 なんもねえ 第 1 小节的 B3）；
    alt：逐帧解码量化出的音。只把落在 main 空隙里（与 main 任何音都不重叠）、平均周期性 ≥ FILL_CONF、
    且 stable(音) 为真的 alt 音补进来。返回按开始时间排好的新列表，不修改输入。"""
    def overlaps(n: dict) -> bool:
        a, b = n["start_beats"], n["start_beats"] + n["dur_beats"]
        return any(m["start_beats"] < b - 1e-6 and m["start_beats"] + m["dur_beats"] > a + 1e-6 for m in main)

    extra = [dict(n) for n in alt if n["conf"] >= FILL_CONF and not overlaps(n) and stable(n)]
    return sorted([dict(n) for n in main] + extra, key=lambda n: n["start_beats"])


def extract_melody(vocals_wav: Path, beats_json: dict, key_pcs: set[int]) -> dict:
    import librosa

    y, _ = librosa.load(str(vocals_wav), sr=SR, mono=True)
    onsets = time_to_beat(beats_json, librosa.onset.onset_detect(y=y, sr=SR, hop_length=HOP, units="time"))
    slot = 1 / 3 if beats_json.get("feel") == "shuffle" else 1 / 4
    times, midi, per = vocal_pitch(vocals_wav)
    beat_pos = time_to_beat(beats_json, times)
    main = quantize(beat_pos, midi, per, onsets, key_pcs, slot)
    times_w, midi_w, per_w = vocal_pitch(vocals_wav, "weighted_argmax", FILL_PERIODICITY)
    beat_w = time_to_beat(beats_json, times_w)
    alt = quantize(beat_w, midi_w, per_w, onsets, key_pcs, slot)

    def stable(n: dict) -> bool:
        m = (beat_w >= n["start_beats"]) & (beat_w < n["start_beats"] + n["dur_beats"])
        v = midi_w[m]
        v = v[~np.isnan(v)]
        return len(v) >= 4 and float(np.std(v)) < FILL_MAX_STD

    notes = tidy_notes(fill_gaps(main, alt, stable))
    return {"grid": "triplet8" if slot == 1 / 3 else "sixteenth", "notes": notes}
