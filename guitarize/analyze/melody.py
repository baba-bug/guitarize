"""人声旋律：torchcrepe 基频 → 按节拍网格（Shuffle 用三连音格）量化成音符 → 调内吸附、八度纠错。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .beats import time_to_beat

SR = 16000
HOP = 160
VOICED_PERIODICITY = 0.45
SNAP_SEMITONES = 0.75


def vocal_pitch(vocals_wav: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import librosa
    import torch
    import torchcrepe

    y, _ = librosa.load(str(vocals_wav), sr=SR, mono=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    f0, per = torchcrepe.predict(torch.tensor(y)[None], SR, HOP, 65.0, 1000.0, "full",
                                 return_periodicity=True, batch_size=1024, device=device)
    f0 = f0[0].cpu().numpy()
    per = torchcrepe.filter.median(per.cpu(), 3)[0].numpy()
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0][: len(f0)]
    rms = np.pad(rms, (0, len(f0) - len(rms)))
    voiced = (per > VOICED_PERIODICITY) & (rms > 0.02 * rms.max())
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
    return [{"start_beats": round(n["_start"] * slot, 4), "dur_beats": round((n["_end"] - n["_start"]) * slot, 4),
             "midi": n["midi"], "out_of_key": n["out_of_key"], "conf": round(float(np.mean(n["_conf"])), 3)}
            for n in notes]


def fix_octaves(notes: list[dict], window: int = 3) -> list[dict]:
    """与前后各 window 个音的中位数相差 ≥ 10 半音的音，按八度移向中位数。"""
    pitches = [n["midi"] for n in notes]
    out = []
    for i, n in enumerate(notes):
        neighbors = pitches[max(0, i - window):i] + pitches[i + 1:i + 1 + window]
        if neighbors:
            d = n["midi"] - float(np.median(neighbors))
            if abs(d) >= 10:
                n = {**n, "midi": n["midi"] - 12 * int(np.sign(d)) * int(round(abs(d) / 12))}
        out.append(n)
    return out


def extract_melody(vocals_wav: Path, beats_json: dict, key_pcs: set[int]) -> dict:
    import librosa

    times, midi, per = vocal_pitch(vocals_wav)
    y, _ = librosa.load(str(vocals_wav), sr=SR, mono=True)
    onsets = librosa.onset.onset_detect(y=y, sr=SR, hop_length=HOP, units="time")
    slot = 1 / 3 if beats_json.get("feel") == "shuffle" else 1 / 4
    notes = quantize(time_to_beat(beats_json, times), midi, per, time_to_beat(beats_json, onsets), key_pcs, slot)
    notes = [n for n in fix_octaves(notes) if n["start_beats"] >= 0]
    return {"grid": "triplet8" if slot == 1 / 3 else "sixteenth", "notes": notes}
