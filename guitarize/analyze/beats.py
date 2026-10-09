"""节拍与小节线：beat_this 拍点 → 补漏拍 / 去多余拍 → HMM 定强拍（4/4，允许个别短小节）→ beats.json。"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from . import features

BEAT_FPS = 50.0          # beat_this 的输出帧率
SHORT_BAR_PENALTY = 3.0  # 2 或 3 拍短小节的罚分
BEATS_PER_BAR = 4


def clean_beats(beats: Sequence[float]) -> np.ndarray:
    """去掉间隔 < 0.5 倍拍距中位数的多余拍；间隔约为 k 倍拍距时均匀补上 k−1 个漏拍。"""
    t = np.asarray(beats, dtype=float)
    if len(t) < 3:
        return t
    med = float(np.median(np.diff(t)))
    out = [t[0]]
    for b in t[1:]:
        gap = b - out[-1]
        if gap < 0.5 * med:
            continue
        k = max(1, int(round(gap / med)))
        start = out[-1]
        out.extend(start + gap * j / k for j in range(1, k))
        out.append(b)
    return np.array(out)


def bar_positions(evidence: Sequence[float], beats_per_bar: int = BEATS_PER_BAR,
                  short_penalty: float = SHORT_BAR_PENALTY) -> list[int]:
    """每拍在小节内的位置（0 = 强拍）。Viterbi：正常 p → p+1；从 1..bpb−2 直接回到 0 构成短小节（罚分）。
    第一拍可以处在任意位置（弱起）。强拍位置获得 evidence 分。"""
    ev = np.asarray(evidence, dtype=float)
    n, p_count = len(ev), beats_per_bar
    neg = -1e18
    score = np.full((n, p_count), neg)
    back = np.zeros((n, p_count), dtype=int)
    score[0] = [ev[0] if p == 0 else 0.0 for p in range(p_count)]
    for i in range(1, n):
        for p in range(p_count):
            prev = (p - 1) % p_count
            best, arg = score[i - 1, prev], prev
            if p == 0:
                for q in range(1, p_count - 1):
                    cand = score[i - 1, q] - short_penalty
                    if cand > best:
                        best, arg = cand, q
            score[i, p] = best + (ev[i] if p == 0 else 0.0)
            back[i, p] = arg
    path = [int(score[-1].argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    return path[::-1]


def window_max(prob: np.ndarray, times: Sequence[float], fps: float = BEAT_FPS, half: float = 0.05) -> np.ndarray:
    out = np.zeros(len(times))
    for i, t in enumerate(times):
        a = max(0, int((t - half) * fps))
        b = max(a + 1, int((t + half) * fps) + 1)
        seg = prob[a:b]
        out[i] = float(seg.max()) if len(seg) else 0.0
    return out


def _unit(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = x.max() if len(x) else 0.0
    return x / m if m > 0 else x


def downbeat_evidence(down_prob_at_beats, novelty, bass_onset) -> np.ndarray:
    """强拍证据 = 2 × beat_this 强拍概率 + 1 × 和声变化（归一）+ 0.5 × 贝斯起音（归一）。"""
    return 2.0 * np.asarray(down_prob_at_beats, dtype=float) + 1.0 * _unit(novelty) + 0.5 * _unit(bass_onset)


def assemble_beats(beat_times: Sequence[float], positions: Sequence[int], feel: str,
                   beats_per_bar: int = BEATS_PER_BAR) -> dict:
    t = np.asarray(beat_times, dtype=float)
    ibi = float(np.median(np.diff(t)))
    pad = int(positions[0])
    pickup = list(positions).index(0) if (pad and 0 in positions) else 0
    beats = []
    bar = 1 if pad else 0
    for i, (ti, p) in enumerate(zip(t, positions)):
        if p == 0:
            bar += 1
        beats.append({"t": round(float(ti), 4), "bar": bar, "beat": int(p) + 1, "g": i + pad})
    firsts: dict[int, dict] = {}
    counts: dict[int, int] = {}
    for b in beats:
        firsts.setdefault(b["bar"], b)
        counts[b["bar"]] = counts.get(b["bar"], 0) + 1
    numbers = sorted(firsts)
    bars = []
    for k, n in enumerate(numbers):
        first = firsts[n]
        t_start = first["t"] - (first["beat"] - 1) * ibi
        t_end = firsts[numbers[k + 1]]["t"] if k + 1 < len(numbers) else beats[-1]["t"] + ibi
        bars.append({"bar": n, "t_start": round(t_start, 4), "t_end": round(t_end, 4), "beats": counts[n]})
    short = [b["bar"] for b in bars[1:-1] if b["beats"] < beats_per_bar]
    return {
        "tempo_bpm": round(60.0 / ibi, 2),
        "feel": feel,
        "time_signature": [beats_per_bar, 4],
        "pickup_beats": pickup,
        "pad_beats": pad,
        "short_bars": short,
        "beats": beats,
        "bars": bars,
    }


def _axis(beats_json: dict) -> tuple[np.ndarray, np.ndarray]:
    g = np.array([b["g"] for b in beats_json["beats"]], dtype=float)
    t = np.array([b["t"] for b in beats_json["beats"]], dtype=float)
    return g, t


def _interp_extrap(x, xp: np.ndarray, fp: np.ndarray):
    x = np.asarray(x, dtype=float)
    y = np.interp(x, xp, fp)
    lo = (fp[1] - fp[0]) / (xp[1] - xp[0])
    hi = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    y = np.where(x < xp[0], fp[0] + (x - xp[0]) * lo, y)
    return np.where(x > xp[-1], fp[-1] + (x - xp[-1]) * hi, y)


def time_to_beat(beats_json: dict, t):
    g, ts = _axis(beats_json)
    return _interp_extrap(t, ts, g)


def beat_to_time(beats_json: dict, beat):
    g, ts = _axis(beats_json)
    return _interp_extrap(beat, g, ts)


def track_beats(mix_wav: Path, stems_dir: Path, overrides: dict | None = None) -> dict:
    import torch
    from beat_this.inference import Audio2Frames, File2Beats, load_audio

    overrides = overrides or {}
    raw, _ = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)(str(mix_wav))
    signal, sr = load_audio(str(mix_wav))
    _, down_logits = Audio2Frames(checkpoint_path="final0", device="cpu")(signal, sr)
    down_prob = torch.sigmoid(torch.as_tensor(down_logits)).numpy()
    beats = clean_beats(raw)
    harmonic = features.load_mix([Path(stems_dir) / "harmonic.wav"])
    novelty = features.chroma_novelty(features.beat_sync_chroma(harmonic, beats))
    env, et = features.onset_envelope(features.load_mix([Path(stems_dir) / "bass.wav"]))
    bass_onset = features.sample_max(env, et, beats, 0.05)
    positions = bar_positions(downbeat_evidence(window_max(down_prob, beats), novelty, bass_onset))
    shift = int(overrides.get("downbeat_shift", 0))
    if shift:
        positions = [(p + shift) % BEATS_PER_BAR for p in positions]
    feel = overrides.get("feel") or features.detect_feel(features.load_mix([Path(stems_dir) / "drums.wav"]), beats)
    return assemble_beats(beats, positions, feel)
