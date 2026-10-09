"""节拍与小节线：beat_this 拍点 → 补漏拍 / 去多余拍 → 按 overrides.tempo_bpm 纠正倍速 / 半速
→ HMM 定强拍（4/4，允许个别短小节）→ beats.json（含速度稳定性）。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from . import features

BEAT_FPS = 50.0          # beat_this 的输出帧率
SHORT_BAR_PENALTY = 3.0  # 2 或 3 拍短小节的罚分
BEATS_PER_BAR = 4
OCTAVE_TOL = 0.08        # 检测速度 / 指定速度 在 2 或 1/2 的 ±8% 内 → 判为倍速 / 半速
TEMPO_WARN = 0.04        # 与指定速度相差超过 4% → 警告（不改拍点）
STABILITY_WINDOW = 16    # 局部速度的滑动窗口（拍距个数）
STABILITY_WARN = 0.06    # 局部速度极差 / 速度 > 6% → 警告


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


def positions_with_shift(evidence: Sequence[float], shift: int = 0,
                         beats_per_bar: int = BEATS_PER_BAR) -> list[int]:
    """shift = 0 时即 bar_positions。shift ≠ 0 时改用严格 4 拍循环（禁止短小节，罚分极大），
    再整体旋转 shift 位，避免在含短小节的序列上旋转破坏小节结构。"""
    if shift == 0:
        return bar_positions(evidence, beats_per_bar)
    pos = bar_positions(evidence, beats_per_bar, short_penalty=1e9)
    return [(p + shift) % beats_per_bar for p in pos]


def _est_bpm(t: np.ndarray) -> float:
    return 60.0 / float(np.median(np.diff(t)))


def _keep_phase(activation: Sequence[float] | None) -> int:
    """隔拍删拍时保留哪一相位：拍 / 强拍激活总和更大的一组；平分时取第一个最强拍所在相位；没有激活时取第一拍。"""
    if activation is None:
        return 0
    a = np.asarray(activation, dtype=float)
    s0, s1 = float(a[0::2].sum()), float(a[1::2].sum())
    if abs(s0 - s1) > 1e-9:
        return int(s1 > s0)
    return int(np.argmax(a)) % 2


def apply_tempo_override(beats: Sequence[float], override_bpm: float | None,
                         activation: Sequence[float] | None = None) -> tuple[np.ndarray, list[str]]:
    """按 overrides.tempo_bpm 纠正拍点。ratio = 检测速度 / 指定速度：
    ≈ 2（±8%）→ 隔拍删去（保留 _keep_phase 选中的相位）；≈ 1/2 → 相邻两拍之间插入中点；
    其它情况拍点不动，相差 > 4% 时给出警告。返回 (拍点, 提示信息)。"""
    t = np.asarray(beats, dtype=float)
    if override_bpm is None or len(t) < 3:
        return t, []
    target = float(override_bpm)
    est = _est_bpm(t)
    ratio = est / target
    notes: list[str] = []
    if abs(ratio / 2 - 1) <= OCTAVE_TOL:
        phase = _keep_phase(activation)
        t = t[phase::2]
        notes.append(f"detected {est:.1f} BPM ≈ 2 × overrides.tempo_bpm {target:g}: kept every other beat "
                     f"(phase {phase})")
    elif abs(ratio / 0.5 - 1) <= OCTAVE_TOL:
        t = np.sort(np.r_[t, (t[:-1] + t[1:]) / 2])
        notes.append(f"detected {est:.1f} BPM ≈ 1/2 × overrides.tempo_bpm {target:g}: inserted midpoints")
    off = abs(_est_bpm(t) - target) / target
    if off > TEMPO_WARN:
        kept = "corrected beats kept" if notes else "beats kept as detected"
        notes.append(f"warning: beat tempo {_est_bpm(t):.1f} BPM differs from overrides.tempo_bpm {target:g} "
                     f"by {100 * off:.1f}%; {kept}")
    return t, notes


def tempo_stability(beats: Sequence[float], window: int = STABILITY_WINDOW) -> dict:
    """滑动窗口（window 个拍距）内的平均速度 → 局部速度的最小 / 最大值与变异系数。拍不够一个窗口时用全长。"""
    t = np.asarray(beats, dtype=float)
    w = min(window, len(t) - 1)
    local = 60.0 * w / (t[w:] - t[:-w])
    return {"local_min_bpm": round(float(local.min()), 2), "local_max_bpm": round(float(local.max()), 2),
            "cv": round(float(local.std() / local.mean()), 4)}


def stability_warning(stability: dict, tempo_bpm: float) -> str | None:
    spread = (stability["local_max_bpm"] - stability["local_min_bpm"]) / tempo_bpm
    if spread <= STABILITY_WARN:
        return None
    return (f"warning: tempo unstable: local tempo {stability['local_min_bpm']:g}–{stability['local_max_bpm']:g} "
            f"BPM ({100 * spread:.1f}% of {tempo_bpm:g}); a fixed-tempo score will drift from the recording")


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
    # 弱起小节（pad > 0）不算短小节；末尾小节天然不完整，也不算。
    first_idx = 1 if pad else 0
    short = [b["bar"] for b in bars[first_idx:-1] if b["beats"] < beats_per_bar]
    return {
        "tempo_bpm": round(60.0 / ibi, 2),
        "tempo_stability": tempo_stability(t),
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
    beat_logits, down_logits = Audio2Frames(checkpoint_path="final0", device="cpu")(signal, sr)
    beat_prob = torch.sigmoid(torch.as_tensor(beat_logits)).numpy()
    down_prob = torch.sigmoid(torch.as_tensor(down_logits)).numpy()
    beats = clean_beats(raw)
    # 先纠正倍速 / 半速，之后的强拍证据、小节线、律动都在纠正后的拍上计算
    beats, notes = apply_tempo_override(beats, overrides.get("tempo_bpm"),
                                        window_max(beat_prob, beats) + window_max(down_prob, beats))
    harmonic = features.load_mix([Path(stems_dir) / "harmonic.wav"])
    novelty = features.chroma_novelty(features.beat_sync_chroma(harmonic, beats))
    env, et = features.onset_envelope(features.load_mix([Path(stems_dir) / "bass.wav"]))
    bass_onset = features.sample_max(env, et, beats, 0.05)
    evidence = downbeat_evidence(window_max(down_prob, beats), novelty, bass_onset)
    positions = positions_with_shift(evidence, int(overrides.get("downbeat_shift", 0)))
    feel = overrides.get("feel") or features.detect_feel(features.load_mix([Path(stems_dir) / "drums.wav"]), beats)
    out = assemble_beats(beats, positions, feel)
    notes.append(stability_warning(out["tempo_stability"], out["tempo_bpm"]))
    for note in filter(None, notes):
        print(f"[beats] {note}", file=sys.stderr)
    return out
