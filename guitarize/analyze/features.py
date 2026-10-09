"""分析阶段共用的音频特征（采样率 22050，hop 512）。"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

SR = 22050
HOP = 512


def _cqt_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    """CQT 色度。librosa 对很短的输入会在内部低八度下采样时发出 n_fft 过大警告，此处屏蔽之。"""
    import librosa

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_fft=.* is too large")
        return librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)


def load_mix(wavs: Iterable[Path], sr: int = SR) -> np.ndarray:
    """读取并相加多个 wav（单声道、重采样到 sr），长度不同时补零。"""
    import librosa

    total = None
    for w in wavs:
        y, _ = librosa.load(str(w), sr=sr, mono=True)
        if total is None:
            total = y
        else:
            n = max(len(total), len(y))
            total = np.pad(total, (0, n - len(total))) + np.pad(y, (0, n - len(y)))
    if total is None:
        raise ValueError("load_mix needs at least one wav")
    return total


def beat_sync_chroma(y: np.ndarray, beat_times: Sequence[float], sr: int = SR) -> np.ndarray:
    """逐拍色度 (12, 拍数)：每拍区间内 CQT 色度的中位数，列单位化。最后一拍区间长度取拍距中位数。"""
    import librosa

    t = np.asarray(beat_times, dtype=float)
    chroma = _cqt_chroma(y, sr)
    step = float(np.median(np.diff(t))) if len(t) > 1 else 0.5
    edges = librosa.time_to_frames(np.append(t, t[-1] + step), sr=sr, hop_length=HOP)
    edges = np.clip(edges, 0, chroma.shape[1])
    cols = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = chroma[:, a:max(a + 1, b)]
        cols.append(np.median(seg, axis=1) if seg.shape[1] else np.zeros(12))
    out = np.stack(cols, axis=1)
    return out / (np.linalg.norm(out, axis=0, keepdims=True) + 1e-9)


def chroma_novelty(cb: np.ndarray) -> np.ndarray:
    """相邻两拍色度的差异（1 − 余弦）；第一拍记 1。"""
    if cb.shape[1] == 0:
        return np.zeros(0)
    return np.r_[1.0, 1.0 - np.sum(cb[:, 1:] * cb[:, :-1], axis=0)]


def onset_envelope(y: np.ndarray, sr: int = SR, hop: int = HOP) -> tuple[np.ndarray, np.ndarray]:
    import librosa

    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    return env, librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop)


def sample_max(env: np.ndarray, env_times: np.ndarray, times: Sequence[float], half_width: float) -> np.ndarray:
    """在每个时间点 ±half_width 秒内取包络最大值。"""
    out = np.zeros(len(times))
    for i, t in enumerate(times):
        m = (env_times >= t - half_width) & (env_times <= t + half_width)
        out[i] = env[m].max() if m.any() else 0.0
    return out


def detect_feel(drums: np.ndarray, beat_times: Sequence[float], sr: int = SR, ratio: float = 1.2) -> str:
    """比较拍间 1/2 处与 2/3 处的起音强度：2/3 处明显更强 → shuffle。"""
    t = np.asarray(beat_times, dtype=float)
    if len(t) < 3:
        return "straight"
    env, et = onset_envelope(drums, sr=sr, hop=256)
    ibi = np.diff(t)
    half = sample_max(env, et, t[:-1] + ibi / 2, 0.02)
    trip = sample_max(env, et, t[:-1] + ibi * 2 / 3, 0.02)
    return "shuffle" if trip.mean() > ratio * max(half.mean(), 1e-9) else "straight"


def window_chroma(y: np.ndarray, windows: Sequence[tuple[float, float]], sr: int = SR) -> np.ndarray:
    """(窗口数, 12)：每个时间窗内 CQT 色度均值，行单位化；空窗为 0。"""
    import librosa

    chroma = _cqt_chroma(y, sr)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=HOP)
    out = np.zeros((len(windows), 12))
    for i, (a, b) in enumerate(windows):
        m = (times >= a) & (times < b)
        if m.any():
            out[i] = chroma[:, m].mean(axis=1)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)
