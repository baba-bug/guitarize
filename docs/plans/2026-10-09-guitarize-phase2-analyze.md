# Guitarize Phase 2：分析（分轨 / 节拍 / 和弦投票 / 旋律 / leadsheet）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从 `source/original.wav`（加可选的网上和弦谱 `analysis/webchart.txt`）生成 Phase 1 校验器所需的参考 `analysis/leadsheet.json`，并用一首已知答案的合成歌曲做闭环验证。

**Architecture:** 五个可单独运行的步骤：stems（Demucs 分轨）→ beats（beat_this 拍点 + 补拍 + 小节线 HMM）→ chords（模板 HMM + 网上和弦谱 DP 对齐 + 逐拍投票）→ melody（torchcrepe 基频 + 节拍网格量化 + 调内吸附）→ leadsheet（合并）。每步产物落盘为 JSON，共用 `analyze/features.py` 中的音频特征。闭环测试用 pretty_midi 生成已知节拍/和弦/旋律的歌曲，经 FluidSynth 渲染为各声部 wav 后跑分析，检查还原精度。

**Tech Stack:** Python 3.11、numpy、scipy、librosa 0.11、torch 2.x（MPS）、demucs 4.1（htdemucs_6s）、beat_this（final0 checkpoint）、torchcrepe、pretty_midi、soundfile、pytest。

**Spec:** `docs/specs/2026-10-09-guitarize-design.md`（§4.3 分析、§5 数据格式、§8 闭环测试）

## 全局约束

- Python：`PY=~/anaconda3/envs/guitarize/bin/python`；测试在仓库根目录运行 `$PY -m pytest`。慢测试标记 `@pytest.mark.slow`。
- 拍轴：**全局拍号 g，第 1 小节第 1 拍 = 0**。有弱起时，弱起拍并入第 1 小节（前面补休止），第一个强拍所在小节为第 2 小节；`pad_beats` = 第 1 小节开头补的拍数。小节号 1-based。
- leadsheet.json 必须满足 Phase 1 校验器的读取方式：`bars[i]` 对应编配第 i 个小节（下标对齐），每小节有 `t_start/t_end/chords[{beat,dur_beats,label,confidence}]/melody[{start_beats,dur_beats,midi,conf}]`。
- 和弦标签一律用 ChordWiki 风格（`B`、`G#m`、`D#/G`、`C#m7`），**不得**输出 `C:maj` 这类冒号格式（Phase 1 的 `theory.parse_chord` 不认）。升号写法。
- 置信度规则（spec §4.3）：可用来源 N=2（模板 + 网上谱）：一致 → `high`，不一致 → `low`；N=1（只有模板）：后验 ≥ 0.7 → `medium`，否则 `low`。网上谱该小节对齐相似度 < 0.5 时视为不可用（N=1）。网上谱为 N.C./只有贝斯 → 标签 `N`、`medium`。
- 强力和弦（`B5`）只比较根音；与音频一致时，标签取网上谱的根音 + 音频的大小三。
- 旋律量化网格：Shuffle → 1/3 拍，直拍 → 1/4 拍；离调内音 ≤ 0.75 半音时吸附到调内音。
- 原曲音频与分轨不入 git；网上和弦谱只保存和弦，不保存歌词。
- 模型权重已缓存：`~/.cache/torch/hub/checkpoints/beat_this-final0.ckpt`；Demucs htdemucs_6s 首次运行会自动下载。
- 相对 spec 的范围取舍（已在控制台账中裁决）：① spec §4.3 的第三方"预训练和弦识别模型"本期不接入，投票只有模板 + 网上谱两方（N ≤ 2），置信度规则照 spec 的 N=2 / N=1 分支；② 不实现按小节色度自相似矩阵找段落——网上谱 DP 对齐已给出段落结构，段落名由主会话写进 `song.yaml` 的 `sections`。

## 文件结构（Phase 2）

```
guitarize/analyze/
  __init__.py        （空）
  features.py        共用音频特征：load_mix / beat_sync_chroma / chroma_novelty / onset_envelope / sample_max / detect_feel / window_chroma
  key.py             estimate_key / scale_pcs / parse_key
  beats.py           clean_beats / bar_positions / window_max / downbeat_evidence / assemble_beats / time_to_beat / beat_to_time / track_beats
  webchart.py        WebBar / POWER_RE / normalize_label / parse_webchart
  chords.py          template_track / chord_tones_any / web_bar_vector / align_webchart / web_label_at / same_chord / vote_beat / reference_chords
  melody.py          vocal_pitch / snap_to_key / quantize / fix_octaves / extract_melody
  stems.py           STEMS / mix_harmonic / separate
  leadsheet.py       build_leadsheet / summary_table
  pipeline.py        STEPS / run_analysis
guitarize/cli.py     （修改）新增 analyze / summary 子命令
tests/
  synth_song.py      闭环用合成歌曲（非测试文件）
  test_features.py test_key.py test_beats.py test_webchart.py test_chords.py test_melody.py test_leadsheet.py
  test_closed_loop.py  （slow）
```

---

### Task 1: 共用特征与调性

**Files:**
- Create: `guitarize/analyze/__init__.py`（空）, `guitarize/analyze/features.py`, `guitarize/analyze/key.py`
- Test: `tests/test_features.py`, `tests/test_key.py`

**Interfaces:**
- Consumes: `theory.pc / pc_name`（Phase 1）
- Produces: `features.SR = 22050`、`features.HOP = 512`、`load_mix(wavs) -> np.ndarray`、`beat_sync_chroma(y, beat_times) -> (12, n)`（列单位化）、`chroma_novelty(cb) -> (n,)`、`onset_envelope(y, sr=SR, hop=HOP) -> (env, times)`、`sample_max(env, env_times, times, half_width) -> (n,)`、`detect_feel(drums_y, beat_times) -> "shuffle"|"straight"`、`window_chroma(y, windows) -> (n, 12)`（行单位化）；`key.estimate_key(chroma_mean) -> (tonic, mode, r)`、`key.scale_pcs(tonic, mode) -> set[int]`、`key.parse_key(text) -> (tonic, mode)`

- [ ] **Step 1: 写失败测试**

`tests/test_features.py`：
```python
import numpy as np
import soundfile as sf

from guitarize.analyze import features

SR = features.SR


def tone(midis, dur, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    return sum(np.sin(2 * np.pi * 440 * 2 ** ((m - 69) / 12) * t) for m in midis) / len(midis)


def test_load_mix_pads_and_sums(tmp_path):
    sf.write(tmp_path / "a.wav", np.ones(100) * 0.1, SR)
    sf.write(tmp_path / "b.wav", np.ones(150) * 0.2, SR)
    y = features.load_mix([tmp_path / "a.wav", tmp_path / "b.wav"])
    assert len(y) == 150
    assert np.allclose(y[:100], 0.3, atol=1e-3) and np.allclose(y[100:], 0.2, atol=1e-3)


def test_beat_sync_chroma_follows_chords():
    y = np.concatenate([tone([60, 64, 67], 1.0), tone([57, 60, 64], 1.0)])
    cb = features.beat_sync_chroma(y, [0.0, 0.5, 1.0, 1.5])
    assert cb.shape == (12, 4)
    assert np.allclose(np.linalg.norm(cb, axis=0), 1.0, atol=1e-6)
    assert set(np.argsort(cb[:, 0])[-3:]) == {0, 4, 7}
    assert set(np.argsort(cb[:, 3])[-3:]) == {9, 0, 4}
    nov = features.chroma_novelty(cb)
    assert nov[0] == 1.0 and nov[2] > nov[1] and nov[2] > nov[3]


def clicks(times, amps, dur, sr=SR):
    y = np.zeros(int(dur * sr))
    burst = np.random.default_rng(0).standard_normal(int(0.01 * sr)) * np.exp(-np.linspace(0, 8, int(0.01 * sr)))
    for t, a in zip(times, amps):
        i = int(t * sr)
        y[i:i + len(burst)] += a * burst[: len(y) - i]
    return y


def test_detect_feel():
    beats = np.arange(0.2, 8.0, 0.5)
    on = list(beats)
    shuffle = clicks(on + list(beats + 0.5 * 2 / 3), [1.0] * len(beats) + [0.6] * len(beats), 8.5)
    straight = clicks(on + list(beats + 0.25), [1.0] * len(beats) + [0.6] * len(beats), 8.5)
    assert features.detect_feel(shuffle, beats) == "shuffle"
    assert features.detect_feel(straight, beats) == "straight"


def test_window_chroma_rows_unit():
    y = tone([62, 66, 69], 2.0)
    w = features.window_chroma(y, [(0.0, 1.0), (1.0, 2.0), (5.0, 6.0)])
    assert w.shape == (3, 12)
    assert set(np.argsort(w[0])[-3:]) == {2, 6, 9}
    assert np.allclose(w[2], 0.0)
```

`tests/test_key.py`：
```python
import numpy as np

from guitarize.analyze.key import MAJOR, MINOR, estimate_key, parse_key, scale_pcs


def test_estimate_key():
    assert estimate_key(np.roll(MAJOR, 11))[:2] == ("B", "major")
    assert estimate_key(np.roll(MINOR, 9))[:2] == ("A", "minor")


def test_scale_and_parse():
    assert scale_pcs("B", "major") == {11, 1, 3, 4, 6, 8, 10}
    assert scale_pcs("A", "minor") == {9, 11, 0, 2, 4, 5, 7}
    assert parse_key("B") == ("B", "major")
    assert parse_key("G#m") == ("G#", "minor")
    assert parse_key("Bb") == ("A#", "major")
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest tests/test_features.py tests/test_key.py -v`
Expected: FAIL（`ModuleNotFoundError: guitarize.analyze`）

- [ ] **Step 3: 实现 `guitarize/analyze/features.py`**

```python
"""分析阶段共用的音频特征（采样率 22050，hop 512）。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

SR = 22050
HOP = 512


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
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
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

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=HOP)
    out = np.zeros((len(windows), 12))
    for i, (a, b) in enumerate(windows):
        m = (times >= a) & (times < b)
        if m.any():
            out[i] = chroma[:, m].mean(axis=1)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)
```

- [ ] **Step 4: 实现 `guitarize/analyze/key.py`**

```python
"""调性估计：Krumhansl–Schmuckler 模板与平均色度做相关。"""
from __future__ import annotations

import numpy as np

from .. import theory

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
```

- [ ] **Step 5: 运行，确认通过**

Run: `$PY -m pytest tests/test_features.py tests/test_key.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add guitarize/analyze tests/test_features.py tests/test_key.py
git commit -m "feat: shared audio features and key estimation"
```

---

### Task 2: 节拍与小节线

**Files:**
- Create: `guitarize/analyze/beats.py`
- Test: `tests/test_beats.py`

**Interfaces:**
- Consumes: `features`（Task 1）
- Produces:
  - `clean_beats(beats) -> np.ndarray`、`bar_positions(evidence, beats_per_bar=4, short_penalty=3.0) -> list[int]`（每拍小节内位置，0 = 强拍）
  - `window_max(prob, times, fps=50.0, half=0.05) -> np.ndarray`、`downbeat_evidence(down_prob, novelty, bass_onset) -> np.ndarray`
  - `assemble_beats(beat_times, positions, feel, beats_per_bar=4) -> dict`（beats.json）
  - beats.json：`{tempo_bpm, feel, time_signature:[4,4], pickup_beats, pad_beats, short_bars:[...], beats:[{t, bar, beat, g}], bars:[{bar, t_start, t_end, beats}]}`
  - `time_to_beat(beats_json, t)`、`beat_to_time(beats_json, g)`（两端线性外推，接受标量或数组）
  - `track_beats(mix_wav, stems_dir, overrides=None) -> dict`（需要 stems_dir 下的 harmonic.wav、bass.wav、drums.wav；`overrides` 支持 `downbeat_shift: int`、`feel: "shuffle"|"straight"`）

- [ ] **Step 1: 写失败测试 `tests/test_beats.py`**

```python
import numpy as np
import pytest

from guitarize.analyze.beats import assemble_beats, bar_positions, beat_to_time, clean_beats, time_to_beat


def test_clean_beats_fills_and_drops():
    out = clean_beats([0, 0.5, 1.0, 2.0, 2.1, 2.5])
    assert out.tolist() == pytest.approx([0, 0.5, 1.0, 1.5, 2.0, 2.5])


def test_bar_positions_with_pickup():
    ev = np.zeros(16)
    ev[[3, 7, 11, 15]] = 1.0
    assert bar_positions(ev) == [1, 2, 3, 0] * 4


def test_bar_positions_allows_strong_short_bar():
    ev = np.zeros(14)
    ev[[0, 4, 8, 10]] = 5.0
    pos = bar_positions(ev)
    assert pos[8] == 0 and pos[9] == 1 and pos[10] == 0 and pos[13] == 3


def test_assemble_beats_pickup_bar_is_padded():
    times = [0.1 + 0.5 * k for k in range(11)]
    bj = assemble_beats(times, [1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], "shuffle")
    assert bj["tempo_bpm"] == 120.0 and bj["pad_beats"] == 1 and bj["pickup_beats"] == 3
    assert bj["beats"][0] == {"t": 0.1, "bar": 1, "beat": 2, "g": 1}
    assert bj["beats"][3] == {"t": 1.6, "bar": 2, "beat": 1, "g": 4}
    assert bj["bars"][0] == {"bar": 1, "t_start": -0.4, "t_end": 1.6, "beats": 3}
    assert bj["bars"][1] == {"bar": 2, "t_start": 1.6, "t_end": 3.6, "beats": 4}
    assert bj["short_bars"] == [] and bj["feel"] == "shuffle"


def test_time_beat_mapping_roundtrip_and_extrapolation():
    times = [0.1 + 0.5 * k for k in range(11)]
    bj = assemble_beats(times, [1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], "straight")
    assert float(time_to_beat(bj, 1.6)) == pytest.approx(4.0)
    assert float(beat_to_time(bj, 0.0)) == pytest.approx(-0.4)
    assert float(time_to_beat(bj, beat_to_time(bj, 7.25))) == pytest.approx(7.25)
    assert time_to_beat(bj, np.array([0.1, 0.35])).tolist() == pytest.approx([1.0, 1.5])
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest tests/test_beats.py -v`
Expected: FAIL（`ModuleNotFoundError: guitarize.analyze.beats`）

- [ ] **Step 3: 实现 `guitarize/analyze/beats.py`**

```python
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
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest tests/test_beats.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add guitarize/analyze/beats.py tests/test_beats.py
git commit -m "feat: beat cleanup, bar-line HMM and beat axis mapping"
```

---

### Task 3: 网上和弦谱解析

**Files:**
- Create: `guitarize/analyze/webchart.py`
- Test: `tests/test_webchart.py`

**Interfaces:**
- Consumes: `theory.parse_chord`（Phase 1）
- Produces: `POWER_RE`（匹配 `B5`、`C#5`）、`WebBar(chords: list[tuple[str, float]], section: str | None)`（比例和为 1，`"N"` = 无和弦 / 只有贝斯）、`normalize_label(token) -> str`、`parse_webchart(text) -> list[WebBar]`

- [ ] **Step 1: 写失败测试 `tests/test_webchart.py`**

```python
import pytest

from guitarize.analyze.webchart import normalize_label, parse_webchart

CHART = """# source: example chart (chords only)
# BPM≒120 legend line
| G5 > | C |
# Verse
| D/F# ---- Em ---- | /B ↓
| N.C. -- /A - /B - |
Am D G N.C.
| Cmaj7 Cmaj7 | xyz |
"""


def test_normalize_label():
    assert normalize_label("G5") == "G5"
    assert normalize_label("D#/G") == "D#/G"
    assert normalize_label("/B") == "N"
    assert normalize_label("N.C.") == "N"
    assert normalize_label("E↓") == "E"
    assert normalize_label("C#5>") == "C#5"
    assert normalize_label("xyz") == "N"


def test_parse_webchart_bars_weights_sections():
    bars = parse_webchart(CHART)
    labels = [[c for c, _ in b.chords] for b in bars]
    assert labels == [["G5"], ["C"], ["D/F#", "Em"], ["N"], ["N"], ["Am"], ["D"], ["G"], ["N"], ["Cmaj7"], ["N"]]
    assert [w for _, w in bars[2].chords] == pytest.approx([0.5, 0.5])
    assert bars[0].section is None and bars[2].section == "Verse" and bars[5].section == "Verse"
    assert all(abs(sum(w for _, w in b.chords) - 1) < 1e-9 for b in bars)
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest tests/test_webchart.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `guitarize/analyze/webchart.py`**

```python
"""网上和弦谱（ChordWiki 风格文本）→ 按小节的和弦列表。只处理和弦，不含歌词。

格式：`|` 分小节；没有 `|` 的行每个和弦记一小节；`#` 开头为段落注释（含 BPM 的图例行、source 行忽略）；
`-` 为八分时值标记、`=` 为十六分，用来分配小节内比例；`>`、`↓`、`＞`、`〇` 为演奏记号；
`/X`（只有贝斯）与 `N.C.` 记为 `N`；无法解析的记号记为 `N`。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .. import theory

POWER_RE = re.compile(r"^([A-G][#b]?)5$")
_MARK_RE = re.compile(r"^[-=>＞〇↓]+$")
_STRIP = "↓＞>〇"


@dataclass
class WebBar:
    chords: list[tuple[str, float]]  # (标签, 占小节比例)
    section: str | None = None


def normalize_label(token: str) -> str:
    t = token.strip().strip(_STRIP).strip()
    if not t or t in ("N.C.", "NC", "N"):
        return "N"
    if t.startswith("/"):
        return "N"
    if POWER_RE.match(t):
        return t
    try:
        return t if theory.parse_chord(t) is not None else "N"
    except ValueError:
        return "N"


def _cell_chords(cell: str) -> list[tuple[str, float]]:
    items: list[list] = []
    for tok in cell.split():
        if _MARK_RE.match(tok):
            if items:
                items[-1][1] += tok.count("-") + 0.5 * tok.count("=")
            continue
        label = normalize_label(tok)
        if items and items[-1][0] == label:
            items[-1][1] += 1.0
        else:
            items.append([label, 1.0])
    total = sum(w for _, w in items)
    return [(label, w / total) for label, w in items] if total else []


def parse_webchart(text: str) -> list[WebBar]:
    bars: list[WebBar] = []
    section: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            body = line[1:].strip()
            if body and "BPM" not in body and not body.lower().startswith("source"):
                section = body
            continue
        cells = line.strip("|").split("|") if "|" in line else line.split()
        for cell in cells:
            chords = _cell_chords(cell)
            if chords:
                bars.append(WebBar(chords, section))
    return bars
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest tests/test_webchart.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add guitarize/analyze/webchart.py tests/test_webchart.py
git commit -m "feat: parse ChordWiki-style chord charts into bars"
```

---

### Task 4: 参考和弦（模板 HMM + 网上谱对齐 + 投票）

**Files:**
- Create: `guitarize/analyze/chords.py`
- Test: `tests/test_chords.py`

**Interfaces:**
- Consumes: `theory.TEMPLATES / PITCH_CLASSES / pc / pc_name / chord_tones / majmin / transpose_label / parse_chord`；`webchart.POWER_RE / WebBar`
- Produces:
  - `LABELS`（24 个，与 `theory.MAJMIN_LABELS` 同序：`C..B` 再 `Cm..Bm`）、`MIN_ALIGN_SIM = 0.5`、`CONFIDENT_POSTERIOR = 0.7`
  - `template_track(cb, stay=0.9, temperature=8.0) -> (labels: list[str], posterior: np.ndarray)`
  - `chord_tones_any(label) -> set[int]`（支持强力和弦）、`web_bar_vector(WebBar) -> (12,)`
  - `align_webchart(audio_bars (N,12), web, jump=1.0, any_jump=2.0) -> {"index": [...], "transpose": int, "sim": [...]}`
  - `web_label_at(bar, frac, transpose) -> str`、`same_chord(a, b) -> bool`、`vote_beat(audio, posterior, web) -> (label, confidence, votes)`
  - `reference_chords(cb, beats_json, web=None) -> dict`（chords.json：`segments[{bar, beat, dur_beats, label, confidence, votes}]`，可选 `web_alignment[{bar, web_index, sim}]`、`web_transpose`）

- [ ] **Step 1: 写失败测试 `tests/test_chords.py`**

```python
import numpy as np

from guitarize import theory
from guitarize.analyze.chords import (LABELS, align_webchart, reference_chords, same_chord,
                                      template_track, vote_beat)
from guitarize.analyze.webchart import WebBar


def vec(label):
    v = np.zeros(12)
    v[list(theory.chord_tones(label))] = 1
    return v / np.linalg.norm(v)


def beats_cb(labels, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    cb = np.stack([vec(l) + noise * rng.random(12) for l in labels], axis=1)
    return cb / np.linalg.norm(cb, axis=0, keepdims=True)


def test_labels_order_matches_theory():
    assert [l.replace("m", "") + (":min" if l.endswith("m") else ":maj") for l in LABELS] == theory.MAJMIN_LABELS


def test_template_track_recovers_progression():
    truth = ["C"] * 4 + ["G"] * 4 + ["Am"] * 4 + ["F"] * 4
    labels, post = template_track(beats_cb(truth))
    assert labels == truth
    assert (post > 0.7).mean() > 0.8


def test_align_webchart_with_jump_and_transpose():
    web = [WebBar([(c, 1.0)]) for c in ["C", "G", "Am", "F", "C", "G", "F", "C"]]
    audio = np.stack([vec(c) for c in ["C", "G", "Am", "F", "F", "C"]])
    out = align_webchart(audio, web)
    assert out["index"] == [0, 1, 2, 3, 6, 7] and out["transpose"] == 0
    up2 = np.stack([vec(c) for c in ["D", "A", "Bm", "G", "G", "D"]])
    out2 = align_webchart(up2, web)
    assert out2["transpose"] == 2 and out2["index"] == [0, 1, 2, 3, 6, 7]


def test_same_chord_and_vote():
    assert same_chord("G#m", "G#5") and same_chord("B", "B/D#") and not same_chord("Em", "D#/G")
    assert vote_beat("G#m", 0.9, "G#5")[:2] == ("G#m", "high")
    assert vote_beat("B", 0.9, "B/D#")[:2] == ("B/D#", "high")
    assert vote_beat("Em", 0.9, "D#/G")[:2] == ("D#/G", "low")
    assert vote_beat("C", 0.8, None)[:2] == ("C", "medium")
    assert vote_beat("C", 0.5, None)[:2] == ("C", "low")
    assert vote_beat("C", 0.9, "N")[:2] == ("N", "medium")
    assert vote_beat("Em", 0.9, "D#/G")[2] == {"template": "Em", "web": "D#/G"}


def test_reference_chords_end_to_end():
    truth = ["C"] * 4 + ["G"] * 4 + ["Am"] * 4 + ["F"] * 4
    cb = beats_cb(truth)
    beats_json = {"time_signature": [4, 4],
                  "beats": [{"t": 0.5 * i, "bar": i // 4 + 1, "beat": i % 4 + 1, "g": i} for i in range(16)]}
    web = [WebBar([(c, 1.0)]) for c in ["C", "G", "Am", "F"]]
    out = reference_chords(cb, beats_json, web)
    segs = [(s["bar"], s["beat"], s["dur_beats"], s["label"], s["confidence"]) for s in out["segments"]]
    assert segs == [(1, 1, 4, "C", "high"), (2, 1, 4, "G", "high"), (3, 1, 4, "Am", "high"), (4, 1, 4, "F", "high")]
    assert [a["web_index"] for a in out["web_alignment"]] == [0, 1, 2, 3] and out["web_transpose"] == 0
    no_web = reference_chords(cb, beats_json)
    assert "web_alignment" not in no_web
    assert {s["confidence"] for s in no_web["segments"]} <= {"medium", "low"}
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest tests/test_chords.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `guitarize/analyze/chords.py`**

```python
"""参考和弦：音频模板匹配 + HMM、网上和弦谱对齐，逐拍投票得出标签与置信度（chords.json）。"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .. import theory
from .webchart import POWER_RE, WebBar

LABELS = list(theory.PITCH_CLASSES) + [n + "m" for n in theory.PITCH_CLASSES]  # 与 theory.MAJMIN_LABELS 同序
_TEMPLATES = theory.TEMPLATES / np.linalg.norm(theory.TEMPLATES, axis=1, keepdims=True)
MIN_ALIGN_SIM = 0.5
CONFIDENT_POSTERIOR = 0.7


def template_track(cb: np.ndarray, stay: float = 0.9, temperature: float = 8.0) -> tuple[list[str], np.ndarray]:
    """cb: (12, n) 单位化逐拍色度 → 逐拍标签（Viterbi 路径）及该标签的边缘后验概率（前向-后向）。"""
    log_e = temperature * (_TEMPLATES @ cb)
    log_e = log_e - logsumexp(log_e, axis=0, keepdims=True)
    k, n = log_e.shape
    log_a = np.full((k, k), np.log((1 - stay) / (k - 1)))
    np.fill_diagonal(log_a, np.log(stay))

    delta = log_e[:, 0] - np.log(k)
    back = np.zeros((k, n), dtype=int)
    for i in range(1, n):
        m = delta[:, None] + log_a
        back[:, i] = m.argmax(axis=0)
        delta = m.max(axis=0) + log_e[:, i]
    path = [int(delta.argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[path[-1], i]))
    path = path[::-1]

    fwd = np.zeros((k, n))
    bwd = np.zeros((k, n))
    fwd[:, 0] = log_e[:, 0] - np.log(k)
    for i in range(1, n):
        fwd[:, i] = logsumexp(fwd[:, i - 1][:, None] + log_a, axis=0) + log_e[:, i]
    for i in range(n - 2, -1, -1):
        bwd[:, i] = logsumexp(log_a + (log_e[:, i + 1] + bwd[:, i + 1])[None, :], axis=1)
    post = fwd + bwd
    post = np.exp(post - logsumexp(post, axis=0, keepdims=True))
    return [LABELS[s] for s in path], np.array([post[s, i] for i, s in enumerate(path)])


def chord_tones_any(label: str) -> set[int]:
    m = POWER_RE.match(label)
    if m:
        r = theory.pc(m.group(1))
        return {r, (r + 7) % 12}
    return theory.chord_tones(label)


def web_bar_vector(bar: WebBar) -> np.ndarray:
    v = np.zeros(12)
    for label, w in bar.chords:
        if label != "N":
            for p in chord_tones_any(label):
                v[p] += w
    n = np.linalg.norm(v)
    return v / n if n else v


def _dp_align(sim: np.ndarray, jump: float, any_jump: float) -> tuple[float, list[int]]:
    """每个音频小节 i 对应一个网上小节 j。转移：j−1 → j 不罚；向前跳过若干小节罚 jump；任意跳（含回跳）罚 any_jump。"""
    n, m = sim.shape
    neg = -1e18
    dp = np.full((n, m), neg)
    back = np.zeros((n, m), dtype=int)
    dp[0] = sim[0]
    for i in range(1, n):
        prev = dp[i - 1]
        run_arg = np.zeros(m, dtype=int)
        best = 0
        for j in range(m):
            if prev[j] > prev[best]:
                best = j
            run_arg[j] = best
        glob = int(prev.argmax())
        for j in range(m):
            cands = [(prev[glob] - any_jump, glob)]
            if j >= 1:
                cands.append((prev[j - 1], j - 1))
            if j >= 2:
                cands.append((prev[run_arg[j - 2]] - jump, int(run_arg[j - 2])))
            score, arg = max(cands)
            dp[i, j] = score + sim[i, j]
            back[i, j] = arg
    path = [int(dp[-1].argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    return float(dp[-1].max()), path[::-1]


def align_webchart(audio_bars: np.ndarray, web: list[WebBar], jump: float = 1.0, any_jump: float = 2.0) -> dict:
    """audio_bars: (N, 12) 单位化小节色度。网上谱整体移调 transpose 个半音后与音频最匹配。"""
    w = np.stack([web_bar_vector(b) for b in web])
    best = None
    for t in range(12):
        sim = audio_bars @ np.roll(w, t, axis=1).T
        score, path = _dp_align(sim, jump, any_jump)
        if best is None or score > best[0] + 1e-9:
            best = (score, t, path, sim)
    _, t, path, sim = best
    return {"index": path, "transpose": t, "sim": [float(sim[i, j]) for i, j in enumerate(path)]}


def web_label_at(bar: WebBar, frac: float, transpose: int) -> str:
    """小节内位置 frac（0..1）处的网上和弦，按 transpose 移调。"""
    acc = 0.0
    label = bar.chords[-1][0]
    for lab, w in bar.chords:
        acc += w
        if frac < acc - 1e-9:
            label = lab
            break
    return label if label == "N" or transpose == 0 else theory.transpose_label(label, transpose)


def _root(label: str) -> int | None:
    if label == "N":
        return None
    m = POWER_RE.match(label)
    if m:
        return theory.pc(m.group(1))
    parsed = theory.parse_chord(label)
    return None if parsed is None else parsed[0]


def same_chord(a: str, b: str) -> bool:
    if a == "N" or b == "N":
        return False
    if POWER_RE.match(a) or POWER_RE.match(b):
        return _root(a) == _root(b)
    return theory.majmin(a) == theory.majmin(b)


def vote_beat(audio: str, posterior: float, web: str | None) -> tuple[str, str, dict]:
    """返回 (标签, 置信度, 各方投票)。web=None 表示该拍没有可用的网上和弦（N=1）。"""
    votes = {"template": audio}
    if web is None:
        return audio, ("medium" if posterior >= CONFIDENT_POSTERIOR else "low"), votes
    votes["web"] = web
    if web == "N":
        return "N", "medium", votes
    if same_chord(audio, web):
        if POWER_RE.match(web):
            return theory.pc_name(_root(web)) + ("m" if audio.endswith("m") else ""), "high", votes
        return web, "high", votes
    return web, "low", votes


def reference_chords(cb: np.ndarray, beats_json: dict, web: list[WebBar] | None = None) -> dict:
    """cb: 与 beats_json["beats"] 一一对应的逐拍色度 (12, n)。"""
    labels, post = template_track(cb)
    beats = beats_json["beats"]
    bpb = beats_json["time_signature"][0]
    numbers = sorted({b["bar"] for b in beats})
    web_at: list[str | None] = [None] * len(beats)
    alignment = None
    if web:
        vecs = []
        for n in numbers:
            idx = [i for i, b in enumerate(beats) if b["bar"] == n]
            v = cb[:, idx].mean(axis=1)
            vecs.append(v / (np.linalg.norm(v) + 1e-9))
        alignment = align_webchart(np.stack(vecs), web)
        for k, n in enumerate(numbers):
            if alignment["sim"][k] < MIN_ALIGN_SIM:
                continue
            wb = web[alignment["index"][k]]
            for i, b in enumerate(beats):
                if b["bar"] == n:
                    web_at[i] = web_label_at(wb, (b["beat"] - 1) / bpb, alignment["transpose"])
    segments: list[dict] = []
    for i, b in enumerate(beats):
        label, conf, votes = vote_beat(labels[i], float(post[i]), web_at[i])
        last = segments[-1] if segments else None
        if (last and last["bar"] == b["bar"] and last["label"] == label and last["confidence"] == conf
                and last["votes"] == votes and last["beat"] + last["dur_beats"] == b["beat"]):
            last["dur_beats"] += 1
        else:
            segments.append({"bar": b["bar"], "beat": b["beat"], "dur_beats": 1, "label": label,
                             "confidence": conf, "votes": votes})
    out: dict = {"segments": segments}
    if alignment is not None:
        out["web_alignment"] = [{"bar": n, "web_index": alignment["index"][k], "sim": round(alignment["sim"][k], 3)}
                                for k, n in enumerate(numbers)]
        out["web_transpose"] = alignment["transpose"]
    return out
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest tests/test_chords.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add guitarize/analyze/chords.py tests/test_chords.py
git commit -m "feat: reference chords by template HMM, web chart alignment and voting"
```

---

### Task 5: 人声旋律

**Files:**
- Create: `guitarize/analyze/melody.py`
- Test: `tests/test_melody.py`

**Interfaces:**
- Consumes: `beats.time_to_beat`（Task 2）
- Produces: `vocal_pitch(wav) -> (times, midi_float_or_nan, periodicity)`、`snap_to_key(m, key_pcs) -> (midi, out_of_key)`、`quantize(beat_pos, midi, periodicity, onset_beats, key_pcs, slot) -> list[dict]`、`fix_octaves(notes, window=3) -> list[dict]`、`extract_melody(vocals_wav, beats_json, key_pcs) -> {"grid": "triplet8"|"sixteenth", "notes": [{start_beats, dur_beats, midi, out_of_key, conf}]}`（只保留 start_beats ≥ 0 的音）

- [ ] **Step 1: 写失败测试 `tests/test_melody.py`**

```python
import numpy as np
import pytest

from guitarize.analyze.melody import fix_octaves, quantize, snap_to_key

C_MAJOR = {0, 2, 4, 5, 7, 9, 11}


def test_snap_to_key():
    assert snap_to_key(61.3, C_MAJOR) == (62, False)
    assert snap_to_key(64.2, C_MAJOR) == (64, False)
    assert snap_to_key(61.0, C_MAJOR) == (61, True)


def frames(segments, per_beat=60):
    """segments: [(start_beat, end_beat, midi or None)] → 逐帧拍位置与音高。"""
    pos, midi = [], []
    for a, b, m in segments:
        n = int((b - a) * per_beat)
        pos += list(a + np.arange(n) / per_beat)
        midi += [np.nan if m is None else m] * n
    return np.array(pos), np.array(midi, dtype=float)


def test_quantize_merges_slots_and_splits_on_onsets():
    pos, midi = frames([(0, 1, 60.1), (1, 2, 62.0), (2, 3, None), (3, 4, 64.0), (4, 6, 60.0)])
    per = np.ones_like(midi)
    notes = quantize(pos, midi, per, onset_beats=np.array([0.0, 1.0, 3.0, 4.0, 5.0]), key_pcs=C_MAJOR, slot=1 / 3)
    got = [(n["start_beats"], n["dur_beats"], n["midi"]) for n in notes]
    assert got == pytest.approx([(0, 1, 60), (1, 1, 62), (3, 1, 64), (4, 1, 60), (5, 1, 60)], abs=1e-3)
    assert all(n["out_of_key"] is False and n["conf"] == 1.0 for n in notes)


def test_fix_octaves():
    notes = [{"midi": m} for m in [60, 62, 76, 64, 65]]
    assert [n["midi"] for n in fix_octaves(notes)] == [60, 62, 64, 64, 65]
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest tests/test_melody.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `guitarize/analyze/melody.py`**

```python
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
    k0 = int(np.floor(beat_pos.min() / slot))
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
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest tests/test_melody.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add guitarize/analyze/melody.py tests/test_melody.py
git commit -m "feat: vocal melody extraction quantized to the beat grid"
```

---

### Task 6: 分轨、leadsheet、分析编排与命令行

**Files:**
- Create: `guitarize/analyze/stems.py`, `guitarize/analyze/leadsheet.py`, `guitarize/analyze/pipeline.py`
- Modify: `guitarize/cli.py`（新增 `analyze`、`summary` 子命令）
- Test: `tests/test_leadsheet.py`

**Interfaces:**
- Consumes: Task 1–5 全部；`SongPaths`、`SongConfig`（Phase 1）
- Produces:
  - `stems.STEMS = ("vocals","drums","bass","guitar","piano","other")`、`stems.mix_harmonic(stems_dir) -> Path`（bass+guitar+piano+other → harmonic.wav）、`stems.separate(mix, stems_dir, model="htdemucs_6s") -> dict[str, Path]`
  - `leadsheet.build_leadsheet(beats, chords, melody, key, sections) -> dict`、`leadsheet.summary_table(leadsheet, capo=0) -> str`
  - `pipeline.STEPS`、`pipeline.run_analysis(paths, cfg, steps=STEPS) -> None`；`song.yaml` 的 `overrides` 支持 `key`（如 `"B"`、`"G#m"`）、`downbeat_shift`、`feel`
  - CLI：`guitarize analyze <slug> [--steps stems,beats,chords,melody,leadsheet]`、`guitarize summary <slug> [--capo N]`

- [ ] **Step 1: 写失败测试 `tests/test_leadsheet.py`**

```python
import numpy as np
import soundfile as sf

from guitarize.analyze.leadsheet import build_leadsheet, summary_table
from guitarize.analyze.stems import mix_harmonic

BEATS = {"tempo_bpm": 120.0, "feel": "straight", "time_signature": [4, 4], "pickup_beats": 0, "pad_beats": 0,
         "short_bars": [], "beats": [], "bars": [{"bar": 1, "t_start": 0.0, "t_end": 2.0, "beats": 4},
                                                  {"bar": 2, "t_start": 2.0, "t_end": 4.0, "beats": 4}]}
CHORDS = {"segments": [
    {"bar": 1, "beat": 1, "dur_beats": 4, "label": "C", "confidence": "high", "votes": {"template": "C", "web": "C"}},
    {"bar": 2, "beat": 1, "dur_beats": 2, "label": "G", "confidence": "low", "votes": {"template": "Em", "web": "G"}},
    {"bar": 2, "beat": 3, "dur_beats": 2, "label": "N", "confidence": "medium", "votes": {"template": "C", "web": "N"}},
]}
MELODY = {"grid": "sixteenth", "notes": [
    {"start_beats": 0.0, "dur_beats": 1.0, "midi": 60, "out_of_key": False, "conf": 0.9},
    {"start_beats": 4.5, "dur_beats": 0.5, "midi": 61, "out_of_key": True, "conf": 0.8},
]}


def test_build_leadsheet_matches_phase1_contract():
    ls = build_leadsheet(BEATS, CHORDS, MELODY, ("C", "major"), {2: "サビ"})
    assert ls["key"] == "C" and ls["mode"] == "major" and ls["beats_per_bar"] == 4
    b1, b2 = ls["bars"]
    assert (b1["bar"], b1["section"], b1["t_start"], b1["t_end"]) == (1, None, 0.0, 2.0)
    assert b2["section"] == "サビ"
    assert [c["label"] for c in b2["chords"]] == ["G", "N"]
    assert b1["melody"] == [{"start_beats": 0.0, "dur_beats": 1.0, "midi": 60, "conf": 0.9, "out_of_key": False}]
    assert b2["melody"][0]["midi"] == 61


def test_summary_table_shows_confidence_and_shapes():
    ls = build_leadsheet(BEATS, CHORDS, MELODY, ("C", "major"), {2: "サビ"})
    text = summary_table(ls, capo=2)
    lines = text.splitlines()
    assert "capo 2" in lines[0]
    assert "1:C(A#)" in lines[1]
    assert "G(F)??" in lines[2] and "template=Em" in lines[2] and "[サビ]" in lines[2]
    assert "C#4*" in lines[2]


def test_mix_harmonic(tmp_path):
    for name, v in {"bass": 0.1, "guitar": 0.2, "piano": 0.0, "other": 0.05}.items():
        sf.write(tmp_path / f"{name}.wav", np.full((100, 2), v), 22050)
    out = mix_harmonic(tmp_path)
    y, sr = sf.read(out)
    assert sr == 22050 and y.shape == (100, 2) and np.allclose(y, 0.35, atol=1e-3)
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest tests/test_leadsheet.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `guitarize/analyze/stems.py`**

```python
"""音源分离：Demucs htdemucs_6s → analysis/stems/*.wav，并合成和声分轨 harmonic.wav（去人声、去鼓）。"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")
HARMONIC_PARTS = ("bass", "guitar", "piano", "other")


def _device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def mix_harmonic(stems_dir: Path) -> Path:
    total, sr = None, None
    for name in HARMONIC_PARTS:
        y, sr = sf.read(str(Path(stems_dir) / f"{name}.wav"), always_2d=True)
        if total is None:
            total = y
        else:
            n = max(len(total), len(y))
            total = np.pad(total, ((0, n - len(total)), (0, 0))) + np.pad(y, ((0, n - len(y)), (0, 0)))
    out = Path(stems_dir) / "harmonic.wav"
    sf.write(str(out), total, sr)
    return out


def separate(mix: Path, stems_dir: Path, model: str = "htdemucs_6s") -> dict[str, Path]:
    stems_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="guitarize-demucs-") as tmp:
        subprocess.run([sys.executable, "-m", "demucs", "-n", model, "-d", _device(), "-o", tmp, str(mix)],
                       check=True, capture_output=True)
        produced = Path(tmp) / model / Path(mix).stem
        for name in STEMS:
            shutil.move(str(produced / f"{name}.wav"), str(stems_dir / f"{name}.wav"))
    mix_harmonic(stems_dir)
    return {name: stems_dir / f"{name}.wav" for name in (*STEMS, "harmonic")}
```

- [ ] **Step 4: 实现 `guitarize/analyze/leadsheet.py`**

```python
"""合并 beats / chords / melody → leadsheet.json（Phase 1 校验器的参考），并输出便于阅读的小节摘要。"""
from __future__ import annotations

from .. import theory

_MARK = {"high": "", "medium": "?", "low": "??"}


def build_leadsheet(beats: dict, chords: dict, melody: dict, key: tuple[str, str], sections: dict[int, str]) -> dict:
    bpb = beats["time_signature"][0]
    by_bar: dict[int, list[dict]] = {}
    for s in chords["segments"]:
        by_bar.setdefault(s["bar"], []).append(s)
    bars = []
    for b in beats["bars"]:
        n = b["bar"]
        g0 = (n - 1) * bpb
        bars.append({
            "bar": n,
            "section": sections.get(n),
            "t_start": b["t_start"],
            "t_end": b["t_end"],
            "chords": [{k: s[k] for k in ("beat", "dur_beats", "label", "confidence", "votes")} for s in by_bar.get(n, [])],
            "melody": [{k: m[k] for k in ("start_beats", "dur_beats", "midi", "conf", "out_of_key")}
                       for m in melody["notes"] if g0 <= m["start_beats"] < g0 + bpb],
        })
    return {"key": key[0], "mode": key[1], "tempo_bpm": beats["tempo_bpm"], "feel": beats["feel"],
            "beats_per_bar": bpb, "pickup_beats": beats["pickup_beats"], "bars": bars}


def summary_table(leadsheet: dict, capo: int = 0) -> str:
    """一小节一行：小节号 · 起始秒 · 段落 | 拍:和弦(指法调)置信标记[投票] | 旋律音（* = 调外）。"""
    head = f"key {leadsheet['key']} {leadsheet['mode']} · ♩≈{leadsheet['tempo_bpm']} · {leadsheet['feel']}"
    lines = [head + (f" · capo {capo}" if capo else "")]
    for b in leadsheet["bars"]:
        chords = []
        for c in b["chords"]:
            label = c["label"]
            if capo and label != "N":
                label = f"{label}({theory.transpose_label(label, -capo)})"
            extra = "" if c["confidence"] == "high" else "[" + " ".join(f"{k}={v}" for k, v in c["votes"].items()) + "]"
            chords.append(f"{c['beat']}:{label}{_MARK[c['confidence']]}{extra}")
        melody = " ".join(theory.midi_name(m["midi"]) + ("*" if m.get("out_of_key") else "") for m in b["melody"])
        section = f"[{b['section']}] " if b["section"] else ""
        lines.append(f"{b['bar']:3d} {b['t_start']:6.1f}s {section}| {'  '.join(chords)} | {melody}")
    return "\n".join(lines)
```

- [ ] **Step 5: 实现 `guitarize/analyze/pipeline.py`**

```python
"""分析编排：stems → beats → chords → melody → leadsheet，每步结果写入 analysis/。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from ..config import SongConfig
from ..paths import SongPaths
from . import features
from .beats import track_beats
from .chords import reference_chords
from .key import estimate_key, parse_key, scale_pcs
from .leadsheet import build_leadsheet
from .melody import extract_melody
from .stems import separate
from .webchart import parse_webchart

STEPS = ("stems", "beats", "chords", "melody", "leadsheet")


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_analysis(paths: SongPaths, cfg: SongConfig, steps: Sequence[str] = STEPS) -> None:
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown analysis steps {sorted(unknown)}; choose from {STEPS}")
    overrides = cfg.overrides
    if "stems" in steps:
        separate(paths.original_wav, paths.stems_dir)
    if "beats" in steps:
        _write(paths.beats_json, track_beats(paths.original_wav, paths.stems_dir, overrides))
    beats = _read(paths.beats_json)
    if "chords" in steps:
        times = [b["t"] for b in beats["beats"]]
        cb = features.beat_sync_chroma(features.load_mix([paths.stem("harmonic")]), times)
        if overrides.get("key"):
            tonic, mode = parse_key(str(overrides["key"]))
            r = None
        else:
            tonic, mode, r = estimate_key(cb.mean(axis=1))
        web = parse_webchart(paths.webchart_txt.read_text(encoding="utf-8")) if paths.webchart_txt.exists() else None
        chords = reference_chords(cb, beats, web)
        chords["key"] = {"tonic": tonic, "mode": mode, "r": r}
        _write(paths.chords_json, chords)
    chords = _read(paths.chords_json)
    key = (chords["key"]["tonic"], chords["key"]["mode"])
    if "melody" in steps:
        _write(paths.melody_json, extract_melody(paths.stem("vocals"), beats, scale_pcs(*key)))
    if "leadsheet" in steps:
        _write(paths.leadsheet_json, build_leadsheet(beats, chords, _read(paths.melody_json), key, cfg.sections))
```

- [ ] **Step 6: 修改 `guitarize/cli.py`：新增 `analyze` 与 `summary`**

在 `main()` 中 `p = sub.add_parser("serve", ...)` 之前加入：
```python
    p = sub.add_parser("analyze", help="分析原曲：分轨 / 节拍 / 和弦 / 旋律 / leadsheet")
    p.add_argument("slug")
    p.add_argument("--steps", default="stems,beats,chords,melody,leadsheet",
                   help="逗号分隔，可选 stems,beats,chords,melody,leadsheet")
    p = sub.add_parser("summary", help="打印 leadsheet 的逐小节摘要")
    p.add_argument("slug")
    p.add_argument("--capo", type=int, default=0)
```
在 `elif args.cmd == "verify":` 分支之后加入：
```python
    elif args.cmd == "analyze":
        from .analyze.leadsheet import summary_table
        from .analyze.pipeline import run_analysis

        run_analysis(paths, cfg, [s.strip() for s in args.steps.split(",") if s.strip()])
        if paths.leadsheet_json.exists():
            import json

            print(summary_table(json.loads(paths.leadsheet_json.read_text(encoding="utf-8"))))
    elif args.cmd == "summary":
        import json

        from .analyze.leadsheet import summary_table

        print(summary_table(json.loads(paths.leadsheet_json.read_text(encoding="utf-8")), capo=args.capo))
```

- [ ] **Step 7: 运行，确认通过**

Run: `$PY -m pytest tests/test_leadsheet.py -v && $PY -m guitarize --help`
Expected: 3 passed；帮助文本列出 `analyze` 与 `summary`

- [ ] **Step 8: Commit**

```bash
git add guitarize/analyze guitarize/cli.py tests/test_leadsheet.py
git commit -m "feat: stems, leadsheet assembly, analysis pipeline and CLI"
```

---

### Task 7: 闭环测试（已知答案的合成歌曲）

**Files:**
- Create: `tests/synth_song.py`, `tests/test_closed_loop.py`

**Interfaces:**
- Consumes: Phase 1 `render.synth.midi_to_wav`；Task 1–6 的分析函数
- Produces: `synth_song.render_song(out_dir) -> {"mix": Path, "stems": Path}`、`PROGRESSION`、`OFFSET`、`SPB`、`bar_start(bar)`、`melody_truth()`、`truth_webchart()`

- [ ] **Step 1: 写合成歌曲 `tests/synth_song.py`**

```python
"""闭环测试用的合成歌曲：已知节拍 / 和弦 / 旋律的 MIDI → 各声部 wav（FluidSynth + sonivox）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

from guitarize import theory
from guitarize.render.synth import midi_to_wav

TEMPO = 120.0
SPB = 60.0 / TEMPO
OFFSET = 0.25  # 第 1 小节起点（秒）
PROGRESSION = ["G", "D/F#", "Em", "C", "G", "D", "C", "D", "G", "D/F#", "Em", "C", "Am", "D", "G", "G"]


def bar_start(bar: int) -> float:
    return OFFSET + (bar - 1) * 4 * SPB


def _triad(label: str, low: int) -> list[int]:
    root, quality, _ = theory.parse_chord(label)
    r = low + ((root - low) % 12)
    return [r, r + (3 if quality == "min" else 4), r + 7]


def _bass(label: str) -> int:
    root, _, bass = theory.parse_chord(label)
    pc = bass if bass is not None else root
    return 40 + ((pc - 40) % 12)


def melody_truth() -> list[tuple[float, float, int]]:
    """(start_beats, dur_beats, midi)：每小节根音、三音、五音、三音四个四分音符（根音在 C4–B4），末小节全音符根音。"""
    out = []
    for i, label in enumerate(PROGRESSION):
        r, third, fifth = _triad(label, 60)
        if i == len(PROGRESSION) - 1:
            out.append((i * 4.0, 4.0, r))
        else:
            out += [(i * 4.0 + k, 1.0, p) for k, p in enumerate([r, third, fifth, third])]
    return out


def truth_webchart() -> str:
    rows = [PROGRESSION[i:i + 4] for i in range(0, len(PROGRESSION), 4)]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n"


def _part(program: int, is_drum: bool = False) -> tuple[pretty_midi.PrettyMIDI, pretty_midi.Instrument]:
    pm = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
    inst = pretty_midi.Instrument(program=program, is_drum=is_drum)
    pm.instruments.append(inst)
    return pm, inst


def build_parts() -> dict[str, pretty_midi.PrettyMIDI]:
    parts = {}
    pm, drums = _part(0, is_drum=True)
    for bar in range(1, len(PROGRESSION) + 1):
        for beat in range(4):
            t = bar_start(bar) + beat * SPB
            drums.notes.append(pretty_midi.Note(110 if beat == 0 else 95, 36 if beat % 2 == 0 else 38, t, t + 0.1))
            for half in (0.0, 0.5):
                th = t + half * SPB
                drums.notes.append(pretty_midi.Note(70, 42, th, th + 0.05))
    parts["drums"] = pm
    pm, bass = _part(33)
    for i, label in enumerate(PROGRESSION):
        for beat in range(4):
            t = bar_start(i + 1) + beat * SPB
            bass.notes.append(pretty_midi.Note(90, _bass(label), t, t + SPB * 0.9))
    parts["bass"] = pm
    pm, guitar = _part(25)
    for i, label in enumerate(PROGRESSION):
        for beat in (0, 2):
            t = bar_start(i + 1) + beat * SPB
            for p in _triad(label, 55):
                guitar.notes.append(pretty_midi.Note(80, p, t, t + 2 * SPB * 0.95))
    parts["guitar"] = pm
    pm, lead = _part(73)
    for s, d, p in melody_truth():
        t = OFFSET + s * SPB
        lead.notes.append(pretty_midi.Note(100, p, t, t + d * SPB * 0.9))
    parts["vocals"] = pm
    return parts


def render_song(out_dir: Path) -> dict[str, Path]:
    """渲染到 out_dir/stems/{vocals,drums,bass,guitar,piano,other}.wav 与 out_dir/mix.wav。"""
    stems = out_dir / "stems"
    stems.mkdir(parents=True, exist_ok=True)
    audio, sr = {}, 44100
    for name, pm in build_parts().items():
        mid = out_dir / f"{name}.mid"
        pm.write(str(mid))
        y, sr = sf.read(str(midi_to_wav(mid, out_dir / f"{name}.raw.wav")), always_2d=True)
        audio[name] = y
    n = max(len(y) for y in audio.values())
    padded = {k: np.pad(y, ((0, n - len(y)), (0, 0))) for k, y in audio.items()}
    padded["piano"] = np.zeros((n, 2))
    padded["other"] = np.zeros((n, 2))
    for name, y in padded.items():
        sf.write(str(stems / f"{name}.wav"), y, sr)
    mix = sum(padded[k] for k in ("drums", "bass", "guitar", "vocals"))
    sf.write(str(out_dir / "mix.wav"), mix / max(1.0, float(np.abs(mix).max())), sr)
    return {"mix": out_dir / "mix.wav", "stems": stems}
```

- [ ] **Step 2: 写闭环测试 `tests/test_closed_loop.py`**

```python
"""闭环：合成歌曲 → 分析 → 与真值比较。验证"耳朵"本身是否可靠。"""
import numpy as np
import pytest

import synth_song as S
from guitarize import theory
from guitarize.analyze import features
from guitarize.analyze.beats import time_to_beat, track_beats
from guitarize.analyze.chords import reference_chords, template_track
from guitarize.analyze.key import estimate_key, scale_pcs
from guitarize.analyze.melody import extract_melody
from guitarize.analyze.stems import STEMS, mix_harmonic, separate
from guitarize.analyze.webchart import parse_webchart

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def song(tmp_path_factory):
    out = S.render_song(tmp_path_factory.mktemp("synth"))
    mix_harmonic(out["stems"])
    out["beats"] = track_beats(out["mix"], out["stems"])
    times = [b["t"] for b in out["beats"]["beats"]]
    out["cb"] = features.beat_sync_chroma(features.load_mix([out["stems"] / "harmonic.wav"]), times)
    return out


def truth_chord_at(t: float) -> str | None:
    k = int((t + 0.05 - S.OFFSET) // (4 * S.SPB))
    return S.PROGRESSION[k] if 0 <= k < len(S.PROGRESSION) else None


def test_beats_tempo_and_downbeats(song):
    bj = song["beats"]
    assert abs(bj["tempo_bpm"] - 120) / 120 < 0.02
    assert bj["feel"] == "straight" and bj["short_bars"] == []
    downbeats = [b["t"] for b in bj["beats"] if b["beat"] == 1]
    hits = sum(any(abs(d - S.bar_start(k)) < 0.07 for d in downbeats) for k in range(1, 17))
    assert hits >= 15


def test_key(song):
    assert estimate_key(song["cb"].mean(axis=1))[:2] == ("G", "major")


def test_template_chords_accuracy(song):
    labels, _ = template_track(song["cb"])
    pairs = [(labels[i], truth_chord_at(b["t"])) for i, b in enumerate(song["beats"]["beats"])]
    pairs = [(a, t) for a, t in pairs if t is not None]
    acc = np.mean([theory.majmin(a) == theory.majmin(t) for a, t in pairs])
    assert acc >= 0.85, acc


def test_reference_chords_with_web_chart(song):
    out = reference_chords(song["cb"], song["beats"], parse_webchart(S.truth_webchart()))
    beats = song["beats"]["beats"]
    per_beat = {}
    for s in out["segments"]:
        for k in range(s["dur_beats"]):
            per_beat[(s["bar"], s["beat"] + k)] = s
    pairs = [(per_beat[(b["bar"], b["beat"])], truth_chord_at(b["t"])) for b in beats]
    pairs = [(s, t) for s, t in pairs if t is not None]
    assert np.mean([s["confidence"] == "high" for s, _ in pairs]) >= 0.8
    assert np.mean([theory.majmin(s["label"]) == theory.majmin(t) for s, t in pairs]) >= 0.95


def test_melody_f1(song):
    mel = extract_melody(song["stems"] / "vocals.wav", song["beats"], scale_pcs("G", "major"))
    truth = [(float(time_to_beat(song["beats"], S.OFFSET + s * S.SPB)), p) for s, _, p in S.melody_truth()]
    est = [(n["start_beats"], n["midi"]) for n in mel["notes"]]
    used, hit = set(), 0
    for ts, tp in truth:
        for j, (es, ep) in enumerate(est):
            if j not in used and ep == tp and abs(es - ts) <= 1 / 3:
                used.add(j)
                hit += 1
                break
    p, r = hit / max(1, len(est)), hit / len(truth)
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    assert f1 >= 0.8, (p, r)


def test_separate_produces_all_stems(song, tmp_path):
    out = separate(song["mix"], tmp_path / "stems")
    assert set(out) == {*STEMS, "harmonic"} and all(p.exists() for p in out.values())
```

- [ ] **Step 3: 运行闭环测试**

Run: `$PY -m pytest tests/test_closed_loop.py -v -m slow`
Expected: 6 passed（约 1–3 分钟，首次运行 Demucs 会下载权重）。
若某项阈值不达标：**不要放宽阈值**，先定位是合成素材问题（例如 sonivox 某音色发声异常，可在报告中给出色度/音高诊断）还是分析算法问题，在报告中写明证据后以 DONE_WITH_CONCERNS 返回，由控制者裁决。

- [ ] **Step 4: 全量测试（含慢测试）**

Run: `$PY -m pytest -v`
Expected: 全部通过

- [ ] **Step 5: Commit**

```bash
git add tests/synth_song.py tests/test_closed_loop.py
git commit -m "test: closed-loop analysis check on a synthesized song with known answers"
```

---

### Task 8: 首曲分析与人工核对（主会话执行，非子代理）

**Files:**
- Create: `songs/nanmonee/song.yaml`（只提交 song.yaml、analysis/webchart.txt、analysis/*.json；原曲与分轨不提交）

- [ ] **Step 1: 写 `songs/nanmonee/song.yaml`**

```yaml
slug: nanmonee
title: なんもねえ
artist: 忘れらんねえよ
version: tv
level: intermediate
arrangements: [strum, finger]
outputs: [html, pdf, txt]
source:
  url: https://www.bilibili.com/video/BV1yfTH6oEiT
  kind: bilibili
  trim: null
reference_charts:
  - https://ja.chordwiki.org/wiki/なんもねえ
overrides:
  key: B
  feel: shuffle
```

- [ ] **Step 2: 跑分析并阅读摘要**

Run: `$PY -m guitarize analyze nanmonee` 然后 `$PY -m guitarize summary nanmonee --capo 4`
检查：小节数 ≈ 61 且无 short_bars；tempo ≈ 167；和弦大段与 ChordWiki 进行（B E D#/G G#m E B/D# C# D# …）一致；低置信小节集中在已知难点（之前原型中第 18–20、52、54 小节）。

- [ ] **Step 3: 标注段落**

根据摘要与对齐结果，把段落写入 `song.yaml` 的 `sections`（例：`{1: Intro-サビ, 21: Aメロ, 29: Bメロ, 37: サビ, 56: Outro}`，以实际摘要为准），再运行 `$PY -m guitarize analyze nanmonee --steps leadsheet`。

- [ ] **Step 4: Commit**

```bash
git add songs/nanmonee/song.yaml songs/nanmonee/analysis/webchart.txt songs/nanmonee/analysis/*.json
git commit -m "data: なんもねえ TV size analysis (beats, chords, melody, leadsheet)"
```
