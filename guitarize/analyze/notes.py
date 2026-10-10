"""旋律 v7：GAME（唱歌 → 音符的模型）在模型主唱 lead_raw.wav 上的分音做骨架，再用主唱的逐帧信号检查：
- 伴唱漏音（主唱轨比伴唱轨弱 3 dB 以上，或左右不相关 = 摆在两边的伴唱）/ 没声音占一半以上的音丢掉；
- 音高：GAME 与 crepe 一致时取平均，再吸附到调内音或当时和弦里的音（D#/G 上的 G 不算调外）；
- 换字（音量下陷 / 辅音的无声）处切开、重新拨弦；长音里音高真的变了（≥ 1 拍）也切开，后半段不重新拨（legato）；
- 同音、中间没有换字 → 一个长音，只拨一下等它延音；换音没有换字时，只有快的装饰音（击 / 勾）和 ≥ 3 半音的滑音不重新拨，
  其余照常拨弦；字头往上滑的一格短音并进后面的音；GAME 漏掉的短音（一两个半音的一步）补上；
- 量化到格（shuffle 三连音，中间那个位置 +1/3 要多一点证据才用）。
在 なんもねえ 上和干净主唱逐帧对过（melody_audit）：伴唱音、切碎的长音、滑音切碎都比逐格量化少。"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from .. import theory
from .beats import beat_to_time, time_to_beat
from .leadsheet import _bar_beats, manual_segments, parse_chord_overrides
from .melody import HOP, SNAP_SEMITONES, SR, VOICED_PERIODICITY, _span, cached_vocal_pitch, snap_to_key, vocal_pitch

V7_METHOD = "v7:game+signal"
FPS = SR / HOP              # 逐帧特征的帧率（10 ms 一帧，与 crepe 相同）
FRAME_LENGTH = 1024         # RMS / 左右相关的窗长（采样）

LEAK_DB = 3.0               # 主唱轨比伴唱轨还弱 3 dB 以上 → 伴唱漏音
LR_CORR_MIN = 0.6           # 主唱左右相关（5 帧平滑）低于它 → 摆在两边的伴唱（真主唱 0.8–0.95）
VOICED_DB = -32.0           # 比最响处低 32 dB 以上不算在唱
ONSET_GAP = 4               # 停了 ≥ 4 帧再出声 → 换字
ONSET_DIP = 5.0             # 音量下陷的突出度 ≥ 5 dB → 换字
SPLIT_DIP = 6.0             # ≥ 6 dB → 强换字点：GAME 并在一起的两个字在这里切开
GAP_DIP = 2.5               # 无声（辅音）3–25 帧、音量也下陷 ≥ 2.5 dB → 强换字点
CONSONANT_FRAMES = (3, 25)
DIP_WINDOW = 40             # find_peaks 的 wlen（帧）
DIP_RECOVER = 15            # 下陷之后最多往后找这么多帧，找音量回升到一半的那一帧
LEAK_FRAC = 0.5             # GAME 的音，时值里一半以上不是主唱在唱 → 丢掉
AGREE = 0.7                 # GAME 与 crepe（时值中间 60% 的平均）相差 ≤ 0.7 半音时取两者平均
CHORD_BIAS = 0.3            # 吸附时当时和弦里的音优先 0.3 半音
NEAR = 4                    # 起点 ±4 帧内有换字点 → 拨弦
CONTIGUOUS = 0.03           # 秒：两个音间隔不超过它算紧接
SCOOP = 0.2                 # 秒：拨弦音的起点最多往前拉到这么远的换字点
SHORT = 0.15                # 秒：字头短于它、后面紧跟 legato 音 → 是滑进来的
SHORT_SLIDE = 2             # 半音
MID_PENALTY = 0.06          # 拍：shuffle 里中间那个三连音位置（+1/3）的罚分
LEGATO_SLIDE = 3            # 半音：两个长音之间字内换音至少这么大才不重新拨（滑弦）；更小的照常拨弦
FILL_FRAMES = 5             # 帧：GAME 漏掉的短音至少有这么多主唱帧……
FILL_SPAN = 25              # 帧：……挤在这么短的一段里（更长的漏掉一般是 GAME 有意不要的滑音 / 气声）
FILL_STD = 0.5              # 半音：音高要稳
FILL_DB = -20.0             # 音量不能太小
FILL_STEP = 2               # 半音：紧接前一个音时只补一两个半音的"一步"；再大的、又轻（比前一个音弱 TAIL_DB）的是甩下去的尾巴
TAIL_DB = 6.0
STABLE_SLOPE = 20.0         # 半音 / 秒：音高斜率低于它算"唱稳了"（不在滑）
DRIFT_SEMITONES = 0.75      # 音里一段稳定地差这么多半音……
DRIFT_SLOTS = 0.6           # ……且至少这么多格长 → 是另一个音（_split_short_drift）
REPAIR_SEMITONES = 0.75     # 量化后最后核对：主唱稳定地唱着、和谱上差 ≥ 0.75 半音（或谱上没音）……
REPAIR_SLOTS = 1.0          # ……至少这么多格长 → 谱上补 / 改成唱的那个音（_repair_mismatches）


@dataclass
class Frames:
    """主唱的逐帧特征（FPS 帧 / 秒）。voiced = 主唱确实在唱：周期性 > 0.45、音量 > −32 dB、不是伴唱漏音、crepe 有音高。"""
    midi: np.ndarray        # crepe 音高（MIDI 浮点；没有音高的帧是 nan）
    per: np.ndarray         # crepe 周期性
    db: np.ndarray          # 主唱音量（相对最响处，3 帧平滑）
    leak: np.ndarray        # 伴唱漏音
    voiced: np.ndarray      # 主唱在唱
    onset: np.ndarray       # 换字点（syllable_onsets）


def _frame(t: float) -> int:
    return int(round(t * FPS))


def middle(i: int, j: int) -> tuple[int, int]:
    """帧区间 [i, j) 去掉头尾各 20%（滑进滑出）的中间部分，至少一帧。"""
    k = int(0.2 * (j - i))
    return i + k, max(i + k + 1, j - k)


def lr_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """逐帧（以 k·HOP 为中心、FRAME_LENGTH 个采样）左右声道的相关系数：居中的主唱接近 1，摆在两边的伴唱低。"""
    from numpy.lib.stride_tricks import sliding_window_view

    pad = FRAME_LENGTH // 2
    fl, fr = (sliding_window_view(np.pad(x, pad), FRAME_LENGTH)[::HOP] for x in (left, right))
    lr, ll, rr = (np.einsum("ij,ij->i", a, b) for a, b in ((fl, fr), (fl, fl), (fr, fr)))
    return lr / np.sqrt(ll * rr + 1e-12)


def _dips(db: np.ndarray, prominence: float) -> tuple[np.ndarray, np.ndarray]:
    """音量下陷（-db 的峰）及其突出度（前后 DIP_WINDOW 帧内）。平的地方（长段安静）会有突出度 0 的峰，
    本来就被 prominence 筛掉，不必警告。"""
    with warnings.catch_warnings():   # scipy 的 PeakPropertyWarning（不在公开接口里，按消息过滤）
        warnings.filterwarnings("ignore", message="some peaks have a prominence of 0")
        peaks, props = find_peaks(-db, prominence=prominence, wlen=DIP_WINDOW)
    return peaks, props["prominences"]


def _recovery(db: np.ndarray, p: int, target: float) -> int:
    """音量下陷 p 之后回升到 target 的第一帧（最多往后 DIP_RECOVER 帧）。"""
    q = p
    while q < len(db) - 1 and db[q] < target and q - p < DIP_RECOVER:
        q += 1
    return q


def syllable_onsets(db: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    """换字点（逐帧 bool）：停了 ≥ ONSET_GAP 帧（或从头起）之后重新出声的那一帧；音量下陷（突出度 ≥ ONSET_DIP dB）
    之后回升到前后最响处与谷底一半的那一帧，若那里还没出声就再等 ≤ 10 帧到出声。"""
    n = len(db)
    onset = np.zeros(n, bool)
    last = np.maximum.accumulate(np.where(voiced, np.arange(n), -1))   # 每帧及之前最后一个在唱的帧
    starts = np.flatnonzero(voiced[1:] & ~voiced[:-1]) + 1
    before = last[starts - 1]
    onset[starts[(before < 0) | (starts - 1 - before >= ONSET_GAP)]] = True
    for p in _dips(db, ONSET_DIP)[0]:
        q = _recovery(db, p, db[p] + (db[max(0, p - 20):p + 20].max() - db[p]) / 2)
        j = q
        while j < n - 1 and not voiced[j] and j - q < 10:
            j += 1
        if voiced[j]:
            onset[j] = True
    return onset


def strong_onsets(db: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    """强换字点（帧号，升序）：GAME 并在一起的两个字从这里切开。音量下陷 ≥ SPLIT_DIP dB 之后回升到一半的那一帧；
    或无声（辅音）3–25 帧、前后 10 帧的音量比无声处高 ≥ GAP_DIP dB、之后接着唱的第一帧
    （只有周期性掉、音量不掉的是沙哑 / 气声，不算）。"""
    strong = {_recovery(db, p, db[p] + prom / 2) for p, prom in zip(*_dips(db, SPLIT_DIP))}
    rises = np.flatnonzero(~voiced[:-1] & voiced[1:]) + 1
    for i in np.flatnonzero(voiced[:-1] & ~voiced[1:]) + 1:          # 无声段 [i, j)
        k = np.searchsorted(rises, i)
        if k == len(rises):
            break
        j = rises[k]
        if CONSONANT_FRAMES[0] <= j - i <= CONSONANT_FRAMES[1] and voiced[j:j + 3].all():
            if min(db[max(0, i - 10):i].max(), db[j:j + 10].max()) - db[i:j].min() >= GAP_DIP:
                strong.add(int(j))
    return np.array(sorted(strong), dtype=int)


def signal_frames(midi, per, rms_lead, rms_backing, lrcorr) -> Frames:
    """crepe 的逐帧音高 / 周期性 + 主唱 / 伴唱轨的 RMS + 主唱的左右相关 → Frames（都按 crepe 的帧数截齐、不够补 0）。"""
    midi, per = np.asarray(midi, float), np.asarray(per, float)
    n = len(midi)

    def fit(a) -> np.ndarray:
        a = np.asarray(a, float)[:n]
        return np.pad(a, (0, n - len(a)))

    rl, rb = fit(rms_lead), fit(rms_backing)
    ref = max(float(rl.max(initial=0.0)), 1e-12)
    db = np.convolve(20 * np.log10(np.maximum(rl, 1e-6) / ref), np.ones(3) / 3, mode="same")
    db_b = 20 * np.log10(np.maximum(rb, 1e-6) / ref)
    leak = (db < db_b - LEAK_DB) | (np.convolve(fit(lrcorr), np.ones(5) / 5, mode="same") < LR_CORR_MIN)
    voiced = (per > VOICED_PERIODICITY) & (db > VOICED_DB) & ~leak & ~np.isnan(midi)
    return Frames(midi=midi, per=per, db=db, leak=leak, voiced=voiced, onset=syllable_onsets(db, voiced))


def frame_features(lead_raw: Path, backing: Path, f0_cache: Path | None = None) -> Frames:
    """模型主唱 lead_raw.wav（立体声）+ 伴唱 backing.wav → Frames。crepe 基频走 cached_vocal_pitch
    （f0_cache = analysis/vocal_f0.npz）；单声道的主唱没有左右相关可看，只按伴唱轨找漏音。"""
    import librosa

    _, midi, per = cached_vocal_pitch(lead_raw, f0_cache) if f0_cache else vocal_pitch(lead_raw)
    y = np.atleast_2d(librosa.load(str(lead_raw), sr=SR, mono=False)[0])
    yb = librosa.load(str(backing), sr=SR, mono=True)[0]
    lrc = lr_correlation(y[0], y[1]) if len(y) > 1 else np.ones(len(midi))
    return signal_frames(midi, per, _rms(y.mean(axis=0)), _rms(yb), lrc)


def _rms(y: np.ndarray) -> np.ndarray:
    import librosa

    return librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP)[0]


def bar_chord_labels(beats: dict, chords: dict, chord_overrides=None) -> dict[int, list[tuple[int, str]]]:
    """每小节的和弦 {小节号: [(拍, 标签)]}，与 build_leadsheet 同一套：chords.json 的和弦段，overrides.chords 整小节覆盖。"""
    bpb = beats["time_signature"][0]
    manual = parse_chord_overrides(chord_overrides, {b["bar"] for b in beats["bars"]}, bpb)
    by_bar: dict[int, list[tuple[int, str]]] = {}
    for s in chords["segments"]:
        by_bar.setdefault(s["bar"], []).append((s["beat"], s["label"]))
    out = {}
    for b in beats["bars"]:
        n = b["bar"]
        if n in manual:
            segs = manual_segments(manual[n], bpb, _bar_beats(b, bpb, beats.get("pad_beats", 0)))
            out[n] = [(s["beat"], s["label"]) for s in segs]
        else:
            out[n] = by_bar.get(n, [])
    return out


def chord_tones_at(bar_labels: dict[int, list[tuple[int, str]]], bpb: int, beat: float) -> set[int]:
    """拍轴上 beat（0 = 第 1 小节第 1 拍）那一刻的和弦音；第 1 小节之前按第 1 小节、最后一小节之后按最后一小节；
    无和弦（N）或还没有和弦时是空集。"""
    bars = sorted(bar_labels)
    if not bars:
        return set()
    label = None
    for b, lab in bar_labels[bars[min(len(bars) - 1, max(0, int(beat // bpb)))]]:
        if b - 1 <= beat % bpb + 1e-6:
            label = lab
    return theory.chord_tones(label) if label else set()


def snap_chord(p: float, key_pcs: set[int], tones: set[int]) -> tuple[int, bool]:
    """浮点音高 → (整数音高, 调外)。候选：1.5 半音内的调内音或当时和弦里的音（如 D#/G 上的 G），和弦音优先 CHORD_BIAS 半音
    （夹在两个候选中间时取和弦音）；最好的候选在 SNAP_SEMITONES + CHORD_BIAS 内就用它（不算调外），否则退回 snap_to_key。
    所以只有既不在调里、也不在当时和弦里的音才标调外。"""
    cands = [c for c in range(math.floor(p) - 2, math.ceil(p) + 3)
             if (c % 12 in key_pcs or c % 12 in tones) and abs(c - p) <= 1.5]
    if cands:
        best = min(cands, key=lambda c: abs(c - p) - (CHORD_BIAS if c % 12 in tones else 0))
        if abs(best - p) <= SNAP_SEMITONES + CHORD_BIAS:
            return best, False
    return snap_to_key(p, key_pcs)


def quantize_start(b: float, div: int, shuffle: bool) -> float:
    """起点（拍）吸附到格（每拍 div 格）；shuffle 里中间那个三连音位置（+1/3）加罚 MID_PENALTY 拍：
    唱在 +1/3 和 +2/3 正中间时取 +2/3（shuffle 的反拍）。"""
    base = math.floor(b)
    cands = [base + k / div for k in range(-1, div + 2)]
    return min(cands, key=lambda c: abs(c - b) + (MID_PENALTY if shuffle and abs(c % 1 - 1 / 3) < 1e-6 else 0))


@dataclass
class _Note:
    a: float                    # 起点（秒）
    e: float                    # 终点（秒）
    midi: int
    out: bool                   # 调外（既不在调里、也不在当时和弦里）
    conf: float                 # 平均周期性
    forced: bool = False        # 强换字点切出来的后半段：一定重新拨弦
    legato_forced: bool = False  # 长音里音高变了切出来的后半段：一定不重新拨
    onset: bool = False         # 重新拨弦（否则 legato）


def _snapper(beats: dict, key_pcs: set[int], bar_labels: dict):
    """(浮点音高, 秒) → snap_chord（按那一刻的和弦）。"""
    bpb = beats["time_signature"][0]
    return lambda p, t: snap_chord(p, key_pcs, chord_tones_at(bar_labels, bpb, float(time_to_beat(beats, t))))


def _from_game(rows, fr: Frames, snap) -> list[_Note]:
    """GAME 的音 → 候选音：时值里一半以上不是主唱在唱的丢掉；GAME 与 crepe（中间 60% 的平均）相差 ≤ AGREE 时取平均；
    再按起点那一刻的和弦吸附。"""
    notes = []
    for a, e, p in rows:
        i = _frame(a)
        j = max(i + 1, _frame(e))
        sung = fr.voiced[i:j]
        if sung.size == 0 or sung.mean() < 1 - LEAK_FRAC:
            continue
        mi, mj = middle(i, j)
        seg = fr.voiced[mi:mj]
        if seg.sum() >= 3:
            c = float(np.mean(fr.midi[mi:mj][seg]))
            if abs(c - p) <= AGREE:
                p = (p + c) / 2
        m, out = snap(p, a)
        notes.append(_Note(a, e, m, out, float(np.mean(fr.per[i:j]))))
    return notes


def _split_at_onsets(notes: list[_Note], fr: Frames, slot_s: float) -> list[_Note]:
    """GAME 并在一起的两个字在强换字点切开（后半段一定重新拨）：切点离两头至少 0.8 格，切点起 3 帧主唱都在唱。"""
    strong = strong_onsets(fr.db, fr.voiced)
    out = []
    for x in notes:
        cuts = [c / FPS for c in strong if x.a + .8 * slot_s <= c / FPS <= x.e - .8 * slot_s and fr.voiced[c:c + 3].all()]
        edges = [x.a, *cuts, x.e]
        out += [replace(x, a=a, e=e, forced=k > 0) for k, (a, e) in enumerate(zip(edges, edges[1:]))]
    return out


def _half_beat_times(beats: dict, a: float, e: float) -> list[float]:
    sb, eb = time_to_beat(beats, np.array([a, e]))
    k0, k1 = int(np.ceil(sb * 2 - 1e-6)), int(np.floor(eb * 2 + 1e-6))
    return [float(t) for t in beat_to_time(beats, np.arange(k0, k1 + 1) / 2)]


def _drift_segments(x: _Note, hb: list[float], fr: Frames, snap) -> list[list]:
    """按半拍 hb 取主唱的平均音高、按当时和弦吸附，连续同音的半拍成段 [起, 止, 音高, 调外]（半拍下标）；
    不到 1 拍（2 个半拍）的段并进前一段，开头不到 0.1 秒的一小段并进后面。唱得太少的半拍沿用前一段。"""
    segs: list[list] = []
    for k, (a, e) in enumerate(zip(hb, hb[1:])):
        i, j = _frame(a), _frame(e)
        sung = fr.voiced[i:j]
        lab = snap(float(np.mean(fr.midi[i:j][sung])), (a + e) / 2) if sung.sum() >= 3 else None
        lab = lab or (tuple(segs[-1][2:]) if segs else (x.midi, x.out))
        if segs and segs[-1][2] == lab[0]:
            segs[-1][1] = k + 1
        else:
            segs.append([k, k + 1, *lab])
    merged: list[list] = []
    for s in segs:
        if merged and s[1] - s[0] < 2:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    if len(merged) > 1 and merged[0][1] - merged[0][0] < 2 and hb[merged[0][1]] - hb[0] < 0.1:
        merged[1][0] = 0
        merged = merged[1:]
    return merged


def _split_drift(notes: list[_Note], fr: Frames, beats: dict, snap) -> list[_Note]:
    """长音（≥ 3 个半拍）里唱的音高真的变了（如 D#/G 上 G#4 → G4，连续 ≥ 1 拍）→ 在那儿切开，后半段 legato；
    只有一段时整个音改成实际唱的音高（逐半拍、按当时和弦）。"""
    out = []
    for x in notes:
        hb = [x.a, *[t for t in _half_beat_times(beats, x.a, x.e) if x.a + 0.05 < t < x.e - 0.05], x.e]
        if len(hb) < 4:
            out.append(x)
            continue
        segs = _drift_segments(x, hb, fr, snap)
        if len(segs) == 1:
            out.append(replace(x, midi=segs[0][2], out=segs[0][3]))
            continue
        out += [replace(x, a=hb[s0], e=hb[s1], midi=m, out=o, forced=x.forced and k == 0, legato_forced=k > 0)
                for k, (s0, s1, m, o) in enumerate(segs)]
    return out


def _stable(fr: Frames) -> np.ndarray:
    """主唱在唱、而且音高没在滑（5 帧平滑后的斜率 < STABLE_SLOPE 半音 / 秒）的帧。"""
    sm = np.convolve(np.nan_to_num(fr.midi), np.ones(5) / 5, mode="same")
    return fr.voiced & (np.abs(np.gradient(sm)) * FPS < STABLE_SLOPE)


def _runs(mask: np.ndarray, max_gap: int = 2) -> list[tuple[int, int]]:
    """mask 里连续为真的段 [起, 止)（中间允许 max_gap 帧以内的假）。"""
    runs, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and (mask[j] or mask[j + 1:j + 1 + max_gap].any()):
            j += 1
        runs.append((i, j))
        i = j
    return runs


def _split_short_drift(notes: list[_Note], fr: Frames, snap, slot_s: float) -> list[_Note]:
    """音里有一段唱得稳、却和这个音实际唱的中心（中位数）差 ≥ DRIFT_SEMITONES 半音、至少 DRIFT_SLOTS 格长（不是颤音的一个峰）→ 那一段是另一个音
    （如长音后面落到低一个全音、句尾往上一挑）：切出来，按当时和弦吸附；前后剩下不到半格的并进它。切出来的段和后面的段
    没有换字就是 legato（拨不拨弦再由 _legato_policy 定）。"""
    stable = _stable(fr)
    min_len = int(round(DRIFT_SLOTS * slot_s * FPS))
    out: list[_Note] = []
    for x in notes:
        i, j = _frame(x.a), _frame(x.e)
        sung = fr.midi[i:j][fr.voiced[i:j]]
        if not len(sung):
            out.append(x)
            continue
        centre = float(np.median(sung))        # 跟这个音实际唱的中心比，不跟写的音高比：唱在两个音中间时不来回切
        off = stable[i:j] & (np.abs(fr.midi[i:j] - centre) >= DRIFT_SEMITONES)
        cuts = []
        for a, b in _runs(off):
            m = fr.midi[i + a:i + b][stable[i + a:i + b]]
            if b - a >= min_len and len(m) and float(np.std(m)) < FILL_STD:
                p, o = snap(float(np.median(m)), (i + a) / FPS)
                if p != x.midi:
                    cuts.append((i + a, i + b, p, o))
        if not cuts:
            out.append(x)
            continue
        pieces, t = [], i                       # [起帧, 止帧, 音高, 调外]；不到半格的零头并进后面那段
        for a, b, p, o in cuts:
            if a - t >= min_len / 2:
                pieces.append([t, a, x.midi, x.out])
                t = a
            pieces.append([t, b, p, o])
            t = b
        if j - t >= min_len / 2:
            pieces.append([t, j, x.midi, x.out])
        else:
            pieces[-1][1] = j
        for k, (a, b, p, o) in enumerate(pieces):
            onset = k > 0 and bool(fr.onset[max(0, a - NEAR):a + NEAR + 1].any())
            out.append(replace(x, a=a / FPS, e=b / FPS, midi=p, out=o, forced=(x.forced if k == 0 else onset),
                               legato_forced=x.legato_forced if k == 0 else not onset))
    return out


def _has_onset(fr: Frames, t: float) -> bool:
    """t 附近 ±NEAR 帧内有换字点，或前面 ONSET_GAP 帧都没在唱。"""
    i = _frame(t)
    return bool(fr.onset[max(0, i - NEAR):i + NEAR + 1].any()) or not fr.voiced[max(0, i - ONSET_GAP):i].any()


def _merge_held(notes: list[_Note], fr: Frames) -> list[_Note]:
    """拨不拨弦：起点有换字（_has_onset）或是强换字点切出来的 → 拨，长音里切出来的 → 不拨；和前一个音不相接的一定拨。
    紧接前一个音、同音高、又不拨 → 同一个长音，并进前一个音（他唱了很久的一个音，只弹一下等它延音）。"""
    merged: list[_Note] = []
    for x in notes:
        x = replace(x, onset=(_has_onset(fr, x.a) or x.forced) and not x.legato_forced)
        if merged:
            prev = merged[-1]
            contiguous = x.a - prev.e <= CONTIGUOUS
            if contiguous and prev.midi == x.midi and not x.onset:
                prev.e = x.e
                continue
            x.onset = x.onset or not contiguous
        else:
            x.onset = True
        merged.append(x)
    return merged


def _scoop(notes: list[_Note], fr: Frames) -> list[_Note]:
    """拨弦音的起点往前拉到 SCOOP 秒内（不越过前一个音的终点）最早的换字点，中间要一直在唱：
    GAME 常把字头的上滑（しゃくり）切掉，起点晚了。"""
    out = []
    for k, x in enumerate(notes):
        if x.onset:
            i = _frame(x.a)
            lo = max(_frame(notes[k - 1].e) if k else 0, i - _frame(SCOOP))
            cand = np.flatnonzero(fr.onset[lo:i])
            if cand.size and fr.voiced[lo + cand[0]:i].all():
                x = replace(x, a=(lo + cand[0]) / FPS)
        out.append(x)
    return out


def _absorb_short(notes: list[_Note]) -> list[_Note]:
    """字头短于 SHORT 秒的拨弦音、后面紧跟 SHORT_SLIDE 半音内的 legato 音：是滑进来的，并进后面那个音（起点提前、由它拨）。"""
    work = [replace(x) for x in notes]
    out = []
    for x, nxt in zip(work, [*work[1:], None]):
        if (x.onset and nxt is not None and not nxt.onset and x.e - x.a < SHORT
                and abs(nxt.midi - x.midi) <= SHORT_SLIDE):
            nxt.a, nxt.onset = x.a, True
            continue
        out.append(x)
    return out


def _fill_gaps(notes: list[_Note], fr: Frames, snap) -> list[_Note]:
    """GAME 漏掉的短音（如 なんもねえ 第 8 小节 C#4 后面轻轻落到 B3）：没被任何音盖住的主唱帧（中间允许 2 帧以内的无声）
    ≥ FILL_FRAMES 帧、整段不超过 FILL_SPAN 帧、音高稳（标准差 < FILL_STD）、平均音量 > FILL_DB → 补一个音。
    紧接前一个音（≤ CONTIGUOUS 秒）时：音高差超过 FILL_STEP 半音是甩下去的尾巴，不补；同音并进前一个音；
    否则没有换字就是 legato。"""
    n = len(fr.voiced)
    covered = np.zeros(n, bool)
    for x in notes:
        covered[max(0, _frame(x.a)):min(n, _frame(x.e))] = True
    free = fr.voiced & ~covered
    runs, i = [], 0
    while i < n:
        if not free[i]:
            i += 1
            continue
        j = i + 1
        while j < n and (free[j] or (not covered[j] and j + 1 < n and free[j + 1:j + 3].any() and not fr.voiced[j])):
            j += 1
        runs.append((i, j))
        i = j
    out = [replace(x) for x in notes]
    for i, j in runs:
        v = fr.voiced[i:j]
        m = fr.midi[i:j][v]
        if v.sum() < FILL_FRAMES or j - i > FILL_SPAN or fr.db[i:j].mean() <= FILL_DB:
            continue
        if float(np.std(m)) >= FILL_STD:           # 滑上去 / 滑下来的一个字：按落点（最后 6 帧）的音高，落点要稳
            m = m[-6:]
            if len(m) < 6 or float(np.std(m)) >= FILL_STD:
                continue
        a, e = i / FPS, j / FPS
        pitch, out_ = snap(float(np.median(m)), a)
        prev = max((x for x in out if x.e <= a + 1e-6), key=lambda x: x.e, default=None)
        contiguous = prev is not None and a - prev.e <= CONTIGUOUS + 1 / FPS
        if contiguous and abs(pitch - prev.midi) > FILL_STEP and \
                fr.db[i:j].mean() < fr.db[_frame(prev.a):_frame(prev.e)].mean() - TAIL_DB:
            continue
        if contiguous and pitch == prev.midi:
            prev.e = e
            continue
        onset = not contiguous or bool(fr.onset[max(0, i - NEAR):i + NEAR + 1].any())
        out.append(_Note(a, e, pitch, out_, float(np.mean(fr.per[i:j])), onset=onset))
    return sorted(out, key=lambda x: x.a)


def _quantize(notes: list[_Note], beats: dict, div: int, shuffle: bool) -> list[dict]:
    """起点 quantize_start，终点四舍五入到格、至少一格；撞格（起点不晚于前一个音）的往后挪一格，前一个音截到它的起点。
    legato 的音带 "legato": True，其余不带这个键。"""
    res: list[dict] = []
    for x in notes:
        s, e = (float(v) for v in time_to_beat(beats, np.array([x.a, x.e])))
        qs = quantize_start(s, div, shuffle)
        qe = max(qs + 1 / div, round(e * div) / div)
        if res:
            prev = res[-1]
            if qs <= prev["start_beats"] + 1e-6:
                qs = prev["start_beats"] + 1 / div
                qe = max(qe, qs + 1 / div)
            if qs < prev["start_beats"] + prev["dur_beats"] - 1e-6:
                prev["dur_beats"] = _span(prev["start_beats"], qs)
        res.append({"start_beats": round(qs, 4), "dur_beats": _span(qs, qe), "midi": x.midi, "out_of_key": x.out,
                    "conf": round(x.conf, 3), **({} if x.onset else {"legato": True})})
    return res


def _merge_legato(notes: list[dict]) -> list[dict]:
    """量化后紧接前一个音的同音高 legato 音并进前一个音。"""
    out: list[dict] = []
    for x in notes:
        if out and x.get("legato") and out[-1]["midi"] == x["midi"] \
                and abs(out[-1]["start_beats"] + out[-1]["dur_beats"] - x["start_beats"]) < 1e-3:
            out[-1]["dur_beats"] = _span(out[-1]["start_beats"], x["start_beats"] + x["dur_beats"])
            continue
        out.append(x)
    return out


def _scoop_grace(notes: list[dict], div: int) -> list[dict]:
    """拨弦的一格短音、紧接着高 1–SHORT_SLIDE 半音的 legato 音：是字头往上滑（しゃくり），合成一个音，用后面那个音高、
    由它拨（起点是短音的起点）。"""
    out: list[dict] = []
    i = 0
    while i < len(notes):
        x, nxt = notes[i], notes[i + 1] if i + 1 < len(notes) else None
        if (nxt is not None and not x.get("legato") and nxt.get("legato") and x["dur_beats"] <= 1 / div + 1e-6
                and 1 <= nxt["midi"] - x["midi"] <= SHORT_SLIDE
                and abs(x["start_beats"] + x["dur_beats"] - nxt["start_beats"]) < 1e-3):
            merged = {k: v for k, v in nxt.items() if k != "legato"}
            merged.update(start_beats=x["start_beats"], dur_beats=_span(x["start_beats"], nxt["start_beats"] + nxt["dur_beats"]))
            out.append(merged)
            i += 2
            continue
        out.append(x)
        i += 1
    return out


def _legato_policy(notes: list[dict], div: int) -> list[dict]:
    """字内换音只在两种情况下不重新拨：快的装饰音（两个音里有一个 ≤ 1 格，用击 / 勾）和 ≥ LEGATO_SLIDE 半音的滑音（滑弦）。
    两个长音之间一两个半音的换音照常拨弦——用户听过：处处击勾 / 滑弦反而不像唱的。"""
    out: list[dict] = []
    for x in notes:
        if x.get("legato") and out:
            prev = out[-1]
            short = min(prev["dur_beats"], x["dur_beats"]) <= 1 / div + 1e-6
            if not (short or abs(x["midi"] - prev["midi"]) >= LEGATO_SLIDE):
                x = {k: v for k, v in x.items() if k != "legato"}
        out.append(x)
    return out


def _repair_mismatches(notes: list[dict], fr: Frames, beats: dict, snap, div: int, shuffle: bool) -> list[dict]:
    """量化后逐帧核对谱和主唱（就是 melody_audit / 人耳查漏音时看的东西）：主唱稳稳唱着（音高不在滑、标准差 < FILL_STD、
    音量 > FILL_DB）、和谱上这时的音差 ≥ REPAIR_SEMITONES 半音（或谱上是休止）、至少 REPAIR_SLOTS 格长 → 漏了一个音：
    在那几格写上唱的音（按当时和弦吸附），原来的音前后剩下的部分保留（不到一格的并掉）。补进来的音有换字点就拨弦，
    否则 legato；原来的音的后半段一定是 legato（同一个字）。"""
    stable = _stable(fr)
    n = len(fr.voiced)
    slot = 1 / div
    spans = [beat_to_time(beats, np.array([x["start_beats"], x["start_beats"] + x["dur_beats"]])) for x in notes]
    written = np.full(n, np.nan)
    for x, (a, e) in zip(notes, spans):
        written[max(0, _frame(a)):min(n, _frame(e))] = x["midi"]
    off = stable & (np.isnan(written) | (np.abs(fr.midi - written) >= REPAIR_SEMITONES))
    min_len = int(round(REPAIR_SLOTS * 60 / beats["tempo_bpm"] / div * FPS))
    work = [dict(x) for x in notes]
    for i, j in _runs(off):
        m = fr.midi[i:j][stable[i:j]]
        if j - i < min_len or not len(m) or float(np.std(m)) >= FILL_STD or fr.db[i:j].mean() <= FILL_DB:
            continue
        p, o = snap(float(np.median(m)), i / FPS)
        if o:                                   # 既不在调里、也不在和弦里（多半是唱偏了）：不改谱
            continue
        qs = round(quantize_start(float(time_to_beat(beats, i / FPS)), div, shuffle), 4)
        qe = round(max(qs + slot, round(float(time_to_beat(beats, j / FPS)) * div) / div), 4)
        onset = bool(fr.onset[max(0, i - NEAR):i + NEAR + 1].any())
        host = next((x for x in work if x["start_beats"] - 1e-3 <= qs < x["start_beats"] + x["dur_beats"] - 1e-3), None)
        if host is not None and host["midi"] == p:
            continue
        new = {"start_beats": round(qs, 4), "dur_beats": _span(qs, qe), "midi": p, "out_of_key": o,
               "conf": round(float(np.mean(fr.per[i:j])), 3), **({} if onset else {"legato": True})}
        if host is None:
            nxt = min((x["start_beats"] for x in work if x["start_beats"] > qs + 1e-3), default=None)
            if nxt is not None and qe > nxt:
                new["dur_beats"] = _span(qs, nxt)
            prev = max((x for x in work if x["start_beats"] < qs - 1e-3), key=lambda x: x["start_beats"], default=None)
            if prev is not None and abs(prev["start_beats"] + prev["dur_beats"] - qs) > 1e-3 and "legato" in new:
                del new["legato"]                                   # 前面空着：不能连，拨弦
            work.append(new)
        else:
            h_end = host["start_beats"] + host["dur_beats"]
            if qs - host["start_beats"] < slot - 1e-3:                # 前面剩下不到一格：整段从原来的音的起点开始
                new.update(start_beats=host["start_beats"], dur_beats=_span(host["start_beats"], qe))
                if host.get("legato"):
                    new["legato"] = True                            # 原来就不拨的地方照旧不拨
                work.remove(host)
            else:
                host["dur_beats"] = _span(host["start_beats"], qs)
            if h_end - qe >= slot - 1e-3:
                work.append({**{k: v for k, v in host.items() if k != "legato"}, "start_beats": round(qe, 4),
                             "dur_beats": _span(qe, h_end), "legato": True})
            else:
                new["dur_beats"] = _span(new["start_beats"], h_end)
            work.append(new)
        work.sort(key=lambda x: x["start_beats"])
    return work


def game_melody(rows, fr: Frames, beats: dict, key_pcs: set[int], bar_labels: dict) -> list[dict]:
    """GAME 的音 [(起秒, 止秒, 浮点音高)] + 主唱逐帧特征 → 旋律音（start_beats / dur_beats / midi / out_of_key / conf，
    legato 的音另有 "legato": True）。去掉第 1 小节之前的音。"""
    shuffle = beats.get("feel") == "shuffle"
    div = 3 if shuffle else 4
    snap = _snapper(beats, key_pcs, bar_labels)
    notes = _from_game(rows, fr, snap)
    notes = _split_at_onsets(notes, fr, 60 / beats["tempo_bpm"] / div)
    notes = _split_short_drift(_split_drift(notes, fr, beats, snap), fr, snap, 60 / beats["tempo_bpm"] / div)
    notes = _fill_gaps(_absorb_short(_scoop(_merge_held(notes, fr), fr)), fr, snap)
    notes = _scoop_grace(_merge_legato(_quantize(notes, beats, div, shuffle)), div)
    notes = _legato_policy(_merge_legato(_repair_mismatches(notes, fr, beats, snap, div, shuffle)), div)
    return [x for x in notes if x["start_beats"] >= 0]


def extract_melody_v7(lead_raw: Path, backing: Path, game_rows, beats_json: dict, key_pcs: set[int],
                      bar_labels: dict, f0_cache: Path | None = None) -> dict:
    """旋律 v7：lead_raw.wav（模型主唱原样）、backing.wav（模型伴唱）、GAME 的音（game.game_notes）、beats.json、调、
    每小节和弦（bar_chord_labels）→ {"grid", "notes"}，格式同 extract_melody，legato 的音多一个 "legato": True。"""
    notes = game_melody(game_rows, frame_features(lead_raw, backing, f0_cache), beats_json, key_pcs, bar_labels)
    return {"grid": "triplet8" if beats_json.get("feel") == "shuffle" else "sixteenth", "notes": notes}
