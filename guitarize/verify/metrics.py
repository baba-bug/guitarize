"""编配 vs 参考（leadsheet.json）的逐小节评分。
弹唱 / 指弹：和弦一致率、旋律（指弹按写法：完整旋律与旋律谱一样考全部参考音，骨架写法只考骨架音）、
与和声分轨的色度相似度、节奏（弹唱）。
旋律谱（melody）：只评旋律（召回按全部参考音）和与人声分轨的色度相似度，并列出要人耳确认的参考音；不评和弦。
卡祖笛谱（kazoo）：只评旋律（与旋律谱同一套评法），另报音域和换气（kazoo_stats）；不评和弦、不对照音频。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .. import theory
from ..config import Thresholds
from ..skeleton import skeleton_mask
from .model import Attack, Bar, Score
from .positions import beat_label

MELODY_TOL = 1 / 3 + 1e-6  # 起音容差（拍）：指弹（骨架音按八分 / 三连音格放，摇摆位置有出入）
# 旋律谱的起音容差：谱上照原唱落到三连音 / 十六分格（演奏时间，三连音格精确），最大落格误差是半格 1/6 拍；
# 差一个三连音格（1/3 拍）就不算命中。1e-3 容下 leadsheet 拍位置 4 位小数的舍入
LINE_TOL = 1 / 6 + 1e-3
SLOTS = 12                 # 4/4 小节的三连音八分格数
LINE_ARRS = ("melody", "kazoo")   # 单旋律线的谱：不评和弦，旋律召回考全部参考音、达标线 melody_line


@dataclass
class BarScore:
    bar: int  # 1-based
    section: str | None = None
    chord_agree: int = 0
    chord_total: int = 0
    chord_issues: list[str] = field(default_factory=list)
    low_conf_beats: list[int] = field(default_factory=list)
    mel_hit: int = 0           # 编配旋律音命中参考（精确率分子）
    mel_est: int = 0           # 编配旋律音数（精确率分母）
    mel_found: int = 0         # 参考骨架音被覆盖（召回率分子；旋律谱：全部参考音）
    mel_ref: int = 0           # 参考骨架音数（召回率分母；旋律谱：全部参考音）
    mel_octave_exact: int = 0
    mel_issues: list[str] = field(default_factory=list)
    mark_issues: list[str] = field(default_factory=list)  # 谱上和弦名与实际音不符
    harmonic_sim: float | None = None
    rhythm_corr: float | None = None
    vocal_sim: float | None = None                         # 旋律谱：与人声分轨的色度相似度
    ear_notes: list[str] = field(default_factory=list)    # 旋律谱：参考里调外 / 低置信、要人耳确认的音

    @property
    def chord_ratio(self) -> float | None:
        return self.chord_agree / self.chord_total if self.chord_total else None

    @property
    def mel_f1(self) -> float | None:
        return f1(self.mel_hit, self.mel_est, self.mel_found, self.mel_ref)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["chord_ratio"] = self.chord_ratio
        d["mel_f1"] = self.mel_f1
        return d


def f1(hit: int, est: int, found: int, ref: int) -> float | None:
    if ref == 0:
        return None  # 参考侧没有可核对的骨架音：召回率无定义，不能记 0 分（est == 0 and ref == 0 同理）
    r = found / ref
    if est == 0:
        return r  # 本小节没有编配旋律音：精确率无定义，取召回率（骨架音被邻小节的音覆盖也不算问题）
    p = hit / est
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


# ---------- 和弦 ----------

def _ref_beats(lbar: dict, bpb: int) -> list[tuple[str, str, str]]:
    """小节内逐拍参考：(majmin, 置信度, 原标签)。置信度 high / medium / low / manual（人耳纠正）。"""
    out = [("N", "high", "N")] * bpb
    for seg in lbar.get("chords", []):
        first = int(round(seg["beat"])) - 1
        n = max(1, int(round(seg["dur_beats"])))
        for k in range(max(0, first), min(bpb, first + n)):
            out[k] = (theory.majmin(seg["label"]), seg.get("confidence", "high"), seg["label"])
    return out


def _pc_weights(sounding, t0: float, t1: float) -> tuple[np.ndarray, int | None]:
    w = np.zeros(12)
    bass = None
    for s in sounding:
        overlap = min(s.end, t1) - max(s.start, t0)
        if overlap > 1e-6:
            w[s.midi % 12] += overlap
            if bass is None or s.midi < bass:
                bass = s.midi
    return w, (None if bass is None else bass % 12)


def _fifth(label: str) -> int | None:
    """和弦标签的五音音名（解析不了时 None）。"""
    try:
        parsed = theory.parse_chord(label)
    except ValueError:
        return None
    return None if parsed is None else (parsed[0] + 7) % 12


def score_chords(score: Score, leadsheet: dict, bars: list[BarScore]) -> None:
    bpb = int(leadsheet.get("beats_per_bar", 4))
    sounding = score.sounding()
    prev: set[str] | None = None
    for bs, lbar, sbar in zip(bars, leadsheet["bars"], score.bars):
        for k, (ref, conf, raw) in enumerate(_ref_beats(lbar, bpb)):
            t0 = sbar.start + k
            w, bass = _pc_weights(sounding, t0, t0 + 1)
            if bass is not None and ref != "N" and bass == _fifth(raw):
                bass = None   # 交替低音弹到参考和弦的五音（指弹的"根-五"低音）：不算转位，不给五音上的和弦加分
            if w.sum() > 0:
                cands = theory.infer_majmin(w, bass)
                prev = cands
            else:
                cands = prev or {"N"}  # 空拍：沿用上一拍的余音
            if ref == "N":
                continue
            if conf == "low":  # 只有 low 不计分；manual（overrides.chords）与 high 一样计分，manual 的 N 同样跳过
                bs.low_conf_beats.append(k + 1)
                continue
            bs.chord_total += 1
            if ref in cands:
                bs.chord_agree += 1
            else:
                shape = theory.transpose_label(raw, -score.capo) if score.capo else raw
                bs.chord_issues.append(f"拍 {beat_label(k)}：参考 {raw}（指法 {shape}），编配听起来是 {'/'.join(sorted(cands))}")


def chord_mark_issues(score: Score, bars: list[BarScore]) -> None:
    """谱上写的和弦名（换算到实际音高）与标记处一拍内实际发声的音不符 → 记入 mark_issues。"""
    sounding = score.sounding()
    for e in score.events:
        if not e.chord or e.bar >= len(bars):
            continue
        expected = theory.majmin(theory.transpose_label(e.chord, score.capo))
        w, bass = _pc_weights(sounding, e.start, e.start + 1)
        if expected == "N" or w.sum() == 0:
            continue
        cands = theory.infer_majmin(w, bass)
        if expected not in cands:
            pos = beat_label(e.start - score.bars[e.bar].start)
            bars[e.bar].mark_issues.append(f"拍 {pos} 标记 {e.chord}，实际音听起来是 {'/'.join(sorted(cands))}")


# ---------- 旋律 ----------

def _bar_of(score: Score, t: float) -> int | None:
    for b in score.bars:
        if b.start - 1e-6 <= t < b.start + b.length - 1e-6:
            return b.index
    return None


def _match(a: Sequence[tuple[float, int]], b: Sequence[tuple[float, int]], tol: float = MELODY_TOL) -> dict[int, int]:
    """a 中每个音找 b 中音名相同、起音最近（≤ tol）且未被占用的音。返回 {a 下标: b 下标}。"""
    used: set[int] = set()
    out: dict[int, int] = {}
    for i, (ta, ma) in enumerate(a):
        best = None
        for j, (tb, mb) in enumerate(b):
            if j in used or (ma - mb) % 12:
                continue
            d = abs(ta - tb)
            if d <= tol and (best is None or d < best[0]):
                best = (d, j)
        if best is not None:
            used.add(best[1])
            out[i] = best[1]
    return out


def score_melody(score: Score, leadsheet: dict, bars: list[BarScore], skeleton: bool = True) -> None:
    """精确率：编配旋律音命中任一参考音；召回率：skeleton 时只考参考骨架音（指弹只弹骨架），否则考全部参考音（旋律谱）。
    指弹（skeleton）：参考没有旋律的小节不计精确率（间奏里弹旋律线不算错）；起音容差 MELODY_TOL。
    旋律谱：被评分小节里谱上的每个音都计精确率，参考没有旋律的小节也算（人声没唱的地方谱上有音就是多余音）；
    起音容差 LINE_TOL。"""
    n = len(bars)
    tol = MELODY_TOL if skeleton else LINE_TOL
    ref_all = sorted((m for b in leadsheet["bars"][:n] for m in b.get("melody", [])), key=lambda m: m["start_beats"])
    mask = skeleton_mask([m["start_beats"] for m in ref_all], [m["dur_beats"] for m in ref_all])
    skel = [m for m, keep in zip(ref_all, mask) if keep or not skeleton]
    missed = "漏掉骨架音" if skeleton else "漏掉旋律音"
    ref_bars = {_bar_of(score, m["start_beats"]) for m in ref_all} - {None}
    # 匹配用被评分小节内的全部编配旋律音，这样落在邻小节（无参考旋律）的同名音也能覆盖参考骨架音；
    # 指弹“该小节有参考旋律”的过滤只用于精确率一侧的计数
    est = [e for e in score.melody(voice=0) if (_bar_of(score, e.start) is not None and _bar_of(score, e.start) < n)]

    hits = _match([(e.start, e.midi) for e in est], [(m["start_beats"], m["midi"]) for m in ref_all], tol)
    founds = _match([(m["start_beats"], m["midi"]) for m in skel], [(e.start, e.midi) for e in est], tol)

    for i, e in enumerate(est):
        bi = _bar_of(score, e.start)
        if skeleton and bi not in ref_bars:
            continue
        bs, pos = bars[bi], beat_label(e.start - score.bars[bi].start)
        bs.mel_est += 1
        if i in hits:
            bs.mel_hit += 1
            if ref_all[hits[i]]["midi"] == e.midi:
                bs.mel_octave_exact += 1
        else:
            bs.mel_issues.append(f"多余/错音：拍 {pos} {theory.midi_name(e.midi)}")
    for j, m in enumerate(skel):
        bi = _bar_of(score, m["start_beats"])
        if bi is None or bi >= n:
            continue
        bs, pos = bars[bi], beat_label(m["start_beats"] - score.bars[bi].start)
        bs.mel_ref += 1
        if j in founds:
            bs.mel_found += 1
        else:   # 吉他谱附上指法调音高；音高谱表（卡祖笛，没有弦）没有指法调
            shape = f"（指法调 {theory.midi_name(m['midi'] - score.capo)}）" if score.tuning else ""
            bs.mel_issues.append(f"{missed}：拍 {pos} {theory.midi_name(m['midi'])}{shape}")


def ear_check(leadsheet: dict, bars: list[BarScore], low_conf: float) -> None:
    """参考旋律里调外、或置信（平均周期性）低于 low_conf 的音 → 该小节的 ear_notes（要人耳确认的音）。
    overrides.melody 写的音（manual）已经人耳确认过，不列。"""
    bpb = int(leadsheet.get("beats_per_bar", 4))
    for i, (bs, lbar) in enumerate(zip(bars, leadsheet["bars"])):
        for m in lbar.get("melody", []):
            if m.get("manual"):
                continue
            why = (["调外"] if m.get("out_of_key") else []) + ([f"置信 {m['conf']:.2f}"] if m["conf"] < low_conf else [])
            if why:
                pos = beat_label(m["start_beats"] - i * bpb)
                bs.ear_notes.append(f"拍 {pos} {theory.midi_name(m['midi'])}（{'，'.join(why)}）")


# ---------- 和声相似度 ----------

def bar_chroma(wav: Path, windows: Sequence[tuple[float, float]]) -> np.ndarray:
    import librosa

    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)  # 不用 chroma_cens：它约 1 秒的平滑会把相邻小节糊在一起
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    out = np.full((len(windows), 12), np.nan)
    for i, (a, b) in enumerate(windows):
        m = (times >= a) & (times < b)
        if m.any():
            out[i] = chroma[:, m].mean(axis=1)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.isnan(a).any() or np.isnan(b).any():
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return None if na == 0 or nb == 0 else float(a @ b / (na * nb))


def _synth_windows(score: Score, n: int) -> list[tuple[float, float]]:
    return [(score.seconds(b.start), score.seconds(b.start + b.length)) for b in score.bars[:n]]


def _ref_windows(leadsheet: dict, n: int) -> list[tuple[float, float]]:
    return [(b["t_start"], b["t_end"]) for b in leadsheet["bars"][:n]]


def score_harmonic(score: Score, leadsheet: dict, bars: list[BarScore], synth_wav: Path, ref_wav: Path,
                   attr: str = "harmonic_sim") -> None:
    """逐小节色度余弦相似度：合成音 vs 参考分轨 ref_wav，写进 BarScore 的 attr。
    弹唱 / 指弹比和声分轨（harmonic_sim），旋律谱比人声分轨（vocal_sim）。"""
    n = len(bars)
    ref = bar_chroma(ref_wav, _ref_windows(leadsheet, n))
    syn = bar_chroma(synth_wav, _synth_windows(score, n))
    for i, bs in enumerate(bars):
        setattr(bs, attr, cosine(ref[i], syn[i]))


# ---------- 节奏（弹唱） ----------

def onset_slots(wavs: Iterable[Path], windows: Sequence[tuple[float, float]], slots: int = SLOTS) -> np.ndarray:
    import librosa

    env = None
    for w in wavs:
        y, sr = librosa.load(str(w), sr=22050, mono=True)
        e = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        env = e if env is None else env[: min(len(env), len(e))] + e[: min(len(env), len(e))]
    times = librosa.frames_to_time(np.arange(len(env)), sr=22050, hop_length=512)
    out = np.zeros((len(windows), slots))
    for i, (a, b) in enumerate(windows):
        step = (b - a) / slots
        for k in range(slots):
            c = a + k * step
            m = (times >= c - step / 2) & (times < c + step / 2)
            out[i, k] = env[m].max() if m.any() else 0.0
    return out


def strum_vector(attacks: Sequence[Attack], bar: Bar, slots: int = SLOTS) -> np.ndarray:
    v = np.zeros(slots)
    for a in attacks:
        if a.bar != bar.index:
            continue
        k = int((a.start - bar.start) / bar.length * slots + 0.5) % slots
        v[k] = max(v[k], 0.6 if a.direction == "up" else 1.0)
    return v


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def score_rhythm(score: Score, leadsheet: dict, bars: list[BarScore], onset_wavs: Sequence[Path]) -> None:
    n = len(bars)
    orig = onset_slots(onset_wavs, _ref_windows(leadsheet, n))
    attacks = score.attacks()
    for i, bs in enumerate(bars):
        bs.rhythm_corr = pearson(strum_vector(attacks, score.bars[i]), orig[i])


# ---------- 汇总 ----------

def score_bars(score: Score, leadsheet: dict | None, arr: str, synth_wav: Path | None = None,
               harmonic_wav: Path | None = None, onset_wavs: Sequence[Path] = (), vocal_wav: Path | None = None,
               low_conf: float = Thresholds.melody_conf_flag, finger_style: str = "full") -> list[BarScore]:
    """vocal_wav / low_conf 只用于旋律谱（arr = melody）：人声分轨相似度、要人耳确认的低置信参考音。
    卡祖笛谱（kazoo）只评旋律（与旋律谱同一套：召回考全部参考音、起音容差 LINE_TOL），不对照音频。
    finger_style 只用于指弹（config.finger_style）：full 的旋律按全部参考音评（与旋律谱同一套：召回考全部音、
    起音容差 LINE_TOL、参考没有旋律的小节里谱上的音也算精确率）；skeleton 只考骨架音。
    旋律谱是单旋律线：不评和弦、也不核对谱上的和弦名（只是标出该按的指型，一条旋律线听不出和弦）。"""
    if leadsheet is None:
        bars = [BarScore(bar=b.index + 1, section=b.section) for b in score.bars]
        if arr != "melody":
            chord_mark_issues(score, bars)
        return bars
    n = min(len(score.bars), len(leadsheet["bars"]))
    bars = [BarScore(bar=i + 1, section=score.bars[i].section or leadsheet["bars"][i].get("section")) for i in range(n)]
    if arr == "kazoo":
        score_melody(score, leadsheet, bars, skeleton=False)
        return bars
    if arr == "melody":
        score_melody(score, leadsheet, bars, skeleton=False)
        ear_check(leadsheet, bars, low_conf)
        if synth_wav is not None and vocal_wav is not None and Path(vocal_wav).exists():
            score_harmonic(score, leadsheet, bars, Path(synth_wav), Path(vocal_wav), attr="vocal_sim")
        return bars
    chord_mark_issues(score, bars)
    score_chords(score, leadsheet, bars)
    if arr == "finger":
        score_melody(score, leadsheet, bars, skeleton=finger_style == "skeleton")
    if synth_wav is not None and harmonic_wav is not None and Path(harmonic_wav).exists():
        score_harmonic(score, leadsheet, bars, Path(synth_wav), Path(harmonic_wav))
    if arr == "strum" and onset_wavs and all(Path(w).exists() for w in onset_wavs):
        score_rhythm(score, leadsheet, bars, [Path(w) for w in onset_wavs])
    return bars


def summarize(bars: list[BarScore], violations: list, th: Thresholds, arr: str,
              arrangement_bars: int | None = None, reference_bars: int | None = None,
              has_reference: bool = True) -> dict:
    ct = sum(b.chord_total for b in bars)
    ca = sum(b.chord_agree for b in bars)
    hit, est = sum(b.mel_hit for b in bars), sum(b.mel_est for b in bars)
    found, ref = sum(b.mel_found for b in bars), sum(b.mel_ref for b in bars)
    chord = ca / ct if ct else None
    mel = f1(hit, est, found, ref)
    hs = [b.harmonic_sim for b in bars if b.harmonic_sim is not None]
    rc = [b.rhythm_corr for b in bars if b.rhythm_corr is not None]
    vs = [b.vocal_sim for b in bars if b.vocal_sim is not None]
    marks = sum(len(b.mark_issues) for b in bars)
    target = th.chord_strum if arr == "strum" else th.chord_finger
    line = arr in LINE_ARRS
    mel_target = th.melody_line if line else th.melody_f1
    # 卡祖笛谱没有指法：violations 是记谱问题（playability.check_notation），不是可弹性
    checks = {"notation" if arr == "kazoo" else "playability": not violations}
    if not line:   # 旋律谱 / 卡祖笛谱是单旋律线：不评和弦
        checks["chord"] = chord is None or chord >= target
    # 没有任何东西被核对（无参考，或参考和弦全是低置信 / 没有旋律）时不能算通过
    if not has_reference:
        checks["reference"] = False
    elif arr == "finger":
        checks["reference"] = ct > 0 and ref > 0
    elif line:
        checks["reference"] = ref > 0
    else:
        checks["reference"] = ct > 0
    if arrangement_bars is not None and reference_bars is not None:
        checks["bar_count"] = arrangement_bars == reference_bars
    if arr == "strum":
        checks["chord_marks"] = marks == 0
    if arr == "finger" or line:
        checks["melody"] = mel is None or mel >= mel_target
    return {
        "violations": len(violations),
        "chord_agreement": chord,
        "melody_precision": hit / est if est else None,
        "melody_recall": found / ref if ref else None,
        "melody_f1": mel,
        "melody_octave_exact": sum(b.mel_octave_exact for b in bars) / hit if hit else None,
        "harmonic_similarity": float(np.mean(hs)) if hs else None,
        "rhythm_correlation": float(np.mean(rc)) if rc else None,
        "vocal_similarity": float(np.mean(vs)) if vs else None,
        "ear_check_bars": sum(1 for b in bars if b.ear_notes),
        "ear_check_notes": sum(len(b.ear_notes) for b in bars),
        "low_conf_beats": sum(len(b.low_conf_beats) for b in bars),
        "scored_beats": ct,
        "melody_ref_notes": ref,
        "arrangement_bars": arrangement_bars,
        "reference_bars": reference_bars,
        "chord_mark_issues": marks,
        "chord_target": None if line else target,
        "melody_target": mel_target if arr == "finger" or line else None,
        "checks": checks,
        "pass": all(checks.values()),
    }


# ---------- 卡祖笛：音域、换气 ----------

def breath_runs(score: Score, limit: float) -> list[tuple[float, float]]:
    """连续发声（中间没有休止）超过 limit 秒的乐句 → [(起点, 终点)]（演奏时间，拍）。
    卡祖笛靠哼唱发声，只能在休止处换气；连奏、连音线、紧挨着的下一个音都算连续。"""
    runs: list[list[float]] = []
    for s in sorted(score.sounding(), key=lambda x: x.start):
        if runs and s.start <= runs[-1][1] + 1e-6:
            runs[-1][1] = max(runs[-1][1], s.end)
        else:
            runs.append([s.start, s.end])
    return [(a, b) for a, b in runs if score.seconds(b - a) > limit]


def kazoo_stats(score: Score, th: Thresholds) -> dict:
    """卡祖笛谱的音域（最低 / 最高音、跨度半音数；降号调写降号）与换气提醒（连续超过 th.breath_seconds 秒没有休止
    的乐句：起止小节（1 起）、秒数、起点在小节里的拍位置）。只报告，不影响达标。"""
    pitches = [s.midi for s in score.sounding()]
    flat = bool(score.bars) and score.bars[0].key < 0
    low, high = (min(pitches), max(pitches)) if pitches else (None, None)
    breaths = []
    for a, b in breath_runs(score, th.breath_seconds):
        first, last = _bar_of(score, a), _bar_of(score, b - 1e-3)   # 终点所在小节：结束在小节线上算前一小节
        breaths.append({"bars": [first + 1, last + 1], "seconds": round(score.seconds(b - a), 1),
                        "start": beat_label(a - score.bars[first].start)})
    names = (theory.midi_name(low, flat), theory.midi_name(high, flat)) if pitches else (None, None)
    return {"range_low": names[0], "range_high": names[1], "range_span": None if low is None else high - low,
            "range": f"{names[0]}–{names[1]}（{high - low} 个半音）" if pitches else None,
            "breath_seconds": th.breath_seconds, "breath_warnings": len(breaths), "breaths": breaths}


def problem_bars(bars: list[BarScore], violations: list, th: Thresholds, arr: str) -> list[int]:
    """旋律谱 / 卡祖笛谱：每个音都要对上（小节旋律 F1 < melody_line 就标出；参考没有旋律、谱上却有音的小节也标出），
    以及与人声分轨相似度低的小节（只有旋律谱对照人声）。"""
    bad = {v.bar for v in violations}
    for b in bars:
        if arr in LINE_ARRS:
            if (b.mel_f1 is not None and b.mel_f1 < th.melody_line) or (b.mel_ref == 0 and b.mel_est > 0) or \
                    (b.vocal_sim is not None and b.vocal_sim < th.bar_vocal_flag):
                bad.add(b.bar)
            continue
        if arr == "strum" and b.mark_issues:
            bad.add(b.bar)
        if b.chord_ratio is not None and b.chord_ratio < th.bar_chord_flag:
            bad.add(b.bar)
        if arr == "finger" and b.mel_f1 is not None and b.mel_f1 < th.bar_melody_flag:
            bad.add(b.bar)
        # 参考小节没有骨架音时 mel_f1 为 None，但编配旋律音仍要有足够比例命中参考
        if arr == "finger" and b.mel_ref == 0 and b.mel_est > 0 and b.mel_hit / b.mel_est < th.bar_melody_flag:
            bad.add(b.bar)
        if b.harmonic_sim is not None and b.harmonic_sim < th.bar_harmonic_flag:
            bad.add(b.bar)
    return sorted(bad)
