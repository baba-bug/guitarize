"""一次校验：构建 → 合成 → 可弹性 → 评分 → reports/<arr>/iter-NN（报告、片段、色度图）+ history.csv。
弹唱 / 指弹：前 MAX_SNIPPETS 个问题小节出试听片段（原曲 ↔ 合成）和色度图。
旋律谱（melody）对照人声分轨：试听片段是人声 ↔ 合成，图是人声基频 vs 谱上的音（音高图）；列出的小节（问题小节、
需人耳确认的小节）全部出试听片段和音高图，不受 MAX_SNIPPETS 截断——音高图是找错音的主要工具，需人耳确认的
小节也都要能听、能看（片段与图在 .gitignore 里，只占本地空间）。
卡祖笛谱（kazoo）没有指法：不查可弹性，只查记谱（小节时值、Shuffle 反拍）；按全部参考音评旋律（与旋律谱同一套），
另报音域和换气提醒（metrics.kazoo_stats）；不对照音频，没有试听片段和图（参考音频在交互谱里选来听）。"""
from __future__ import annotations

import csv
import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import theory
from ..config import ARR_NAMES, SongConfig, finger_style, load_profile, melody_source
from ..paths import SongPaths
from ..render import alphatab
from ..render.synth import midi_to_wav
from . import metrics
from .positions import beat_label
from .model import Score, load_score
from .playability import Violation, check_notation, check_playability, soft_stats

HISTORY_FIELDS = ["iter", "time", "violations", "chord_agreement", "melody_f1",
                  "harmonic_similarity", "rhythm_correlation", "pass",
                  "chord_mark_issues", "arrangement_bars", "reference_bars", "scored_beats", "vocal_similarity",
                  "range_span", "breath_warnings"]
MAX_SNIPPETS = 20
CHECK_NAMES = {"playability": "可弹性", "chord": "和弦", "melody": "旋律", "chord_marks": "和弦标记",
               "bar_count": "小节数", "reference": "参考不足", "notation": "记谱"}


def next_iter_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name[5:]) for p in root.glob("iter-*") if p.name[5:].isdigit()]
    out = root / f"iter-{(max(nums) + 1 if nums else 1):02d}"
    out.mkdir()
    return out


def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def melody_reference(paths: SongPaths, overrides: dict) -> Path:
    """旋律谱对照的人声：识别旋律用的那条分轨。melody.json 记着它（melody 步骤写的 source），以它为准；
    没有记（旧的 melody.json、还没跑 melody 步骤）时按 overrides.melody_source（默认 lead）。
    lead 是去掉伴唱的主唱 stems/lead.wav（analyze 的 lead 步骤）；vocals、或 lead.wav 不在时用整条人声 stems/vocals.wav。"""
    recorded = (_load_json(paths.melody_json) or {}).get("source")
    source = (melody_source(recorded, "melody.json source") if recorded is not None
              else melody_source(overrides.get("melody_source")))
    lead = paths.stem("lead")
    return lead if source == "lead" and lead.exists() else paths.stem("vocals")


def run_verify(paths: SongPaths, cfg: SongConfig, arr: str) -> Path:
    profile = load_profile(cfg.level)
    built = alphatab.build(paths.atex(arr), paths.build_dir, arr)
    score = load_score(built["json"])
    kazoo = arr == "kazoo"   # 卡祖笛：没有指法、不对照音频（见模块说明），不用合成
    synth = None if kazoo else midi_to_wav(built["mid"], paths.build(arr, "wav"))
    violations = check_notation(score) if kazoo else check_playability(score, profile, arr)
    leadsheet = _load_json(paths.leadsheet_json)
    vocals = melody_reference(paths, cfg.overrides) if arr == "melody" else None   # 只有旋律谱对照人声
    style = finger_style(cfg.overrides)   # 指弹旋律的评法跟着写法走
    bars = metrics.score_bars(score, leadsheet, arr, synth_wav=synth, harmonic_wav=paths.stem("harmonic"),
                              onset_wavs=(paths.stem("drums"), paths.stem("guitar")), vocal_wav=vocals,
                              low_conf=cfg.thresholds.melody_conf_flag, finger_style=style)
    summary = metrics.summarize(
        bars, violations, cfg.thresholds, arr,
        arrangement_bars=len(score.bars),
        reference_bars=len(leadsheet["bars"]) if leadsheet else None,
        has_reference=leadsheet is not None,
    )
    problems = metrics.problem_bars(bars, violations, cfg.thresholds, arr)
    if arr == "melody":
        summary["vocal_reference"] = vocals.name   # 相似度、试听片段、音高图对照的分轨
    if arr == "finger":
        summary["finger_style"] = style
    if kazoo:
        summary.update(metrics.kazoo_stats(score, cfg.thresholds))

    root = paths.report_root(arr)
    out = next_iter_dir(root)
    if kazoo:
        snippets, plots = {}, {}
    elif arr == "melody":   # 问题小节在前，再是需人耳确认的小节；全部出片段和音高图（见模块说明）
        listen = problems + [b.bar for b in bars if b.ear_notes and b.bar not in problems]
        snippets = _write_snippets(out, listen, score, leadsheet, synth, vocals, "vocal")
        contour = _vocal_contour(paths, leadsheet, vocals) if leadsheet and listen else None
        plots = _write_pitch_plots(out, listen, score, leadsheet, contour)
    else:                 # 片段和色度图同一个上限
        shown = problems[:MAX_SNIPPETS]
        snippets = _write_snippets(out, shown, score, leadsheet, synth, paths.original_wav)
        plots = _write_plots(out, shown, score, leadsheet, synth, paths.stem("harmonic"))
    (out / "scores.json").write_text(json.dumps({
        "summary": summary,
        "problems": problems,
        "bars": [b.to_dict() for b in bars],
        "violations": [asdict(v) for v in violations],
        "soft": {} if kazoo else soft_stats(score),   # 软统计是按弦的手形
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    history = root / "history.csv"
    prev = _last_row(history)
    (out / "report.md").write_text(
        render_report(arr, out.name, summary, prev, violations, bars, problems, snippets, plots, score,
                      has_reference=leadsheet is not None),
        encoding="utf-8",
    )
    _append_history(history, out.name, summary)
    return out


# ---------- 片段与图 ----------

def _cut(wav: Path, a: float, b: float, dst: Path) -> None:
    info = sf.info(str(wav))
    start = max(0, int(a * info.samplerate))
    stop = min(info.frames, int(b * info.samplerate))
    data, sr = sf.read(str(wav), start=start, stop=stop)
    sf.write(str(dst), data, sr)


def _write_snippets(out: Path, problems: list[int], score: Score, leadsheet: dict | None,
                    synth: Path, original: Path, ref_name: str = "orig") -> dict[int, list[str]]:
    """problems 里每个小节的 A/B 试听（上限由调用方截取）：bar-NNN_<ref_name>.wav（参考：原曲，旋律谱是人声分轨）
    ↔ bar-NNN_synth.wav。"""
    res: dict[int, list[str]] = {}
    folder = out / "snippets"
    for bar in problems:
        i = bar - 1
        files = []
        folder.mkdir(exist_ok=True)
        if leadsheet and original.exists() and i < len(leadsheet["bars"]):
            lb = leadsheet["bars"][i]
            p = folder / f"bar-{bar:03d}_{ref_name}.wav"
            _cut(original, max(0.0, lb["t_start"]), lb["t_end"], p)
            files.append(p.relative_to(out).as_posix())
        if i < len(score.bars):
            sb = score.bars[i]
            p = folder / f"bar-{bar:03d}_synth.wav"
            _cut(synth, score.seconds(sb.start), score.seconds(sb.start + sb.length), p)
            files.append(p.relative_to(out).as_posix())
        res[bar] = files
    return res


def _write_plots(out: Path, problems: list[int], score: Score, leadsheet: dict | None,
                 synth: Path, harmonic: Path) -> dict[int, str]:
    if not leadsheet or not harmonic.exists() or not problems:
        return {}
    import librosa
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def load(wav: Path):
        y, sr = librosa.load(str(wav), sr=22050, mono=True)
        c = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
        return c, librosa.frames_to_time(np.arange(c.shape[1]), sr=sr, hop_length=512)

    ref_c, ref_t = load(harmonic)
    syn_c, syn_t = load(synth)
    folder = out / "plots"
    folder.mkdir(exist_ok=True)
    res: dict[int, str] = {}
    n = min(len(score.bars), len(leadsheet["bars"]))
    for bar in problems:
        i = bar - 1
        if i >= n:
            continue
        lb, sb = leadsheet["bars"][i], score.bars[i]
        panels = [
            (ref_c, ref_t, lb["t_start"], lb["t_end"], "original (harmonic stem)"),
            (syn_c, syn_t, score.seconds(sb.start), score.seconds(sb.start + sb.length), "arrangement (synth)"),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
        for ax, (c, t, a, b, name) in zip(axes, panels):
            m = (t >= a) & (t < b)
            ax.imshow(c[:, m], aspect="auto", origin="lower", cmap="magma")
            ax.set_title(name)
            ax.set_yticks(range(12))
            ax.set_yticklabels(theory.PITCH_CLASSES)
        fig.suptitle(f"bar {bar} chroma")
        p = folder / f"bar-{bar:03d}.png"
        fig.savefig(p, dpi=90, bbox_inches="tight")
        plt.close(fig)
        res[bar] = p.relative_to(out).as_posix()
    return res


def _vocal_contour(paths: SongPaths, leadsheet: dict, vocals: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """vocals（melody_reference）的基频（与 melody 步骤同一条 Viterbi 曲线，缓存在 analysis/vocal_f0.npz）
    → (拍位置, MIDI 浮点音高或 nan)。拍轴与 melody.json 相同（beats.json 的 time_to_beat）；没有 beats.json 时
    按 leadsheet 的小节起止时间逐小节线性换算。没有这条分轨时 None。"""
    if not vocals.exists():
        return None
    from ..analyze.melody import cached_vocal_pitch

    times, midi, _ = cached_vocal_pitch(vocals, paths.vocal_f0)
    if paths.beats_json.exists():
        from ..analyze.beats import time_to_beat

        beats = np.asarray(time_to_beat(json.loads(paths.beats_json.read_text(encoding="utf-8")), times), dtype=float)
    else:
        bpb = leadsheet.get("beats_per_bar", 4)
        edges = [b["t_start"] for b in leadsheet["bars"]] + [leadsheet["bars"][-1]["t_end"]]
        beats = np.interp(times, edges, np.arange(len(edges)) * bpb, left=np.nan, right=np.nan)
    return beats, np.asarray(midi, dtype=float)


def _write_pitch_plots(out: Path, bars_to_plot: list[int], score: Score, leadsheet: dict | None,
                       contour: tuple[np.ndarray, np.ndarray] | None) -> dict[int, str]:
    """旋律谱的逐小节音高图（找错音的主要工具）：横轴是拍（小节内，1 起），纵轴是 MIDI 音高。
    蓝线：人声基频；橙色横条：谱上的音（实际音高 = 指法调 + 变调夹）；虚线框：参考旋律音（leadsheet / melody.json）。
    竖线：拍线（实线）与细分（Shuffle 三连音、直拍十六分，点线）。"""
    if not leadsheet or contour is None or not bars_to_plot:
        return {}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    f0_beats, f0 = contour
    tab = [s for s in score.sounding() if s.voice == 0]
    refs = [m for b in leadsheet["bars"] for m in b.get("melody", [])]
    sub = 3 if leadsheet.get("feel") == "shuffle" else 4
    folder = out / "plots"
    folder.mkdir(exist_ok=True)
    res: dict[int, str] = {}
    n = min(len(score.bars), len(leadsheet["bars"]))
    for bar in bars_to_plot:
        i = bar - 1
        if i >= n:
            continue
        a, length = score.bars[i].start, score.bars[i].length
        lo, hi = a - 0.25, a + length + 0.25
        win = (f0_beats >= lo) & (f0_beats < hi)
        mine = [s for s in tab if s.start < hi and s.end > lo]
        ref = [m for m in refs if m["start_beats"] < hi and m["start_beats"] + m["dur_beats"] > lo]
        voiced = f0[win][~np.isnan(f0[win])]
        pitches = [s.midi for s in mine] + [m["midi"] for m in ref] + list(voiced) or [60]
        y0, y1 = int(np.floor(min(pitches))) - 2, int(np.ceil(max(pitches))) + 2
        fig, ax = plt.subplots(figsize=(8, 3.6))
        for k in range(int(round(length * sub)) + 1):
            on_beat = k % sub == 0
            ax.axvline(1 + k / sub, color="0.7" if on_beat else "0.88", lw=1.0 if on_beat else 0.6,
                       ls="-" if on_beat else ":", zorder=0)
        for m in ref:
            ax.add_patch(Rectangle((m["start_beats"] - a + 1, m["midi"] - 0.45), m["dur_beats"], 0.9,
                                   fill=False, ls="--", ec="0.3", lw=1.0, zorder=1))
        for s in mine:
            ax.add_patch(Rectangle((s.start - a + 1, s.midi - 0.3), s.end - s.start, 0.6,
                                   color="tab:orange", alpha=0.75, zorder=2))
        ax.plot(f0_beats[win] - a + 1, f0[win], color="tab:blue", lw=1.3, zorder=3)
        ax.set_xlim(lo - a + 1, hi - a + 1)
        ax.set_ylim(y0, y1)
        ax.set_xticks(range(1, int(round(length)) + 2))
        ax.set_xlabel("beat")
        step = 1 if y1 - y0 <= 24 else 2
        ticks = list(range(y0, y1 + 1, step))
        ax.set_yticks(ticks)
        ax.set_yticklabels([theory.midi_name(p) for p in ticks], fontsize=7)
        ax.grid(axis="y", color="0.93", lw=0.5)
        ax.legend(handles=[Line2D([], [], color="tab:blue", lw=1.3, label="vocal f0 (crepe)"),
                           Patch(color="tab:orange", alpha=0.75, label=f"tab (sounding, capo {score.capo})"),
                           Patch(fill=False, ls="--", ec="0.3", label="reference (melody.json)")],
                  loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)
        ax.set_title(f"bar {bar} · vocal pitch vs tab", fontsize=10)
        p = folder / f"bar-{bar:03d}.png"
        fig.savefig(p, dpi=90, bbox_inches="tight")
        plt.close(fig)
        res[bar] = p.relative_to(out).as_posix()
    return res


# ---------- history ----------

def _last_row(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return rows[-1] if rows else None


def _append_history(path: Path, iter_name: str, s: dict) -> None:
    new = not path.exists()
    if not new:  # 旧版表头列数较少：先把已有行迁移到新表头，避免错位
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fields, old_rows = reader.fieldnames, list(reader)
        if old_fields != HISTORY_FIELDS:
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS, restval="")
                w.writeheader()
                w.writerows({k: r.get(k, "") for k in HISTORY_FIELDS} for r in old_rows)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if new:
            w.writeheader()
        row = {"iter": iter_name, "time": dt.datetime.now().isoformat(timespec="seconds")}
        row.update({k: ("" if s.get(k) is None else s[k]) for k in HISTORY_FIELDS[2:]})
        w.writerow(row)


def _num(v):
    if v in (None, ""):
        return None
    if v in ("True", "False"):
        return v == "True"
    try:
        return int(v)
    except ValueError:
        return float(v)


# ---------- 报告 ----------

def _fmt(v, pct: bool = False) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✅" if v else "❌"
    if isinstance(v, float):
        return f"{v:.1%}" if pct else f"{v:.3f}"
    return str(v)


def render_report(arr: str, iter_name: str, s: dict, prev: dict | None, violations: list[Violation],
                  bars: list[metrics.BarScore], problems: list[int], snippets: dict[int, list[str]],
                  plots: dict[int, str], score: Score, has_reference: bool = True) -> str:
    failed = [CHECK_NAMES.get(k, k) for k, ok in s["checks"].items() if not ok]
    status = "✅ 达标" if s["pass"] else "❌ 未达标：" + "、".join(failed)
    lines = [f"# {ARR_NAMES[arr]}校验报告 · {iter_name}", "", f"**结论：{status}**", ""]
    if not has_reference:
        lines += ["> 没有参考（leadsheet.json），只做了可弹性检查，未做差异校验。", ""]
    if s.get("vocal_reference"):
        lines += [f"> 对照的人声分轨：analysis/stems/{s['vocal_reference']}（相似度、试听片段、音高图）", ""]
    kazoo = arr == "kazoo"
    if kazoo:
        lines += ["> 对照 leadsheet.json 里确认过的旋律（与旋律谱同一份，按实际音高）；卡祖笛没有指法，只查记谱，"
                  "不对照音频。", ""]
    lines += ["| 指标 | 本轮 | 上一轮 | 达标线 |", "|---|---|---|---|"]
    rows = [("记谱问题（小节时值 / Shuffle 反拍）" if kazoo else "可弹性违规", "violations", False, "0")]
    if arr in ("melody", "kazoo"):   # 单旋律线：不评和弦；召回按全部参考音
        rows += [("旋律 F1", "melody_f1", False, f"≥ {s['melody_target']:.2f}"),
                 ("旋律精确率", "melody_precision", True, ""),
                 ("旋律召回率（全部参考音）", "melody_recall", True, ""),
                 ("八度完全正确", "melody_octave_exact", True, "")]
    if arr == "melody":   # 相似度对照人声分轨
        rows += [("人声相似度（均值）", "vocal_similarity", False, "仅用于定位"),
                 ("需人耳确认（调外 / 低置信音所在小节）", "ear_check_bars", False, "转人耳确认")]
    if kazoo:
        rows += [("音域", "range", False, "仅报告"),
                 (f"换气提醒（连续超过 {s['breath_seconds']:g} 秒没有休止）", "breath_warnings", False, "仅报告")]
    if arr in ("strum", "finger"):
        rows.append(("和弦一致率", "chord_agreement", True, f"≥ {s['chord_target']:.0%}"))
    skeleton = s.get("finger_style") == "skeleton"
    if arr == "finger":
        rows += [("旋律 F1", "melody_f1", False, f"≥ {s['melody_target']:.2f}"),
                 ("旋律精确率", "melody_precision", True, ""),
                 (f"旋律召回率（{'骨架音' if skeleton else '全部参考音'}）", "melody_recall", True, ""),
                 ("八度完全正确", "melody_octave_exact", True, "")]
    if arr in ("strum", "finger"):
        rows.append(("和弦标记不符", "chord_mark_issues", False, "0" if arr == "strum" else "仅报告"))
        rows.append(("和声相似度（均值）", "harmonic_similarity", False, "仅用于定位"))
    if arr == "strum":
        rows.append(("节奏相关度（均值）", "rhythm_correlation", False, "仅报告"))
    if arr in ("strum", "finger"):
        rows.append(("低置信参考拍", "low_conf_beats", False, "转人耳确认"))
    for name, key, pct, target in rows:
        pv = _num(prev.get(key)) if prev and key in prev else None
        lines.append(f"| {name} | {_fmt(s[key], pct)} | {_fmt(pv, pct)} | {target} |")
        if key == "violations":
            lines.append(f"| 小节数（编配 / 参考） | {s['arrangement_bars']} / {s['reference_bars'] or '—'} | — | 相等 |")

    lines += ["", f"## {'记谱问题' if kazoo else '可弹性违规'}（{len(violations)}）"]
    lines += [f"- 第 {v.bar} 小节 · 拍 {beat_label(v.start - score.bars[v.bar - 1].start)} · `{v.rule}`：{v.detail}"
              for v in violations] or ["- 无"]

    by_bar = {b.bar: b for b in bars}
    lines += ["", f"## 问题小节（{len(problems)}）"]
    if not problems:
        lines.append("- 无")
    for bar in problems:
        b = by_bar.get(bar)
        sec = f" [{b.section}]" if b and b.section else ""
        lines += ["", f"### 第 {bar} 小节{sec}"]
        if b:
            if b.chord_ratio is not None:
                lines.append(f"- 和弦一致 {b.chord_agree}/{b.chord_total}")
                lines += [f"  - {x}" for x in b.chord_issues]
            if b.mel_f1 is not None:
                cover = "骨架覆盖" if arr == "finger" and skeleton else "覆盖"
                lines.append(f"- 旋律 F1 {b.mel_f1:.2f}（命中 {b.mel_hit}/{b.mel_est}，{cover} {b.mel_found}/{b.mel_ref}）")
                lines += [f"  - {x}" for x in b.mel_issues]
            elif arr in ("melody", "kazoo") and b.mel_est:   # 参考没有旋律：F1 无定义，谱上的音都列出来
                lines.append(f"- 参考这一小节没有旋律，谱上却有 {b.mel_est} 个音")
                lines += [f"  - {x}" for x in b.mel_issues]
            lines += [f"- 和弦标记不符：{x}" for x in b.mark_issues]
            if b.harmonic_sim is not None:
                lines.append(f"- 和声相似度 {b.harmonic_sim:.2f}")
            if b.vocal_sim is not None:
                lines.append(f"- 人声相似度 {b.vocal_sim:.2f}")
            if b.ear_notes:
                lines.append(f"- 需人耳确认：{'；'.join(b.ear_notes)}")
        lines += [f"- 违规 `{v.rule}`：{v.detail}" for v in violations if v.bar == bar]
        if snippets.get(bar):
            lines.append("- 试听：" + " ↔ ".join(snippets[bar]))
        if plots.get(bar):
            lines.append(f"- {'音高图' if arr == 'melody' else '色度图'}：{plots[bar]}")

    if kazoo:
        lines += ["", f"## 换气提醒（连续超过 {s['breath_seconds']:g} 秒没有休止，{len(s['breaths'])} 处）"]
        lines += [f"- 第 {a}–{z} 小节：{x['seconds']:g} 秒没有休止（第 {a} 小节拍 {x['start']} 起）"
                  for x in s["breaths"] for a, z in [x["bars"]]] or ["- 无"]
        return "\n".join(lines) + "\n"
    if arr == "melody":
        ear = [b for b in bars if b.ear_notes]
        lines += ["", f"## 需人耳确认（参考旋律里调外 / 低置信的音，共 {s['ear_check_notes']} 个，{len(ear)} 小节）"]
        for b in ear:
            extra = "".join(f" · {k}：{v}" for k, v in (("试听", " ↔ ".join(snippets.get(b.bar, []))),
                                                       ("音高图", plots.get(b.bar))) if v and b.bar not in problems)
            lines.append(f"- 第 {b.bar} 小节：{'；'.join(b.ear_notes)}{extra}")
        if not ear:
            lines.append("- 无")
        return "\n".join(lines) + "\n"

    low = [b for b in bars if b.low_conf_beats]
    lines += ["", f"## 需人耳确认（参考和弦低置信，共 {s['low_conf_beats']} 拍）"]
    lines += [f"- 第 {b.bar} 小节：拍 {', '.join(map(str, b.low_conf_beats))}" for b in low] or ["- 无"]
    return "\n".join(lines) + "\n"
