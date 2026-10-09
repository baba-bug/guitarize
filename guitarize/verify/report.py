"""一次校验：构建 → 合成 → 可弹性 → 评分 → reports/<arr>/iter-NN（报告、片段、色度图）+ history.csv。"""
from __future__ import annotations

import csv
import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import theory
from ..config import ARR_NAMES, SongConfig, load_profile
from ..paths import SongPaths
from ..render import alphatab
from ..render.synth import midi_to_wav
from . import metrics
from .positions import beat_label
from .model import Score, load_score
from .playability import Violation, check_playability, soft_stats

HISTORY_FIELDS = ["iter", "time", "violations", "chord_agreement", "melody_f1",
                  "harmonic_similarity", "rhythm_correlation", "pass",
                  "chord_mark_issues", "arrangement_bars", "reference_bars", "scored_beats"]
MAX_SNIPPETS = 20
CHECK_NAMES = {"playability": "可弹性", "chord": "和弦", "melody": "旋律", "chord_marks": "和弦标记",
               "bar_count": "小节数", "reference": "参考不足"}


def next_iter_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name[5:]) for p in root.glob("iter-*") if p.name[5:].isdigit()]
    out = root / f"iter-{(max(nums) + 1 if nums else 1):02d}"
    out.mkdir()
    return out


def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def run_verify(paths: SongPaths, cfg: SongConfig, arr: str) -> Path:
    profile = load_profile(cfg.level)
    built = alphatab.build(paths.atex(arr), paths.build_dir, arr)
    score = load_score(built["json"])
    synth = midi_to_wav(built["mid"], paths.build(arr, "wav"))
    violations = check_playability(score, profile, arr)
    leadsheet = _load_json(paths.leadsheet_json)
    bars = metrics.score_bars(score, leadsheet, arr, synth_wav=synth, harmonic_wav=paths.stem("harmonic"),
                              onset_wavs=(paths.stem("drums"), paths.stem("guitar")))
    summary = metrics.summarize(
        bars, violations, cfg.thresholds, arr,
        arrangement_bars=len(score.bars),
        reference_bars=len(leadsheet["bars"]) if leadsheet else None,
        has_reference=leadsheet is not None,
    )
    problems = metrics.problem_bars(bars, violations, cfg.thresholds, arr)

    root = paths.report_root(arr)
    out = next_iter_dir(root)
    snippets = _write_snippets(out, problems, score, leadsheet, synth, paths.original_wav)
    plots = _write_plots(out, problems, score, leadsheet, synth, paths.stem("harmonic"))
    (out / "scores.json").write_text(json.dumps({
        "summary": summary,
        "problems": problems,
        "bars": [b.to_dict() for b in bars],
        "violations": [asdict(v) for v in violations],
        "soft": soft_stats(score),
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
                    synth: Path, original: Path) -> dict[int, list[str]]:
    res: dict[int, list[str]] = {}
    folder = out / "snippets"
    for bar in problems[:MAX_SNIPPETS]:
        i = bar - 1
        files = []
        folder.mkdir(exist_ok=True)
        if leadsheet and original.exists() and i < len(leadsheet["bars"]):
            lb = leadsheet["bars"][i]
            p = folder / f"bar-{bar:03d}_orig.wav"
            _cut(original, lb["t_start"], lb["t_end"], p)
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
    for bar in problems[:MAX_SNIPPETS]:
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
    lines += ["| 指标 | 本轮 | 上一轮 | 达标线 |", "|---|---|---|---|"]
    rows = [("可弹性违规", "violations", False, "0"),
            ("和弦一致率", "chord_agreement", True, f"≥ {s['chord_target']:.0%}")]
    if arr == "finger":
        rows += [("旋律 F1", "melody_f1", False, f"≥ {s['melody_target']:.2f}"),
                 ("旋律精确率", "melody_precision", True, ""),
                 ("旋律召回率（骨架音）", "melody_recall", True, ""),
                 ("八度完全正确", "melody_octave_exact", True, "")]
    rows.append(("和弦标记不符", "chord_mark_issues", False, "0" if arr == "strum" else "仅报告"))
    rows.append(("和声相似度（均值）", "harmonic_similarity", False, "仅用于定位"))
    if arr == "strum":
        rows.append(("节奏相关度（均值）", "rhythm_correlation", False, "仅报告"))
    rows.append(("低置信参考拍", "low_conf_beats", False, "转人耳确认"))
    for name, key, pct, target in rows:
        pv = _num(prev.get(key)) if prev and key in prev else None
        lines.append(f"| {name} | {_fmt(s[key], pct)} | {_fmt(pv, pct)} | {target} |")
        if key == "violations":
            lines.append(f"| 小节数（编配 / 参考） | {s['arrangement_bars']} / {s['reference_bars'] or '—'} | — | 相等 |")

    lines += ["", f"## 可弹性违规（{len(violations)}）"]
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
                lines.append(f"- 旋律 F1 {b.mel_f1:.2f}（命中 {b.mel_hit}/{b.mel_est}，骨架覆盖 {b.mel_found}/{b.mel_ref}）")
                lines += [f"  - {x}" for x in b.mel_issues]
            lines += [f"- 和弦标记不符：{x}" for x in b.mark_issues]
            if b.harmonic_sim is not None:
                lines.append(f"- 和声相似度 {b.harmonic_sim:.2f}")
        lines += [f"- 违规 `{v.rule}`：{v.detail}" for v in violations if v.bar == bar]
        if snippets.get(bar):
            lines.append("- 试听：" + " ↔ ".join(snippets[bar]))
        if plots.get(bar):
            lines.append(f"- 色度图：{plots[bar]}")

    low = [b for b in bars if b.low_conf_beats]
    lines += ["", f"## 需人耳确认（参考和弦低置信，共 {s['low_conf_beats']} 拍）"]
    lines += [f"- 第 {b.bar} 小节：拍 {', '.join(map(str, b.low_conf_beats))}" for b in low] or ["- 无"]
    return "\n".join(lines) + "\n"
