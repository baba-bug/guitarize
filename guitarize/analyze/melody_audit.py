"""每版旋律都要和原唱比（audit-melody）：拿 melody.json（或别的版本）对照这首歌主唱的逐帧特征——模型主唱原样 lead_raw.wav、
伴唱 backing.wav、crepe，与旋律 v7 同一套（notes.frame_features）：
- coverage 覆盖：主唱确实在唱的帧里有音符的比例；
- pitch_ok 音准：音符时值中间 60% 的主唱平均音高（颤音取平均 = 听感音高）与音符差 ≤ 0.5 半音的比例，wrong 是差得多的个数；
- backing_or_empty 伴唱音：时值里一半以上是伴唱漏音 / 没人唱的音；
- chopped_held 切碎的长音：紧接前一个同音、中间没有换字、却又拨一下；
- onset_recall 节奏：每个换字点 ±1 格内有拨弦音的比例；pluck_ms：拨弦音离最近换字点的中位距离（毫秒，2 格内的才算）；
- repeat_f1 重复段一致：song.yaml 里同名的段落（如两遍サビ）逐音对上的 F1（±1 格、同音高，按短的那段比）。"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from ..config import SongConfig
from ..paths import SongPaths
from .beats import beat_to_time
from .notes import FPS, Frames, frame_features, middle

PITCH_OK = 0.5      # 半音
SLOT_TOL = 1        # 重复段逐音对上：起点差不超过这么多格
COLUMNS = ("notes", "legato", "coverage", "pitch_ok", "wrong", "backing_or_empty", "chopped_held", "onset_recall",
           "pluck_ms", "repeat_f1")


def _div(beats: dict) -> int:
    """每拍几格：shuffle 三连音 3，直拍十六分 4。"""
    return 3 if beats.get("feel") == "shuffle" else 4


def audit_notes(notes: list[dict], fr: Frames, beats: dict) -> dict:
    """一版旋律对照主唱逐帧特征的指标（见模块说明；不含重复段一致）。"""
    n = len(fr.voiced)
    sung = fr.voiced

    def span(x: dict) -> tuple[int, int]:
        a, e = beat_to_time(beats, np.array([x["start_beats"], x["start_beats"] + x["dur_beats"]]))
        return int(round(a * FPS)), int(round(e * FPS))

    covered = np.zeros(n, bool)
    ok = wrong = leaky = chopped = 0
    for k, x in enumerate(notes):
        i, j = span(x)
        i, j = max(0, i), min(n, j)
        if j <= i:
            continue
        covered[i:j] = True
        if sung[i:j].mean() < 0.5:
            leaky += 1
            continue
        mi, mj = middle(i, j)
        seg = sung[mi:mj]
        ref = np.mean(fr.midi[mi:mj][seg]) if seg.sum() >= 2 else np.mean(fr.midi[i:j][sung[i:j]])
        if abs(x["midi"] - ref) <= PITCH_OK:
            ok += 1
        else:
            wrong += 1
        if k and notes[k - 1]["midi"] == x["midi"] and not x.get("legato"):
            if abs(span(notes[k - 1])[1] - i) <= 3 and not fr.onset[max(0, i - 5):i + 6].any() \
                    and fr.voiced[max(0, i - 3):i + 3].all():
                chopped += 1
    slot = int(round(FPS * 60 / beats["tempo_bpm"] / _div(beats)))
    plucks = np.array([span(x)[0] for x in notes if not x.get("legato")])
    onsets = np.flatnonzero(fr.onset & sung)
    recall = float(np.mean([np.abs(plucks - o).min() <= slot for o in onsets])) if len(plucks) and len(onsets) else 0.0
    dist = [np.abs(onsets - p).min() for p in plucks] if len(onsets) else []
    err = [d * 1000 / FPS for d in dist if d <= 2 * slot]
    return {"notes": len(notes), "legato": sum(bool(x.get("legato")) for x in notes),
            "coverage": round(float(covered[sung].mean()), 3) if sung.any() else 0.0,
            "pitch_ok": round(ok / max(1, ok + wrong), 3), "wrong": wrong, "backing_or_empty": leaky,
            "chopped_held": chopped, "onset_recall": round(recall, 3),
            "pluck_ms": int(np.median(err)) if err else None}


def repeat_pairs(sections: dict[int, str], last_bar: int) -> list[tuple[str, tuple[int, int], tuple[int, int]]]:
    """song.yaml 的 sections 里同名段落两两配对 [(段名, (起, 止小节), (起, 止小节))]：每段到下一段之前为止
    （最后一段到 last_bar），两段都按短的那段截齐。"""
    starts = sorted(sections)
    spans: dict[str, list[tuple[int, int]]] = {}
    for k, bar in enumerate(starts):
        end = starts[k + 1] - 1 if k + 1 < len(starts) else last_bar
        if end >= bar:
            spans.setdefault(sections[bar], []).append((bar, end))
    pairs = []
    for name, found in spans.items():
        for a, b in itertools.combinations(found, 2):
            length = min(a[1] - a[0], b[1] - b[0])
            pairs.append((name, (a[0], a[0] + length), (b[0], b[0] + length)))
    return pairs


def _matches(notes: list[dict], a: tuple[int, int], b: tuple[int, int], bpb: int, div: int) -> tuple[int, int, int]:
    """两段（小节区间）逐音对上：起点（相对段首，以格计）差 ≤ SLOT_TOL、同音高。返回 (对上的个数, 前段音数, 后段音数)。"""
    def pick(span: tuple[int, int]) -> list[tuple[int, int]]:
        lo, hi = (span[0] - 1) * bpb, span[1] * bpb
        return [(round((x["start_beats"] - lo) * div), x["midi"]) for x in notes if lo <= x["start_beats"] < hi]

    first, second = pick(a), pick(b)
    used, hit = set(), 0
    for s, m in first:
        for k, (s2, m2) in enumerate(second):
            if k not in used and abs(s - s2) <= SLOT_TOL and m == m2:
                used.add(k)
                hit += 1
                break
    return hit, len(first), len(second)


def repeat_consistency(notes: list[dict], sections: dict[int, str], beats: dict) -> dict:
    """同名段落逐音对上的 F1：{"f1": 所有配对合起来（没有重复的段落时 None）, "pairs": 每一对}。"""
    bpb, div = beats["time_signature"][0], _div(beats)
    last_bar = max((b["bar"] for b in beats["bars"]), default=0)
    pairs, hits, total = [], 0, 0
    for name, a, b in repeat_pairs(sections, last_bar):
        hit, na, nb = _matches(notes, a, b, bpb, div)
        hits, total = hits + hit, total + na + nb
        pairs.append({"section": name, "bars": [list(a), list(b)], "f1": round(2 * hit / max(1, na + nb), 3)})
    return {"f1": round(2 * hits / max(1, total), 3) if pairs else None, "pairs": pairs}


def _label(path: Path, root: Path) -> str:
    """报告里的文件名：在歌目录里的写相对路径，其余照传进来的写。"""
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def audit_melody(paths: SongPaths, cfg: SongConfig, melody_files: list[Path] | tuple = ()) -> list[dict]:
    """analysis/melody.json（在的话）和 melody_files 各一行指标，写进 reports/melody_audit.json 并返回。"""
    ref = {"lead_raw": paths.stem("lead_raw"), "backing": paths.stem("backing")}
    missing = [p.name for p in ref.values() if not p.exists()]
    if missing:
        raise ValueError(f"audit-melody compares against the karaoke model's lead and backing; missing "
                         f"{', '.join(missing)} — run `guitarize analyze {paths.slug} --steps lead` with the model")
    files = [*([paths.melody_json] if paths.melody_json.exists() else []), *map(Path, melody_files)]
    if not files:
        raise ValueError(f"no melody to audit; run `guitarize analyze {paths.slug} --steps melody` or pass --melody")
    if not paths.beats_json.exists():
        raise ValueError(f"missing beats.json; run `guitarize analyze {paths.slug} --steps beats` first")
    beats = json.loads(paths.beats_json.read_text(encoding="utf-8"))
    fr = frame_features(ref["lead_raw"], ref["backing"], paths.vocal_f0)
    rows = []
    for f in files:
        melody = json.loads(Path(f).read_text(encoding="utf-8"))
        repeats = repeat_consistency(melody["notes"], cfg.sections, beats)
        rows.append({"melody": _label(f, paths.root), "method": melody.get("method"),
                     **audit_notes(melody["notes"], fr, beats), "repeat_f1": repeats["f1"], "repeats": repeats["pairs"]})
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    report = {"reference": {k: _label(p, paths.root) for k, p in ref.items()}, "rows": rows}
    (paths.reports_dir / "melody_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
    return rows


def format_table(rows: list[dict]) -> str:
    """一版旋律一行：文件名 + COLUMNS（没有的值写 -）。"""
    head = ["melody", *COLUMNS]
    cells = [[r["melody"], *("-" if r[c] is None else str(r[c]) for c in COLUMNS)] for r in rows]
    widths = [max(len(row[k]) for row in [head, *cells]) for k in range(len(head))]
    fmt = lambda row: "  ".join(c.ljust(w) if k == 0 else c.rjust(w) for k, (c, w) in enumerate(zip(row, widths)))
    return "\n".join(fmt(row) for row in [head, *cells])
