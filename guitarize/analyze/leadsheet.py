"""合并 beats / chords / melody → leadsheet.json（Phase 1 校验器的参考），并输出便于阅读的小节摘要。"""
from __future__ import annotations

import re
import sys

from .. import theory
from .beats import beats_signature

_MARK = {"high": "", "medium": "?", "low": "??", "manual": "!"}  # manual = overrides.chords 人耳纠正，等同 high 可信
_BAR_KEY_RE = re.compile(r"-?[0-9]+")
_NO_CHORD_LABELS = ("N", "N.C.", "NC")


def check_signatures(beats: dict, chords: dict, melody: dict) -> None:
    """chords.json / melody.json 记下的 beats_signature 与当前 beats.json 不符 → 报错；
    旧文件没有签名、或签名缺字段（如还没有 feel）→ 只警告，并只比较双方都有的字段。"""
    sig = beats_signature(beats)
    for name, data, step in (("chords.json", chords, "chords"), ("melody.json", melody, "melody")):
        got = data.get("beats_signature")
        if got is None:
            print(f"[leadsheet] warning: {name} has no beats_signature (older file); cannot tell whether it matches "
                  f"beats.json — re-run --steps {step} if beats changed", file=sys.stderr)
            continue
        if any(got[k] != sig[k] for k in got.keys() & sig.keys()):
            raise ValueError(f"{name} was computed from an older beats.json; re-run --steps {step} "
                             f"({name} {got} vs beats.json {sig})")
        missing = sorted(sig.keys() - got.keys())
        if missing:
            print(f"[leadsheet] warning: {name} beats_signature has no {', '.join(missing)} (older file); cannot tell "
                  f"whether that matches beats.json — re-run --steps {step} if beats changed", file=sys.stderr)


def check_melody_key(melody: dict, key: tuple[str, str]) -> None:
    """melody.json 记下的量化调（旋律音按它吸附 / 标调外）与 leadsheet 要用的调不同 → 报错；旧文件没有记 → 只警告。"""
    got = melody.get("key")
    if not (isinstance(got, dict) and "tonic" in got and "mode" in got):
        print(f"[leadsheet] warning: melody.json has no key (older file); cannot tell whether it was computed in "
              f"{key[0]} {key[1]} — re-run --steps melody if the key changed", file=sys.stderr)
    elif (theory.pc(got["tonic"]), got["mode"]) != (theory.pc(key[0]), key[1]):
        raise ValueError(f"melody.json was computed in {got['tonic']} {got['mode']}; re-run --steps melody "
                         f"(the leadsheet will use {key[0]} {key[1]})")


def _bar_number(key) -> int:
    """overrides.chords 的小节键：int 或纯数字字符串（YAML 两种写法都有）→ int；其它（含 bool）→ ValueError。"""
    if isinstance(key, int) and not isinstance(key, bool):
        return key
    if isinstance(key, str) and _BAR_KEY_RE.fullmatch(key.strip()):
        return int(key.strip())
    raise ValueError(f"overrides.chords bar {key}: not a bar number (leadsheet bars are numbered from 1)")


def _override_label(bar: int, token) -> str:
    """一个覆盖标签 → 规范标签：N / N.C. / NC → N；其余必须被 theory.parse_chord 认作和弦（含 E5 强力和弦、转位）。"""
    if not isinstance(token, str):
        raise ValueError(f"overrides.chords bar {bar}: chord label must be text, got {token!r}")
    label = token.strip()
    if label in _NO_CHORD_LABELS:
        return "N"
    try:
        parsed = theory.parse_chord(label)
    except ValueError:
        parsed = None
    if parsed is None:  # 解析失败；空串、"-" 这类 parse_chord 当作"无和弦"的写法也不收，无和弦只写 N
        raise ValueError(f"overrides.chords bar {bar}: bad chord label {token!r}")
    return label


def _override_labels(bar: int, value, bpb: int) -> list[str]:
    """一小节的覆盖值 → 标签列表。字符串按空格分，标签数须整除 bpb（均分）；列表须恰好 bpb 个（逐拍）。"""
    if isinstance(value, str):
        tokens = value.split()
        if not tokens:
            raise ValueError(f"overrides.chords bar {bar}: bad chord label {value!r} (empty; write N for no chord)")
        if bpb % len(tokens):
            raise ValueError(f"overrides.chords bar {bar}: {len(tokens)} labels do not split {bpb} beats evenly "
                             f"({value!r}); give one label per beat as a list instead")
    elif isinstance(value, list):
        tokens = value
        if len(tokens) != bpb:
            raise ValueError(f"overrides.chords bar {bar}: a list needs one label per beat ({bpb} beats), "
                             f"got {len(tokens)}")
    else:
        raise ValueError(f"overrides.chords bar {bar}: expected a label string or a list of {bpb} labels, got {value!r}")
    return [_override_label(bar, t) for t in tokens]


def parse_chord_overrides(raw, bar_numbers: set[int], bpb: int) -> dict[int, list[str]]:
    """overrides.chords → {小节号: 标签列表}（按小节号排序）。全部校验完才返回，出错时一个小节都不套用。"""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"overrides.chords must be a mapping of bar number -> chord labels, got {type(raw).__name__}")
    out: dict[int, list[str]] = {}
    for key, value in raw.items():
        bar = _bar_number(key)
        if bar not in bar_numbers:
            raise ValueError(f"overrides.chords bar {bar}: no such bar (the leadsheet has {len(bar_numbers)} bars, "
                             "numbered from 1)")
        if bar in out:
            raise ValueError(f"overrides.chords bar {bar}: given twice")
        out[bar] = _override_labels(bar, value, bpb)
    return dict(sorted(out.items()))


def _bar_beats(bar: dict, bpb: int, pad_beats: int) -> range:
    """小节里真有拍点的拍号（1-based）：弱起补拍的第 1 小节缺开头的拍，末尾不满的小节缺后面的拍。"""
    n = min(bar.get("beats", bpb), bpb)
    return range(bpb - n + 1, bpb + 1) if (bar["bar"] == 1 and pad_beats) else range(1, n + 1)


def manual_segments(labels: list[str], bpb: int, beats: range) -> list[dict]:
    """覆盖标签 → 和弦段：标签均分 bpb 拍，只取 beats 里的拍，相邻同名合并。confidence 记为 manual。"""
    per_beat = [label for label in labels for _ in range(bpb // len(labels))]
    segments: list[dict] = []
    for beat in beats:
        label = per_beat[beat - 1]
        if segments and segments[-1]["label"] == label:
            segments[-1]["dur_beats"] += 1
        else:
            segments.append({"beat": beat, "dur_beats": 1, "label": label, "confidence": "manual",
                             "votes": {"manual": label}})
    return segments


def build_leadsheet(beats: dict, chords: dict, melody: dict, key: tuple[str, str], sections: dict[int, str],
                    chord_overrides: dict | None = None) -> dict:
    """chord_overrides 即 song.yaml 的 overrides.chords（格式见 SongConfig）：只替换所指小节的和弦段，
    chords.json 不动，所以重跑 chords / 重新生成后人耳纠正仍在。"""
    if beats.get("short_bars"):
        raise ValueError(f"beats.json has short bars {beats['short_bars']}: genuine 2/4 or 3/4 bars (common in "
                         "TV-size edits) are not supported yet — the leadsheet assumes every bar has 4 beats. "
                         "Only if the downbeat phase was mis-detected (bar lines off by a beat or two) is "
                         "overrides.downbeat_shift the fix; then re-run --steps beats")
    check_signatures(beats, chords, melody)
    check_melody_key(melody, key)
    bpb = beats["time_signature"][0]
    manual = parse_chord_overrides(chord_overrides, {b["bar"] for b in beats["bars"]}, bpb)
    for n, labels in manual.items():
        print(f"[leadsheet] bar {n} chords overridden: {' '.join(labels)}", file=sys.stderr)
    by_bar: dict[int, list[dict]] = {}
    for s in chords["segments"]:
        by_bar.setdefault(s["bar"], []).append(s)
    bars = []
    for b in beats["bars"]:
        n = b["bar"]
        g0 = (n - 1) * bpb
        segments = (manual_segments(manual[n], bpb, _bar_beats(b, bpb, beats.get("pad_beats", 0)))
                    if n in manual else by_bar.get(n, []))
        bars.append({
            "bar": n,
            "section": sections.get(n),
            "t_start": b["t_start"],
            "t_end": b["t_end"],
            "chords": [{k: s[k] for k in ("beat", "dur_beats", "label", "confidence", "votes")} for s in segments],
            "melody": [{k: m[k] for k in ("start_beats", "dur_beats", "midi", "conf", "out_of_key")}
                       for m in melody["notes"] if g0 <= m["start_beats"] < g0 + bpb],
        })
    return {"key": key[0], "mode": key[1], "tempo_bpm": beats["tempo_bpm"], "feel": beats["feel"],
            "beats_per_bar": bpb, "pickup_beats": beats["pickup_beats"], "bars": bars}


def _votes_note(chord: dict) -> str:
    """high 不加注；manual 只写 [manual]（没有投票）；其余列出各方投票。"""
    if chord["confidence"] == "high":
        return ""
    if chord["confidence"] == "manual":
        return "[manual]"
    return "[" + " ".join(f"{k}={v}" for k, v in chord["votes"].items()) + "]"


def summary_table(leadsheet: dict, capo: int = 0) -> str:
    """一小节一行：小节号 · 起始秒 · 段落 | 拍:和弦(指法调)置信标记[投票] | 旋律音（* = 调外）。
    置信标记：无 = high，? = medium，?? = low，! = manual（人耳纠正，括号里写 [manual] 而非投票）。"""
    head = f"key {leadsheet['key']} {leadsheet['mode']} · ♩≈{leadsheet['tempo_bpm']} · {leadsheet['feel']}"
    lines = [head + (f" · capo {capo}" if capo else "")]
    for b in leadsheet["bars"]:
        chords = []
        for c in b["chords"]:
            label = c["label"]
            if capo and label != "N":
                label = f"{label}({theory.transpose_label(label, -capo)})"
            chords.append(f"{c['beat']}:{label}{_MARK[c['confidence']]}{_votes_note(c)}")
        melody = " ".join(theory.midi_name(m["midi"]) + ("*" if m.get("out_of_key") else "") for m in b["melody"])
        section = f"[{b['section']}] " if b["section"] else ""
        lines.append(f"{b['bar']:3d} {b['t_start']:6.1f}s {section}| {'  '.join(chords)} | {melody}")
    return "\n".join(lines)
