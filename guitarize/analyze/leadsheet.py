"""合并 beats / chords / melody → leadsheet.json（Phase 1 校验器的参考），并输出便于阅读的小节摘要。"""
from __future__ import annotations

import sys

from .. import theory
from .beats import beats_signature

_MARK = {"high": "", "medium": "?", "low": "??"}


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


def build_leadsheet(beats: dict, chords: dict, melody: dict, key: tuple[str, str], sections: dict[int, str]) -> dict:
    if beats.get("short_bars"):
        raise ValueError(f"beats.json has short bars {beats['short_bars']}: genuine 2/4 or 3/4 bars (common in "
                         "TV-size edits) are not supported yet — the leadsheet assumes every bar has 4 beats. "
                         "Only if the downbeat phase was mis-detected (bar lines off by a beat or two) is "
                         "overrides.downbeat_shift the fix; then re-run --steps beats")
    check_signatures(beats, chords, melody)
    check_melody_key(melody, key)
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
