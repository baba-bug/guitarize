"""合并 beats / chords / melody → leadsheet.json（Phase 1 校验器的参考），并输出便于阅读的小节摘要。"""
from __future__ import annotations

from .. import theory

_MARK = {"high": "", "medium": "?", "low": "??"}


def build_leadsheet(beats: dict, chords: dict, melody: dict, key: tuple[str, str], sections: dict[int, str]) -> dict:
    if beats.get("short_bars"):
        raise ValueError(f"beats.json has short bars {beats['short_bars']}: fix with overrides.downbeat_shift "
                         "before building the leadsheet (melody bucketing assumes 4-beat bars)")
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
