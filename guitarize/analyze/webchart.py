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
