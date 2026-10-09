"""小节内位置的乐手记法：1、1e、1&、1a、1+⅓。"""
from __future__ import annotations

TOL = 0.02


def beat_label(pos_in_bar: float) -> str:
    """pos_in_bar 为小节内 0-based 拍数；返回「拍号 + 细分」，如 3、3&、3e、3a、3+⅓。"""
    n = int(pos_in_bar)
    frac = pos_in_bar - n
    if frac > 1 - TOL:
        n, frac = n + 1, 0.0
    beat = n + 1
    if frac < TOL:
        return str(beat)
    for target, suffix in ((0.5, "&"), (2 / 3, "&"), (1 / 3, "+⅓"), (0.25, "e"), (0.75, "a")):
        if abs(frac - target) < TOL:
            return f"{beat}{suffix}"
    return f"{pos_in_bar + 1:.2f}"
