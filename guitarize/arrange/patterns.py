"""节奏型：8 个八分格。按与原曲起音的相关度选型。

弹唱（strum，PATTERNS）：D = 下扫，U = 上扫，x = 闷音下扫，- = 延续。
强力和弦弹唱（power，POWER_PATTERNS）：A = 空弦重音下扫（f + 重音），D = 下扫（f），M = 闷音（palm mute）下扫（f），
d = 轻的下扫（mp），U = 上扫（mp），- = 延续（让和弦接着响）。强力和弦的型按原曲吉他分轨（なんもねえ 的两把失真吉他）
逐段归纳，只用吉他分轨选型（arrange/power.py）；重音向量的权重按吉他分轨里的起音强度给：闷音很暗，起音只有 0.3。"""
from __future__ import annotations

import numpy as np

PATTERNS = {
    "whole": "D-------",
    "half": "D---D---",
    "quarter": "D-D-D-D-",
    "push": "D-DUD-DU",
    "pop": "D-DU-UDU",
    "eighth": "DUDUDUDU",
}
DEFAULT = "push"
POWER_PATTERNS = {
    "verse": "M-A-MUD-",       # Aメロ：1 闷音，2 空弦重音，3 闷音 + 反拍短促的空弦上扫，4 空弦；2、4 的反拍不弹
    "verse_end": "D-AU----",   # Bメロ 后半：1、2、2 的反拍，然后延音到小节末
    "chorus": "A-d--d--",      # サビ：1 重扫 + 延音的音墙，2、3 的反拍轻扫（摇摆八分的弱脉动）
    "hit": "A-------",         # 間奏 / 尾奏：一击 + 余音
}
POWER_DEFAULT = "verse"   # 没有吉他分轨时
POWER_FLAT = "hit"        # 吉他分轨有，但起音没有可辨认的节奏（持续的音墙 / 余音）
_WEIGHT = {"D": 1.0, "x": 0.8, "U": 0.6, "-": 0.0, "A": 1.0, "M": 0.3, "d": 0.6}


def slots_per_beat(feel: str) -> int:
    return 3 if feel == "shuffle" else 2


def accent_vector(pattern: str, feel: str) -> np.ndarray:
    """节奏型 → 每拍 slots_per_beat 格的重音向量；反拍八分落在 swing 位置（shuffle 为第 3 格）。"""
    spb = slots_per_beat(feel)
    v = np.zeros(4 * spb)
    for k, ch in enumerate(pattern):
        pos = (k // 2) * spb + (0 if k % 2 == 0 else spb - 1)
        v[pos] = max(v[pos], _WEIGHT[ch])
    return v


def correlations(onset: np.ndarray | None, feel: str, table: dict[str, str] = PATTERNS) -> dict[str, float]:
    """table 里每个节奏型的重音向量与原曲起音向量的相关度；没有数据或起音向量没有起伏时为空。"""
    if onset is None or np.std(onset) == 0:
        return {}
    return {name: float(np.corrcoef(accent_vector(p, feel), onset)[0, 1]) for name, p in table.items()}


def choose_pattern(onset: np.ndarray | None, feel: str, table: dict[str, str] = PATTERNS,
                   default: str = DEFAULT) -> str:
    """与原曲起音向量相关度最高的节奏型；相关度相同取扫弦次数少的；无数据时用默认型。"""
    cors = correlations(onset, feel, table)
    if not cors:
        return default
    return max(table, key=lambda name: (round(cors[name], 3), -sum(c != "-" for c in table[name])))
