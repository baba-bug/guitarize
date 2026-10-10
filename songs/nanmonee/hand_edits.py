"""なんもねえ 的手改（在 `guitarize arrange --force` 重新生成之后重放，生成器改进时手改不会丢）。

用法：python songs/nanmonee/hand_edits.py strum|finger
依据：贝斯分轨逐格音高 + 鼓 / 吉他分轨起音（间奏 / 尾奏 riff、37 小节 break、62 小节结尾一击），
以及逐小节和声相似度（scores.json 的 harmonic_sim）——只保留让它变高、且不引入可弹性违规的改动。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
G = "(3.1 0.2 0.3 0.4 2.5 3.6)"


def _edit(voice: str, edits: dict[int, tuple[str, str]]) -> str:
    """edits：{小节: (新的一行, 说明)}；新行里写 {old} 时代入原行（用于在原行上做替换）。"""
    lines = voice.split("\n")
    hit = set()
    for i, line in enumerate(lines):
        m = re.match(r"// bar (\d+)", line)
        if m and int(m.group(1)) in edits:
            n = int(m.group(1))
            body, note = edits[n]
            old = lines[i + 1]
            prefix = re.match(r'(\\section "[^"]*" )?', old).group(0)
            new = body(old[len(prefix):]) if callable(body) else body
            if new != old[len(prefix):]:
                lines[i] = line + " · hand: " + note
                lines[i + 1] = prefix + new
            hit.add(n)
    missing = set(edits) - hit
    assert not missing, f"bars not found: {sorted(missing)}"
    return "\n".join(lines)


def strum(text: str) -> str:
    stop = "r.2 (0.6 2.5).4 {tu 3} (2.6 4.5).4 {tu 3} (0.4 2.3).4 {tu 3} |"
    b_riff = f'{G}.2 {{ch "G" sd}} (0.6 2.5).4 {{tu 3}} (3.6 5.5).4 {{tu 3}} (0.6 2.5).4 {{tu 3}} |'
    e_riff = '3.5.4 {ch "C"} 2.4.8 0.3.8 0.4.4 {ch "D"} 4.4.8 2.3.8 |'
    return _edit(text, {
        19: (stop, "stop + triplet hits"), 59: (stop, "stop + triplet hits"),
        20: (b_riff, "riff hits"), 60: (b_riff, "riff hits"),
        21: (e_riff, "riff"), 61: (e_riff, "riff"),
        37: ('(2.1 3.2 2.3 0.4).4 {ch "D" sd} (2.1 1.2 2.3 0.4).4 {ch "D7" sd} r.2 |', "break before サビ"),
        62: (f'r.4 {G}.2 {{ch "G" sd d}} |', "final hit"),
    })


def finger(text: str) -> str:
    head, v1, v2 = text.split("\\voice\n")
    melody = {int(m.group(1)): body for m, body in
              ((re.match(r"// bar (\d+)", a), b) for a, b in zip(v1.split("\n"), v1.split("\n")[1:])) if m}

    def full_c(old: str) -> str:                       # C 指型补空弦 G：完整三和弦
        return old.replace("(2.4 3.5)", "(0.3 2.4 3.5)")

    def b7_top(old: str) -> str:                       # B7 指型补 1 弦 2 品 F#（指型本来的顶音）
        return old.replace("(1.4 2.5)", "(2.1 1.4 2.5)")

    stop = ("r.2 0.6.4 {tu 3} 2.6.4 {tu 3} 0.4.4 {tu 3} |", "riff (bass)")
    b_riff = ("3.6.2 0.6.4 {tu 3} 3.6.4 {tu 3} 0.6.4 {tu 3} |", "riff (bass)")
    e_riff = ("3.5.4 2.4.8 0.3.8 0.4.4 4.4.8 2.3.8 |", "riff (bass)")
    edits2 = {19: stop, 59: stop, 20: b_riff, 60: b_riff, 21: e_riff, 61: e_riff,
              37: ("(4.4 5.5).4 (4.4 5.5).4 r.2 |", "break before サビ"),
              62: ("r.4 (0.4 2.5 3.6).2 {d} |", "final hit"),
              35: (lambda old: re.sub(r"2\.5\.2 \|$", "(4.2 2.5).2 |", old), "D# (B-shape third) on string 2"),
              36: (lambda old: old.replace("(2.4 3.5)", "(3.1 2.4 3.5)"), "G on top")}
    for n in (3, 6, 11, 14, 39, 42, 47, 50):
        edits2[n] = (full_c, "full C triad (open G)")
    for n, body in melody.items():
        if n not in edits2 and not re.search(r"\b\d+\.1\.", body):
            edits2[n] = (b7_top, "B7-shape top F#")
    v2 = _edit(v2, edits2)
    def marks(old: str, name: str) -> list[str]:      # 指型名可能和别的拍效果写在一起（三连音拍里是 {ch "D" tu 3}）
        return ["{" + m + "}" for m in re.findall(f'ch "{name}"', old)]

    v1 = _edit(v1, {
        21: (lambda old: " ".join(["r.2", *marks(old, "C"), "r.2", *marks(old, "D")]) + " |", "riff bar: melody rests"),
        62: (lambda old: old.replace("4.2.2 |", "3.1.2 |"), "end on G"),
    })
    return "\\voice\n".join([head, v1, v2])


if __name__ == "__main__":
    arr = sys.argv[1]
    path = HERE / "arrangements" / f"{arr}.atex"
    path.write_text({"strum": strum, "finger": finger}[arr](path.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"hand edits applied: {path}")
