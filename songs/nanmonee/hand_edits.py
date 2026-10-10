"""なんもねえ 的手改（在 `guitarize arrange --force` 重新生成之后重放，生成器改进时手改不会丢）。

用法：python songs/nanmonee/hand_edits.py strum|finger [歌曲目录]（默认是本脚本所在的目录）
依据：贝斯分轨逐格音高 + 鼓 / 吉他分轨起音（间奏 / 尾奏 riff、37 小节 break、62 小节结尾一击），
以及逐小节和声相似度（scores.json 的 harmonic_sim）——只保留让它变高、且不引入可弹性违规的改动。
指弹是完整旋律写法（合音由生成器配）：只替换整小节的 riff / break / 结尾一击，旧的填充补丁（C 指型补空弦 G、
B7 指型补顶音等）已由生成器的合音取代。
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
    stop = ("r.2 0.6.4 {tu 3} 2.6.4 {tu 3} 0.4.4 {tu 3} |", "riff (bass)")
    b_riff = ("3.6.2 0.6.4 {tu 3} 3.6.4 {tu 3} 0.6.4 {tu 3} |", "riff (bass)")
    e_riff = ("3.5.4 2.4.8 0.3.8 0.4.4 4.4.8 2.3.8 |", "riff (bass)")
    # break：D 的根音 + 五音（3 弦 2 品 + 4 弦空弦）——旋律第 1 拍的 C4 在 2 弦 1 品，4 弦 4 品 + 5 弦 5 品跨度放不下；
    # 结尾一击：G 和弦，最高音落在 1 弦 3 品的 G（主音）上——人声在第 58 小节就唱完了，旋律声部这一小节是休止
    v2 = _edit(v2, {19: stop, 59: stop, 20: b_riff, 60: b_riff, 21: e_riff, 61: e_riff,
                    37: ("(2.3 0.4).4 (2.3 0.4).4 r.2 |", "break before サビ"),
                    62: ("r.4 (3.1 0.4 2.5 3.6).2 {d} |", "final hit, ending on G")})

    def marks(old: str, name: str) -> list[str]:      # 指型名可能和别的拍效果写在一起（三连音拍里是 {ch "D" tu 3}）
        return ["{" + m + "}" for m in re.findall('ch "' + re.escape(name) + '"', old)]

    v1 = _edit(v1, {
        21: (lambda old: " ".join(["r.2", *marks(old, "C"), "r.2", *marks(old, "D")]) + " |", "riff bar: melody rests"),
    })
    return "\\voice\n".join([head, v1, v2])


if __name__ == "__main__":
    arr = sys.argv[1]
    path = (Path(sys.argv[2]) if len(sys.argv) > 2 else HERE) / "arrangements" / f"{arr}.atex"
    path.write_text({"strum": strum, "finger": finger}[arr](path.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"hand edits applied: {path}")
