import re
"""编配编排：自检（真实 alphaTab 构建）、修补回归、4/4 限制、命令行输出。"""
import json

import pytest
from conftest import capo4_bar12_leadsheet

from guitarize.arrange import atex
from guitarize.arrange.finger import generate_finger
from guitarize.arrange.pipeline import self_check
from guitarize.cli import main
from guitarize.config import load_profile
from guitarize.render.alphatab import AlphaTexError

MID = load_profile("intermediate")


def test_self_check_passes_clean_draft_and_flags_span():
    good = atex.header("T", 100, 0, "straight", "tabs") + "// bar 1\n(3.6 1.4).1 |\n"
    assert self_check(good, MID, "finger") == []
    bad = atex.header("T", 100, 0, "straight", "tabs") + "// bar 1\n(8.1 1.6).1 |\n"
    found = self_check(bad, MID, "finger")
    assert [(v.bar, v.rule) for v in found] == [(1, "span")]


def _bar(i, section, mel):
    return {"bar": i + 1, "section": section, "t_start": i * 1.44, "t_end": (i + 1) * 1.44,
            "chords": [{"beat": 1, "dur_beats": 4, "label": "D#/G", "confidence": "high"}],
            "melody": [{"start_beats": i * 4 + s, "dur_beats": d, "midi": m, "conf": 1.0} for s, d, m in mel]}


def _regression_sheet():
    # 评审回归（capo 4 第 47–48 小节的样子）：实际 D#/G 在 capo 4 下是 B7/D#，低音 D#3 = 4 弦 1 品。
    # 第 1 小节旋律在 1 弦 0–1 品；第 2 小节第 1 拍空弦 E4 + 低音 4 弦 1 品（二分音符），
    # 第 2–4 拍旋律 A4 / A#4 / A4 要 1 弦 5–6 品——低音还按着，跨度 1–5 超限，只有渲染后的自检看得见。
    bar1 = _bar(0, "サビ", [(0, 1, 69), (1, 1, 68), (2, 1, 69), (3 + 2 / 3, 1, 69)])
    bar2 = _bar(1, None, [(0, 1, 68), (1, 1, 73), (2, 1, 74), (3, 1, 73)])
    return {"key": "B", "mode": "major", "tempo_bpm": 167.0, "feel": "shuffle", "beats_per_bar": 4,
            "bars": [bar1, bar2]}


def test_b7_over_d_sharp_regression_has_no_violations_after_repair():
    calls = []

    def check(text):
        calls.append(self_check(text, MID, "finger"))
        return calls[-1]

    text = generate_finger(_regression_sheet(), 4, MID, "T", check=check)
    assert [(v.bar, v.rule) for v in calls[0]] == [(2, "span")]      # 修补前：第 2 小节第 2 拍跨度超限
    assert calls[-1] == [] and self_check(text, MID, "finger") == []
    assert "// bar 2 · bass cut @2 (span)" in text
    v2_bar2 = text.split("\\voice")[2].split("// bar 2\n")[1].splitlines()[0]
    assert v2_bar2.startswith("(") and "1.4)" in v2_bar2.split()[1]  # 低音 D#3 仍在 4 弦 1 品，只是截短


def test_arrange_rejects_non_four_four(demo_song):
    ls = json.loads(demo_song.leadsheet_json.read_text(encoding="utf-8"))
    ls["beats_per_bar"] = 3
    demo_song.leadsheet_json.write_text(json.dumps(ls), encoding="utf-8")
    demo_song.atex("finger").unlink()
    with pytest.raises(ValueError, match=r"only 4/4 is supported for now \(beats_per_bar=3\)"):
        main(["--songs-dir", str(demo_song.root.parent), "arrange", "demo", "--arr", "finger"])
    assert not demo_song.atex("finger").exists()


def test_arrange_prints_self_check(demo_song, capsys):
    songs = str(demo_song.root.parent)
    for arr in ("strum", "finger"):
        assert main(["--songs-dir", songs, "arrange", "demo", "--arr", arr, "--force"]) == 0
    out = capsys.readouterr().out
    assert out.count("自检：0 处违规") == 2


def test_arrange_lists_violations_without_failing(demo_song, capsys, monkeypatch):
    from guitarize.arrange import pipeline
    from guitarize.verify.playability import Violation

    fake = [Violation(b, 4.0 * (b - 1), "span", f"跨度 {b}") for b in range(1, 8)]
    monkeypatch.setattr(pipeline, "self_check", lambda text, profile, kind: fake)
    assert main(["--songs-dir", str(demo_song.root.parent), "arrange", "demo", "--arr", "strum", "--force"]) == 0
    out = capsys.readouterr().out
    assert "自检：7 处违规" in out
    assert "bar 1 span：跨度 1" in out and "bar 5 span：跨度 5" in out
    assert "bar 6 span" not in out


def test_tied_bass_counts_as_held_in_self_check():
    # 和弦只延续低音（连音线）：verify 的手型把延续的 4 弦 1 品算作仍按着
    head = atex.header("T", 100, 0, "straight", "tabs")

    def doc(melody):
        return head + "\\voice\n// bar 1\n" + melody + " |\n\\voice\n// bar 1\n(0.2 1.4).4 -.4.4 r.2 |\n"

    found = self_check(doc("0.1.4 6.1.4 0.1.2"), MID, "finger")
    assert [(v.start, v.rule) for v in found] == [(1.0, "span")]    # 1–6 品
    assert self_check(doc("0.1.4 3.1.4 0.1.2"), MID, "finger") == []


def test_fill_cut_then_shift_is_repaired_and_keeps_the_beat3_bass():
    # 填充被同弦旋律截断、低音接着响 → 第 3 拍新低音（5 弦 6 品）离它 5 品：真实自检报换把，
    # 修补截短前一个起音时仍按着的低音，第 3 拍低音保留
    calls = []

    def check(text):
        calls.append(self_check(text, MID, "finger"))
        return calls[-1]

    text = generate_finger(capo4_bar12_leadsheet(), 4, MID, "T", check=check)
    assert [(v.bar, v.rule) for v in calls[0]] == [(1, "shift")]
    assert calls[-1] == [] and len(calls) == 2
    assert "// bar 1 [サビ] · fill cut @2 (same string) · bass cut @2.5 (shift)" in text
    bar = text.split("\\voice")[2].split("// bar 1\n")[1].splitlines()[0]
    assert re.sub(r' \{(?:ch "[^"]*"\s*)?(?:dy \w+)?\}', "", bar) == "(0.2 1.4).4 -.4.8 r.8 (0.2 6.5).2 |"


@pytest.mark.parametrize("error", [
    AlphaTexError("alphaTex build failed for /tmp/x/draft.atex:\nL4:9 Unexpected 'Dot' token."),
    FileNotFoundError(2, "No such file or directory", "node"),
])
def test_arrange_survives_a_failed_self_check(demo_song, capsys, monkeypatch, error):
    from guitarize.render import alphatab

    def build(*args, **kwargs):
        raise error

    monkeypatch.setattr(alphatab, "build", build)
    songs = str(demo_song.root.parent)
    for arr in ("finger", "strum"):
        demo_song.atex(arr).unlink()
        assert main(["--songs-dir", songs, "arrange", "demo", "--arr", arr]) == 0
        assert demo_song.atex(arr).read_text(encoding="utf-8").count("\\track") == 1    # 草稿照写
    out = capsys.readouterr().out
    wanted = "L4:9 Unexpected 'Dot' token." if isinstance(error, AlphaTexError) else str(error).splitlines()[0]
    assert out.count(f"自检失败：{wanted}") == 2     # alphaTex 的定位诊断优先于临时文件路径那行
    assert "处违规" not in out


def test_strum_draft_is_checked_inside_generate_strum(demo_song, monkeypatch):
    from guitarize.arrange import pipeline

    inside, calls = [False], []
    generate = pipeline.generate_strum

    def spy(*args, **kwargs):
        inside[0] = True
        try:
            return generate(*args, **kwargs)
        finally:
            inside[0] = False

    monkeypatch.setattr(pipeline, "generate_strum", spy)
    monkeypatch.setattr(pipeline, "self_check", lambda text, profile, kind: calls.append((kind, inside[0])) or [])
    assert main(["--songs-dir", str(demo_song.root.parent), "arrange", "demo", "--arr", "strum", "--force"]) == 0
    assert calls == [("strum", True)]      # 生成器的修补循环里检查过；最后的自检用同一份结果


def test_first_line_prefers_the_alphatex_diagnostic():
    from guitarize.arrange.pipeline import _first_line
    err = RuntimeError("alphaTex build failed for /tmp/x/draft.atex:\n  L12:8 Unexpected end of file\n  more")
    assert _first_line(err) == "L12:8 Unexpected end of file"
    assert _first_line(RuntimeError("node not found")) == "node not found"
