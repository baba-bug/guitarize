"""编配编排：自检（真实 alphaTab 构建）、修补回归、4/4 限制、命令行输出。"""
import json

import pytest

from guitarize.arrange import atex
from guitarize.arrange.finger import generate_finger
from guitarize.arrange.pipeline import self_check
from guitarize.cli import main
from guitarize.config import load_profile

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
