from guitarize.render import alphatab
from guitarize.render.ascii_tab import render_ascii
from guitarize.verify.model import load_score


def test_single_bar_layout(build_tex):
    text = render_ascii(build_tex(":4 0.1 1.1 2.1 3.1 |"))
    lines = text.splitlines()
    assert "e|0---1---2---3---|" in lines
    assert "B|----------------|" in lines
    assert "E|----------------|" in lines


def test_shuffle_and_capo_header(build_tex):
    from conftest import TEX_HEAD

    text = render_ascii(build_tex("\\tf triplet-8th\n:1 0.1 |", head=TEX_HEAD + "\\capo 4\n"))
    assert "Capo 4" in text and "Shuffle" in text


def test_demo_strum_has_chords_and_arrows(demo_song, tmp_path):
    score = load_score(alphatab.build(demo_song.atex("strum"), tmp_path / "b", "s")["json"])
    text = render_ascii(score)
    lines = text.splitlines()
    chord_lines = [l for l in lines if l.lstrip().startswith("C ")]
    assert chord_lines, text
    assert any("↓" in l and "↑" in l for l in lines)
    assert sum(1 for l in lines if l.startswith("e|")) == 2  # 8 小节 / 每行 4 小节
