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


STRING_PREFIXES = ("e|", "B|", "G|", "D|", "A|", "E|")


def test_empty_bar_has_no_stray_rows(build_tex):
    text = render_ascii(build_tex(":4 0.1 1.1 2.1 3.1 | | :1 3.1 |"))
    for line in text.splitlines()[2:]:  # 前两行是标题和信息行
        if line.startswith(STRING_PREFIXES):
            continue
        assert "-" not in line, text


def test_long_section_label_keeps_alignment(build_tex):
    text = render_ascii(build_tex(
        '\\section "Pre-Chorus Variation"\n:4 0.1 0.1 0.1 0.1 | \\section "Z"\n:4 0.1 0.1 0.1 0.1 |'
    ))
    lines = text.splitlines()
    top = next(l for l in lines if "[Z]" in l)
    e_row = next(l for l in lines if l.startswith("e|"))
    closing = e_row.index("|", 2)  # 第 1 小节的收尾竖线
    assert top.index("2 [Z]") == closing + 1, text
    string_rows = [l for l in lines if l.startswith(STRING_PREFIXES)]
    assert len(string_rows) == 6 and len({len(l) for l in string_rows}) == 1, text


def test_japanese_section_label_alignment(build_tex):
    from guitarize.render.ascii_tab import _display_width

    text = render_ascii(build_tex(
        '\\section "サビ"\n:4 0.1 0.1 0.1 0.1 | \\section "Aメロ"\n:4 0.1 0.1 0.1 0.1 |'
    ))
    lines = text.splitlines()
    top = next(l for l in lines if "2 [Aメロ]" in l)
    e_row = next(l for l in lines if l.startswith("e|"))
    closing = e_row.index("|", 2)  # 第 1 小节的收尾竖线（ASCII 行，列数即显示宽度）
    assert _display_width(top[: top.index("2 [Aメロ]")]) == closing + 1, text
