import pytest

from guitarize.arrange import atex
from guitarize.render import alphatab
from guitarize.verify.model import load_score


def test_split_beats():
    assert atex.split_beats(4) == [(1, False)]
    assert atex.split_beats(1.5) == [(4, True)]
    assert atex.split_beats(2.5) == [(2, False), (8, False)]
    assert atex.split_beats(0.75) == [(8, True)]
    with pytest.raises(ValueError):
        atex.split_beats(0.1)


def test_tokens():
    assert atex.notes_token([]) == "r"
    assert atex.notes_token([(1, 3)]) == "3.1"
    assert atex.notes_token([(6, 3), (5, 2)]) == "(3.6 2.5)"
    assert atex.notes_token([(1, 3)], tie=True) == "-.1"
    assert atex.notes_token([(6, 0), (5, 0)], dead=True) == "(x.6 x.5)"
    assert atex.beat_tokens([(1, 3)], 2.5, ['ch "G"', "sd"]) == ['3.1.2 {ch "G" sd}', "-.1.8"]
    assert atex.beat_tokens([], 1.5) == ["r.4 {d}"]


def test_split_for_swing_pairs_offbeat_onsets(tmp_path):
    items = [atex.Item(0, 1, [(1, 3)]), atex.Item(1, 0.5, []), atex.Item(1.5, 2.5, [(1, 5)], ["sd"])]
    pieces = atex.split_for_swing(items, "shuffle")
    assert [(p.start, p.length, cont) for p, cont in pieces] == [(0, 1, False), (1, 0.5, False), (1.5, 0.5, False), (2.0, 2.0, True)]
    assert pieces[2][0].effects == ["sd"] and pieces[3][0].effects == []
    assert atex.split_for_swing(items, "straight") == [(it, False) for it in items]
    tokens = atex.voice_bar_tokens(items, "shuffle")
    assert tokens == ["3.1.4", "r.8", "5.1.8 {sd}", "-.1.2"]
    head = atex.header("T", 120, 0, "shuffle", "tabs")
    src = tmp_path / "s.atex"
    src.write_text(atex.document(head, [[atex.bar_line(tokens, 1)]]), encoding="utf-8")
    score = load_score(alphatab.build(src, tmp_path / "b", "s")["json"])
    from guitarize.config import load_profile
    from guitarize.verify.playability import check_playability
    assert check_playability(score, load_profile("intermediate"), "finger") == []
    assert [round(a.start, 3) for a in score.attacks()] == [0.0, 1.667]


def test_document_builds_with_alphatab(tmp_path):
    head = atex.header('My "Song"', 168.4, 4, "shuffle", "score tabs", {"G": (3, 0, 0, 0, 2, 3)})
    v1 = [atex.bar_line(atex.beat_tokens([(1, 3)], 4, ['ch "G"']), 1, "サビ")]
    v2 = [atex.bar_line(atex.beat_tokens([(6, 3)], 4), 1)]
    text = atex.document(head, [v1, v2])
    assert "// bar 1 [サビ]" in text and '\\section "サビ"' in text and text.count("\\voice") == 2
    src = tmp_path / "t.atex"
    src.write_text(text, encoding="utf-8")
    score = load_score(alphatab.build(src, tmp_path / "b", "t")["json"])
    assert score.capo == 4 and score.tempo == 168 and score.bars[0].section == "サビ"
    assert score.bars[0].feel == "triplet8th"
    assert [n.midi for n in score.melody(0)] == [71]


def test_quote_escapes_strings():
    assert atex.quote('Outro "end"') == '"Outro \\"end\\""'
    assert atex.quote("a\\b") == '"a\\\\b"'
    assert atex.quote("x\ny") == '"x y"'


def test_names_with_quotes_and_backslashes_build(tmp_path):
    head = atex.header('C:\\songs "x"', 120, 0, "straight", "tabs", {'G"x': (3, 0, 0, 0, 2, 3)})
    bar = atex.bar_line(atex.beat_tokens([(1, 3)], 4, ['ch "G"']), 1, 'Outro "end"\\')
    src = tmp_path / "q.atex"
    src.write_text(atex.document(head, [[bar]]), encoding="utf-8")
    score = load_score(alphatab.build(src, tmp_path / "b", "q")["json"])
    assert score.title == 'C:\\songs "x"'
    assert score.bars[0].section == 'Outro "end"\\'


def test_tied_item_continues_only_its_notes():
    # 和弦里只有低音接着响：先写整个和弦，再用连音线只延续 4 弦
    items = [atex.Item(0, 1, [(2, 0), (4, 1)]), atex.Item(1, 1, [(4, 1)], tie=True), atex.Item(2, 2, [])]
    assert atex.voice_bar_tokens(items, "straight") == ["(0.2 1.4).4", "-.4.4", "r.2"]


def test_tied_item_in_shuffle_is_split_but_not_an_onset(tmp_path):
    # 反拍上的连音线延续没有起音：不触发八分对切分；被别的反拍起音切开时，各片段仍是连音线
    items = [atex.Item(0, 1.5, [(2, 0), (4, 1)]), atex.Item(1.5, 2.5, [(4, 1)], tie=True)]
    assert atex.voice_bar_tokens(items, "shuffle") == ["(0.2 1.4).4 {d}", "-.4.2", "-.4.8"]
    items = [atex.Item(0, 1.5, [(2, 0), (4, 1)]), atex.Item(1.5, 1.0, [(4, 1)], tie=True), atex.Item(2.5, 1.5, [(1, 3)])]
    tokens = atex.voice_bar_tokens(items, "shuffle")
    assert tokens == ["(0.2 1.4).4 {d}", "-.4.8", "-.4.8", "3.1.8", "-.1.4"]
    head = atex.header("T", 120, 0, "shuffle", "tabs")
    src = tmp_path / "t.atex"
    src.write_text(atex.document(head, [[atex.bar_line(tokens, 1)]]), encoding="utf-8")
    score = load_score(alphatab.build(src, tmp_path / "b", "t")["json"])
    from guitarize.config import load_profile
    from guitarize.verify.playability import check_playability
    assert check_playability(score, load_profile("intermediate"), "finger") == []
    assert [round(a.start, 3) for a in score.attacks()] == [0.0, 2.667]
