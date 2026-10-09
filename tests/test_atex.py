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
