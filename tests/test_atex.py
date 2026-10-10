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


def test_key_signature_names_follow_convention():
    # 只用升号的调名要换成惯用的降号调号（alphaTab 不认 A#、D# 大调这类名字）
    assert atex.key_signature("B", "major") == "B"
    assert atex.key_signature("A#", "major") == "Bb"
    assert atex.key_signature("D#", "major") == "Eb"
    assert atex.key_signature("F#", "major") == "F#"
    assert atex.key_signature("G#", "minor") == "G#minor"
    assert atex.key_signature("A#", "minor") == "Bbminor"
    assert atex.key_signature("D#", "minor") == "D#minor"
    assert atex.key_signature("Bb", "major") == "Bb"


def test_header_key_signature_builds(tmp_path):
    for tonic, mode in [("B", "major"), ("A#", "major"), ("D#", "major"), ("A#", "minor"), ("C", "minor")]:
        head = atex.header("T", 120, 4, "shuffle", "score tabs", key=(tonic, mode))
        assert f"\\ks {atex.key_signature(tonic, mode)}" in head
        src = tmp_path / f"k-{tonic}-{mode}.atex"
        src.write_text(atex.document(head, [[atex.bar_line(["0.1.1"], 1)]]), encoding="utf-8")
        alphatab.build(src, tmp_path / "b", src.stem)  # 调号名不被 alphaTab 接受时这里会抛 AlphaTexError
    assert "\\ks" not in atex.header("T", 120, 0, "straight", "tabs")


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


def test_split_triplet():
    # 三连音拍里的时值：1/3 → 三连音八分、2/3 → 三连音四分、整拍 → 普通四分（不标 tu）
    assert atex.split_triplet(1 / 3) == [(8, True)]
    assert atex.split_triplet(2 / 3) == [(4, True)]
    assert atex.split_triplet(1.0) == [(4, False)]
    with pytest.raises(ValueError):
        atex.split_triplet(0.5)


def test_triplet_beat_tokens():
    assert atex.beat_tokens([(3, 4)], 1 / 3, tuplet=True) == ["4.3.8 {tu 3}"]
    assert atex.beat_tokens([(2, 7)], 2 / 3, ['ch "G"', "dy f"], tuplet=True) == ['7.2.4 {ch "G" dy f tu 3}']
    assert atex.beat_tokens([(2, 7)], 2 / 3, tie=True, tuplet=True) == ["-.2.4 {tu 3}"]
    assert atex.beat_tokens([], 1 / 3, tuplet=True) == ["r.8 {tu 3}"]


def test_beat_with_a_third_is_written_as_triplets(tmp_path):
    # B3 三连音八分 + F#4 三连音四分、F#4 连音线延续到第 3 拍；三连音拍的前后照旧是八分对 / 普通时值
    items = [atex.Item(0, 0.5, [(1, 0)], ['ch "G"']), atex.Item(0.5, 0.5, [(1, 2)]),
             atex.Item(1, 1 / 3, [(3, 4)]), atex.Item(1 + 1 / 3, 5 / 3, [(2, 7)], ['ch "C"']), atex.Item(3, 1, [])]
    tokens = atex.voice_bar_tokens(items, "shuffle")
    assert tokens == ['0.1.8 {ch "G"}', "2.1.8", "4.3.8 {tu 3}", '7.2.4 {ch "C" tu 3}', "-.2.4", "r.4"]
    head = atex.header("T", 167, 0, "shuffle", "score tabs", {"G": (3, 0, 0, 0, 2, 3), "C": (0, 1, 0, 2, 3, None)})
    src = tmp_path / "t.atex"
    src.write_text(atex.document(head, [[atex.bar_line(tokens, 1)]]), encoding="utf-8")
    score = load_score(alphatab.build(src, tmp_path / "b", "t")["json"])
    assert [round(a.start, 4) for a in score.attacks()] == [0.0, round(2 / 3, 4), 1.0, round(4 / 3, 4)]
    assert [e.tuplet for e in score.events] == [1, 1, 3, 3, 1, 1]
    assert [e.chord for e in score.events if e.chord] == ["G", "C"]
    held = [s for s in score.sounding() if s.start > 1.1]                 # F#4（2 弦 7 品）响到第 3 拍末
    assert len(held) == 1 and held[0].start == pytest.approx(4 / 3) and held[0].end == pytest.approx(3.0)
    from guitarize.config import load_profile
    from guitarize.verify.playability import check_playability
    assert check_playability(score, load_profile("intermediate"), "finger") == []


def test_split_triplets_marks_and_cuts_only_triplet_beats():
    # 跨进三连音拍的音在拍线处切开：拍内的片段标 tuplet，切出的后续片段是连音线延续（休止则是休止）
    items = [atex.Item(0, 1 + 1 / 3, [(1, 3)], ["sd"]), atex.Item(1 + 1 / 3, 2 / 3, []), atex.Item(2, 2, [])]
    out = atex.split_triplets(items)
    assert [(round(p.start, 4), round(p.length, 4), p.tie, p.tuplet, list(p.effects)) for p in out] == [
        (0, 1, False, False, ["sd"]), (1, 0.3333, True, True, []), (1.3333, 0.6667, False, True, []),
        (2, 2, False, False, [])]
    assert out[1].notes == [(1, 3)]
    assert atex.split_triplets([atex.Item(0, 4, [])]) == [atex.Item(0, 4, [])]   # 没有三连音拍：原样


def test_split_for_swing_does_not_cut_or_pair_tuplets(tmp_path):
    # 三连音拍里的片段不切、也不当作反拍起音；前一拍的八分对照旧
    items = [atex.Item(0, 0.5, []), atex.Item(0.5, 0.5, [(1, 3)]),
             atex.Item(1, 1 / 3, [(1, 5)], tuplet=True), atex.Item(1 + 1 / 3, 2 / 3, [(1, 3)], tuplet=True),
             atex.Item(2, 2, [])]
    pieces = atex.split_for_swing(items, "shuffle")
    assert [(p.start, p.length, p.tuplet) for p, _ in pieces] == [(it.start, it.length, it.tuplet) for it in items]
    assert not any(cont for _, cont in pieces)
    tokens = atex.voice_bar_tokens(items, "shuffle")
    assert tokens == ["r.8", "3.1.8", "5.1.8 {tu 3}", "3.1.4 {tu 3}", "r.2"]
    # 反拍起音一直按到三连音拍里：在拍线处切开，前半段与前一个八分成对，后半段写成三连音连音线延续
    items = [atex.Item(0, 0.5, []), atex.Item(0.5, 5 / 6, [(1, 3)]), atex.Item(1 + 1 / 3, 2 / 3, [(1, 5)]),
             atex.Item(2, 2, [])]
    tokens = atex.voice_bar_tokens(items, "shuffle")
    assert tokens == ["r.8", "3.1.8", "-.1.8 {tu 3}", "5.1.4 {tu 3}", "r.2"]
    head = atex.header("T", 120, 0, "shuffle", "tabs")
    src = tmp_path / "t.atex"
    src.write_text(atex.document(head, [[atex.bar_line(tokens, 1)]]), encoding="utf-8")
    score = load_score(alphatab.build(src, tmp_path / "b", "t")["json"])
    assert [round(a.start, 4) for a in score.attacks()] == [round(2 / 3, 4), round(4 / 3, 4)]
    snd = score.sounding()
    assert snd[0].start == pytest.approx(2 / 3) and snd[0].end == pytest.approx(4 / 3)


def test_straight_items_are_untouched_by_triplet_splitting():
    items = [atex.Item(0, 1, [(1, 3)]), atex.Item(1, 0.5, []), atex.Item(1.5, 2.5, [(1, 5)], ["sd"])]
    assert atex.split_triplets(items) == items
    assert atex.voice_bar_tokens(items, "straight") == ["3.1.4", "r.8", "5.1.2 {sd}", "-.1.8"]


def test_chord_diagrams_drawn_above_the_beats(tmp_path):
    # 用户要求：每小节标的指型旁边直接画出和弦图（不只是谱首的列表）
    head = atex.header("T", 120, 4, "shuffle", "score tabs", {"G": (3, 0, 0, 0, 2, 3)}, key=("B", "major"))
    assert "\\chordDiagramsInScore" in head
    src = tmp_path / "d.atex"
    src.write_text(atex.document(head, [[atex.bar_line(atex.beat_tokens([(1, 3)], 4, ['ch "G"']), 1)]]), encoding="utf-8")
    alphatab.build(src, tmp_path / "b", "d")              # alphaTab 认这个元数据，构建不报错
    assert "\\chordDiagramsInScore" not in atex.header("T", 120, 0, "straight", "tabs")   # 没有和弦图时不加
