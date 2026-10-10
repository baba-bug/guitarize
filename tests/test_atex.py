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


def test_held_notes_ring_under_fresh_notes_in_one_beat(tmp_path):
    # 低音（6 弦）接着响，同一拍新拨两个合音：和弦里低音写成连音线、合音照常起音；拆开的后续片段都是连音线
    assert atex.notes_token([(2, 0), (3, 0), (6, 3)], held={6}) == "(0.2 0.3 -.6)"
    items = [atex.Item(0, 1, [(6, 3)]), atex.Item(1, 1 / 3, [(2, 0), (3, 0), (6, 3)], held=frozenset({6})),
             atex.Item(4 / 3, 2 / 3, [(6, 3)], tie=True), atex.Item(2, 2, [])]
    tokens = atex.voice_bar_tokens(items, "shuffle")
    assert tokens == ["3.6.4", "(0.2 0.3 -.6).8 {tu 3}", "-.6.4 {tu 3}", "r.2"]
    long = [atex.Item(0, 1, [(6, 3)]), atex.Item(1, 1.25, [(2, 0), (6, 3)], held=frozenset({6})),
            atex.Item(2.25, 1.75, [])]
    assert atex.voice_bar_tokens(long, "straight")[:3] == ["3.6.4", "(0.2 -.6).4", "(-.2 -.6).16"]
    head = atex.header("T", 120, 0, "shuffle", "tabs")
    src = tmp_path / "h.atex"
    src.write_text(atex.document(head, [[atex.bar_line(tokens, 1)]]), encoding="utf-8")
    score = load_score(alphatab.build(src, tmp_path / "b", "h")["json"])
    assert [(round(s.start, 3), round(s.end, 3), s.midi) for s in score.sounding()] == [
        (0.0, 2.0, 43), (1.0, 1.333, 59), (1.0, 1.333, 55)]
    assert [(round(a.start, 3), len(a.notes)) for a in score.attacks()] == [(0.0, 1), (1.0, 2)]


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


def test_rest_cuts_only_at_the_beat_lines_next_to_its_offbeat_ends():
    # 从反拍开始：本拍余下的部分单独切出；结束在反拍：最后一条拍线处切开；中间的整拍不切
    assert atex.rest_cuts(0.5, 3.0) == [1.0]
    assert atex.rest_cuts(1.0, 2.5) == [2.0]
    assert atex.rest_cuts(0.5, 1.5) == [1.0]
    assert atex.rest_cuts(1.5, 3.5) == [2.0, 3.0]
    assert atex.rest_cuts(0.0, 4.0) == [] and atex.rest_cuts(2.0, 3.0) == []
    assert atex.rest_cuts(0.5, 1.0) == [] and atex.rest_cuts(2.0, 2.5) == []   # 不跨拍线


def test_shuffle_rest_after_a_beat_eighth_is_split_at_the_beat_line():
    # 拍点八分 + 从反拍开始、跨过拍线的休止：休止先写本拍余下的八分（与前面的音凑成八分对），再写整拍
    items = [atex.Item(0, 0.5, [(1, 0)]), atex.Item(0.5, 2.5, [], ['txt "A"']), atex.Item(3, 1, [(1, 3)])]
    assert atex.voice_bar_tokens(items, "shuffle") == ["0.1.8", 'r.8 {txt "A"}', "r.2", "3.1.4"]
    assert atex.voice_bar_tokens(items, "straight") == ["0.1.8", 'r.2 {txt "A"}', "r.8", "3.1.4"]   # 直拍不变
    # 连音线延续的拍点八分同样成对；音和连音线延续照旧（附点四分音不切：只改休止）
    items = [atex.Item(0, 1, [(2, 0)]), atex.Item(1, 0.5, [(2, 0)], tie=True), atex.Item(1.5, 1.5, []),
             atex.Item(3, 1, [(1, 3)])]
    assert atex.voice_bar_tokens(items, "shuffle") == ["0.2.4", "-.2.8", "r.8", "r.4", "3.1.4"]
    assert atex.voice_bar_tokens(items, "straight") == ["0.2.4", "-.2.8", "r.4 {d}", "3.1.4"]
    items = [atex.Item(0, 1.5, [(2, 0)]), atex.Item(1.5, 1.5, []), atex.Item(3, 1, [(1, 3)])]
    assert atex.voice_bar_tokens(items, "shuffle") == ["0.2.4 {d}", "r.8", "r.4", "3.1.4"]
    # 从拍点开始的整拍休止照旧合并；从拍点开始、结束在反拍的休止在最后一条拍线处切开
    items = [atex.Item(0, 1, [(1, 0)]), atex.Item(1, 3, [])]
    assert atex.voice_bar_tokens(items, "shuffle") == ["0.1.4", "r.2 {d}"]
    pieces = atex.split_for_swing([atex.Item(0, 1, [(1, 0)]), atex.Item(1, 1.5, []),
                                   atex.Item(2.5, 1.5, [(1, 0)], tie=True)], "shuffle")
    assert [(p.start, p.length, cont) for p, cont in pieces] == [
        (0, 1, False), (1, 1.0, False), (2.0, 0.5, True), (2.5, 1.5, False)]


def test_shuffle_beat_eighth_before_an_offbeat_rest_plays_swung(tmp_path):
    # 用户听感：拍点八分后面跟着跨拍的反拍休止时，alphaTab 把这个八分按直拍（半拍）播放，旋律节奏发飘。
    # 切开之后它与休止的前一个八分成对：演奏时值 2/3 拍，休止从 +2/3 开始
    bar1 = [atex.Item(0, 0.5, [(1, 0)]), atex.Item(0.5, 2.5, []), atex.Item(3, 1, [(1, 3)])]
    bar2 = [atex.Item(0, 1, [(2, 0)]), atex.Item(1, 0.5, [(2, 0)], tie=True), atex.Item(1.5, 1.5, []),
            atex.Item(3, 0.5, [(1, 3)]), atex.Item(3.5, 0.5, [(1, 5)])]
    head = atex.header("T", 120, 0, "shuffle", "tabs")
    lines = [atex.bar_line(atex.voice_bar_tokens(items, "shuffle"), i + 1) for i, items in enumerate([bar1, bar2])]
    src = tmp_path / "s.atex"
    src.write_text(atex.document(head, [lines]), encoding="utf-8")
    score = load_score(alphatab.build(src, tmp_path / "b", "s")["json"])
    first = score.events[0]
    assert first.start == 0 and first.written_dur == 0.5 and first.dur == pytest.approx(2 / 3)
    rest = score.events[1]
    assert rest.rest and rest.start == pytest.approx(2 / 3) and rest.dur == pytest.approx(1 / 3)
    held = [s for s in score.sounding() if 4 <= s.start < 5]                 # 0.2 连音线延续到第 2 拍的拍点八分
    assert len(held) == 1 and held[0].end == pytest.approx(4 + 1 + 2 / 3)
    assert [round(a.start, 4) for a in score.attacks()] == [0.0, 3.0, 4.0, 7.0, round(7 + 2 / 3, 4)]
    from guitarize.config import load_profile
    from guitarize.verify.playability import check_playability
    assert check_playability(score, load_profile("intermediate"), "finger") == []   # 含 straight_offbeat


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


# ---------- 音符效果（击勾弦 h、连奏滑音 sl、滑出 sod / sou）：挂在音尾，即这个音最后一个记号上 ----------

def test_note_effects_follow_the_note_inside_and_outside_chords():
    # alphaTab 1.8.4：音符效果紧跟在 品.弦 后面、时值前面（'5.3{h}.4'）；和弦里跟在各自的音后面；
    # 拍效果仍写在时值后面（'5.3{h}.4 {d}'）。'5.3.4{h}' 会被当成拍效果，报 Unrecognized property
    assert atex.notes_token([(3, 5)], effects={3: ["h"]}) == "5.3{h}"
    assert atex.notes_token([(3, 5), (2, 3)], effects={3: ["h"]}) == "(5.3{h} 3.2)"
    assert atex.notes_token([(3, 5)], tie=True, effects={3: ["sl"]}) == "-.3{sl}"
    assert atex.notes_token([(3, 5)], effects={2: ["h"]}) == "5.3"                  # 别的弦的效果不挂
    assert atex.beat_tokens([(3, 5)], 1.5, note_effects={3: ["h"]}) == ["5.3{h}.4 {d}"]
    # 拆成几个记号时：拍效果在第一个，音符效果在最后一个（连音线延续的那一段，紧挨着下一个音）
    assert atex.beat_tokens([(3, 5)], 2.5, ['ch "G"'], note_effects={3: ["h"]}) == ['5.3.2 {ch "G"}', "-.3{h}.8"]


def _legato_score(tmp_path, tokens, name="t"):
    head = atex.header("T", 120, 0, "shuffle", "tabs")
    src = tmp_path / f"{name}.atex"
    src.write_text(atex.document(head, [[atex.bar_line(tokens, 1)]]), encoding="utf-8")
    return load_score(alphatab.build(src, tmp_path / "b", name)["json"])


def test_swing_split_keeps_the_note_effect_on_the_last_piece_only(tmp_path):
    # Shuffle：3 弦 5 品从拍点按到反拍，接着反拍上击弦到 7 品。八分对切分把 5 品切成 4 分 + 连音线八分：
    # h 只在连音线那一段（紧挨着 7 品），不丢也不重复
    items = [atex.Item(0, 1.5, [(3, 5)], note_effects={3: ["h"]}), atex.Item(1.5, 0.5, [(3, 7)]), atex.Item(2, 2, [])]
    pieces = atex.split_for_swing(items, "shuffle")
    assert [dict(p.note_effects or {}) for p, _ in pieces] == [{}, {3: ["h"]}, {}, {}]
    tokens = atex.voice_bar_tokens(items, "shuffle")
    assert tokens == ["5.3.4", "-.3{h}.8", "7.3.8", "r.2"]
    score = _legato_score(tmp_path, tokens)
    notes = [(round(e.written_start, 4), n.fret, n.tie, n.hammer) for e in score.events for n in e.notes]
    assert notes == [(0.0, 5, False, False), (1.0, 5, True, True), (1.5, 7, False, False)]
    assert [round(a.start, 3) for a in score.attacks()] == [0.0, 1.667]          # 击弦的音照样按摇摆演奏


def test_triplet_split_keeps_the_note_effect_on_the_last_piece_only(tmp_path):
    # 跨进三连音拍的音在拍线处切开：sl 挂在拍内那一段（三连音的连音线延续）上
    items = [atex.Item(0, 4 / 3, [(3, 5)], note_effects={3: ["sl"]}), atex.Item(4 / 3, 2 / 3, [(3, 9)]),
             atex.Item(2, 2, [])]
    out = atex.split_triplets(items)
    assert [dict(p.note_effects or {}) for p in out] == [{}, {3: ["sl"]}, {}, {}]
    tokens = atex.voice_bar_tokens(items, "shuffle")
    assert tokens == ["5.3.4", "-.3{sl}.8 {tu 3}", "9.3.4 {tu 3}", "r.2"]
    score = _legato_score(tmp_path, tokens)
    notes = [(round(e.written_start, 4), n.fret, n.slide) for e in score.events for n in e.notes]
    assert notes == [(0.0, 5, None), (1.0, 5, "legato"), (1.3333, 9, None)]


def test_slide_out_and_chord_note_effects_build(tmp_path):
    items = [atex.Item(0, 1, [(3, 5), (2, 3)], note_effects={3: ["h"]}), atex.Item(1, 1, [(3, 7), (2, 3)]),
             atex.Item(2, 1, [(1, 3)], note_effects={1: ["sod"]}), atex.Item(3, 1, [(1, 5)], note_effects={1: ["sou"]})]
    tokens = atex.voice_bar_tokens(items, "straight")
    assert tokens == ["(5.3{h} 3.2).4", "(7.3 3.2).4", "3.1{sod}.4", "5.1{sou}.4"]
    score = _legato_score(tmp_path, tokens)
    assert [[(n.fret, n.hammer, n.slide) for n in e.notes] for e in score.events] == [
        [(5, True, None), (3, False, None)], [(7, False, None), (3, False, None)], [(3, False, "out_down")],
        [(5, False, "out_up")]]
