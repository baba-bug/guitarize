"""旋律谱（arr = melody）：leadsheet 的每一个旋律音、按原唱的节奏（Shuffle 三连音位置精确）、单声部、可弹。"""
import re

import pytest

from guitarize.arrange import melody as M
from guitarize.arrange.melody import generate_melody
from guitarize.arrange.pipeline import self_check
from guitarize.config import Thresholds, load_profile
from guitarize.render import alphatab
from guitarize.verify import metrics
from guitarize.verify.model import load_score
from guitarize.verify.playability import Violation, check_playability

MID = load_profile("intermediate")
T3 = 1 / 3


def _bar(i, mel, chord="N", section=None):
    return {"bar": i + 1, "section": section, "t_start": i * 1.44, "t_end": (i + 1) * 1.44,
            "chords": [{"beat": 1, "dur_beats": 4, "label": chord, "confidence": "high"}],
            "melody": [{"start_beats": round(i * 4 + o, 4), "dur_beats": round(d, 4), "midi": m, "conf": 0.9,
                        "out_of_key": False} for o, d, m in mel]}


def _sheet(bars, feel="shuffle", tempo=167.0):
    return {"key": "B", "mode": "major", "tempo_bpm": tempo, "feel": feel, "beats_per_bar": 4, "bars": bars}


def _build(text, tmp_path, name="m"):
    src = tmp_path / f"{name}.atex"
    src.write_text(text, encoding="utf-8")
    return load_score(alphatab.build(src, tmp_path / "b", name)["json"])


def _voice_bar(text, n):
    return text.split(f"// bar {n}")[1].split("\n", 1)[1].splitlines()[0]


def _rhythm(line):
    """一行拍子记号 → 节奏：起音 n、连音线延续 -、休止 r；只留时值码和 d / tu 3（和弦 / 力度标记去掉）。"""
    line = re.sub(r'(?:ch "[^"]*"|dy \w+)\s*', "", line)
    line = re.sub(r" \{\s*\}", "", line).replace(" }", "}")
    tokens = re.findall(r"(\([^)]*\)|[^\s(]+?)\.(\d+)( \{[^}]*\})?(?= |$)", line.split("\\section")[-1].rstrip(" |"))
    return [("r" if n == "r" else "-" if n.startswith("-") else "n") + f".{c}{e}" for n, c, e in tokens
            if not n.startswith('"')]


# なんもねえ 第 1–2 小节（进副歌的拾音 + 副歌第一句），Shuffle 167
BAR1 = [(1 + 2 * T3, T3, 66), (2, 1, 66), (3, T3, 59), (3 + T3, 4 * T3, 66)]
BAR2 = [(2, T3, 64), (2 + T3, 2 * T3, 63), (3, 2 * T3, 66), (3 + 2 * T3, T3, 66)]


def _onsets(score):
    return [(round(s.start, 4), s.midi) for s in score.melody(voice=0)]


def test_keeps_every_note_at_its_sung_position(tmp_path):
    # 骨架会丢掉的短音（B3 1/3 拍、+⅓ 的 D#4）、一拍三个音都保留，起音与原唱一致（Shuffle 的 +⅓ / +⅔ 精确）
    third_beat = [(0, T3, 61), (T3, T3, 63), (2 * T3, T3, 64), (1, 1, 66), (2, 2, 68)]
    ls = _sheet([_bar(0, BAR1), _bar(1, BAR2), _bar(2, third_beat)])
    text = generate_melody(ls, 0, MID, "T")
    score = _build(text, tmp_path)
    want = [(round(m["start_beats"], 4), m["midi"]) for b in ls["bars"] for m in b["melody"]]
    assert _onsets(score) == want
    assert check_playability(score, MID, "melody") == []
    bars = metrics.score_bars(score, ls, "melody")
    s = metrics.summarize(bars, [], Thresholds(), "melody")
    assert s["melody_f1"] == pytest.approx(1.0) and s["melody_recall"] == pytest.approx(1.0)


def test_shuffle_beats_with_a_third_are_triplets_others_swing_pairs():
    ls = _sheet([_bar(0, BAR1), _bar(1, BAR2)])
    text = generate_melody(ls, 0, MID, "T")
    # 第 1 小节：2+⅔ 的 F#4 是摇摆八分对的后一个（记在 2.5），第 4 拍有 +⅓ 起音 → 三连音；F#4 连到下一小节
    assert _rhythm(_voice_bar(text, 1)) == ["r.4", "r.8", "n.8", "n.4", "n.8 {tu 3}", "n.4 {tu 3}"]
    # 第 2 小节：连音线延续到 +⅔（摇摆八分），第 3 拍 +⅓ → 三连音，第 4 拍 4 / 4+⅔ → 摇摆八分对
    bar2 = _rhythm(_voice_bar(text, 2))
    assert bar2[0] == "-.8" and bar2[-4:] == ["n.8 {tu 3}", "n.4 {tu 3}", "n.8", "n.8"]
    assert "tu 3" not in " ".join(bar2[:-4])


def test_durations_as_sung_and_rests_stay_rests(tmp_path):
    # 直拍：四分音符后空一拍（休止），十六分音符照唱的时值写；不把音延长到下一个音
    ls = _sheet([_bar(0, [(0, 1, 64), (2, 0.25, 66), (2.5, 0.75, 67), (3.25, 0.25, 69)])], feel="straight", tempo=100)
    text = generate_melody(ls, 0, MID, "T")
    assert _rhythm(_voice_bar(text, 1)) == ["n.4", "r.4", "n.16", "r.16", "n.8 {d}", "n.16", "r.8"]
    assert "tu 3" not in text
    score = _build(text, tmp_path)
    sounding = [(round(s.start, 4), round(s.end - s.start, 4), s.midi) for s in score.sounding()]
    assert sounding == [(0.0, 1.0, 64), (2.0, 0.25, 66), (2.5, 0.75, 67), (3.25, 0.25, 69)]


def test_shuffle_short_note_then_rest_is_a_swung_eighth():
    # 拍点上 1/3 拍的音后面是休止：不为它单写三连音，写成摇摆八分（音尾落到反拍八分）+ 八分休止
    ls = _sheet([_bar(0, [(0, T3, 64), (1, 1, 66), (2, 2, 68)])])
    text = generate_melody(ls, 0, MID, "T")
    assert _rhythm(_voice_bar(text, 1)) == ["n.8", "r.8", "n.4", "n.2"]


def test_held_note_ties_across_the_bar_line_and_keeps_the_chord_label(tmp_path):
    ls = _sheet([_bar(0, [(0, 1, 64), (3, 2, 66)], chord="C"), _bar(1, [(2, 2, 68)], chord="G")], feel="straight")
    text = generate_melody(ls, 0, MID, "T")
    bar2 = _voice_bar(text, 2)
    assert _rhythm(bar2)[0] == "-.4"
    assert re.match(r'-\.\d\.4 \{ch "G"\}', bar2)                     # 指型名挂在连音线延续那一拍上
    score = _build(text, tmp_path)
    assert [(round(s.start, 4), round(s.end, 4), s.midi) for s in score.sounding()] == [
        (0.0, 1.0, 64), (3.0, 5.0, 66), (6.0, 8.0, 68)]
    assert [e.chord for e in score.events if e.chord] == ["C", "G"]


def test_single_voice_display_and_a_chord_shape_on_every_bar(tmp_path):
    mel = [(0, 1, 64), (1, 1, 66), (2, 2, 68)]
    ls = _sheet([_bar(0, mel, "B", "サビ"), _bar(1, mel, "B"), _bar(2, mel, "E", "Aメロ")], feel="straight")
    text = generate_melody(ls, 4, MID, "T")
    assert "\\voice" not in text and "\\staff {score tabs}" in text and "\\ks B" in text
    assert "\\capo 4" in text and "dy f" in text and "// bar 1 [サビ]" in text and "// bar 3 [Aメロ]" in text
    for n, shape in ((1, "G"), (2, "G"), (3, "C")):                  # 实际 B / E 在 capo 4 下是指法调 G / C
        assert f'ch "{shape}"' in _voice_bar(text, n)
    assert '\\chord "G"' in text and '\\chord "C"' in text
    # 段落名写成第一拍的文字（txt），不用 \\section：alphaTab 把段落标记和同一拍的和弦名画在同一行、叠在一起
    assert "\\section" not in text
    assert re.match(r'\d+\.\d\.4 \{ch "G" txt "サビ" dy f\}', _voice_bar(text, 1))
    assert re.match(r'\d+\.\d\.4 \{ch "C" txt "Aメロ"\}', _voice_bar(text, 3))
    score = _build(text, tmp_path)
    first = [next(e for e in score.events if e.bar == i and not e.rest) for i in range(3)]
    assert [(e.chord, e.text) for e in first] == [("G", "サビ"), ("G", None), ("C", "Aメロ")]
    from guitarize.render.ascii_tab import render_ascii
    ascii_text = render_ascii(score)
    assert "1 [サビ]" in ascii_text and "3 [Aメロ]" in ascii_text   # 文本六线谱照样标段落


def test_chord_change_inside_a_held_note_splits_it_with_a_tie():
    # 第 3 拍换和弦时旋律长音还在响：在第 3 拍切开、后半写成连音线，指型名挂在那里
    ls = _sheet([{**_bar(0, [(0, 4, 64)]), "chords": [{"beat": 1, "dur_beats": 2, "label": "C", "confidence": "high"},
                                                     {"beat": 3, "dur_beats": 2, "label": "G", "confidence": "high"}]}],
                feel="straight")
    text = generate_melody(ls, 0, MID, "T")
    line = _voice_bar(text, 1)
    assert _rhythm(line) == ["n.2", "-.2"] and re.search(r'-\.\d\.2 \{ch "G"\}', line)


def test_singer_octave_policy_is_shared_with_fingerstyle():
    # 太低的旋律按 octave_plan 整曲升八度（与指弹同一策略）；overrides.melody_octave 的整数照样生效
    low = [(0, 1, 43), (1, 1, 45), (2, 1, 47), (3, 1, 48)]
    ls = _sheet([_bar(0, low)], feel="straight")
    line, _ = M.melody_line(ls, 0)
    assert [n.pitch for n in line] == [55, 57, 59, 60]
    line, _ = M.melody_line(ls, 0, octave=24)
    assert [n.pitch for n in line] == [67, 69, 71, 72]


def test_leap_heavy_line_has_no_violations_and_keeps_every_pitch(tmp_path):
    # 一个八度以上的大跳、三连音间隔（167 bpm），只靠把位选择（不移八度、不丢音）就要 0 违规
    leaps = [52, 74, 55, 76, 50, 72, 53, 71, 48, 69, 57, 76]
    mel = [(k * T3, T3, p) for k, p in enumerate(leaps)]
    ls = _sheet([_bar(0, mel[:12])])
    calls = []

    def check(text):
        calls.append(self_check(text, MID, "melody"))
        return calls[-1]

    text = generate_melody(ls, 0, MID, "T", check=check)
    score = _build(text, tmp_path)
    assert check_playability(score, MID, "melody") == [] and calls[-1] == []
    assert [p for _, p in _onsets(score)] == leaps
    assert "melody+12" not in text and "melody-12" not in text


def test_prefers_the_top_four_strings_and_low_positions(tmp_path):
    # D3–A3（指法调）都能在 4 弦 / 3 弦低把位弹到，不去 5 弦
    mel = [(0, 1, 50), (1, 1, 52), (2, 1, 53), (3, 1, 55), (4, 1, 57)]
    ls = _sheet([_bar(0, mel[:4]), _bar(1, mel[4:])], feel="straight")
    score = _build(generate_melody(ls, 0, MID, "T", octave=0), tmp_path)
    notes = [n for e in score.events if not e.rest for n in e.notes]
    assert all(n.string <= 4 for n in notes) and max(n.fret for n in notes) <= 3


def test_shift_repair_refrets_the_offending_note_keeping_its_pitch():
    # 自检报第 2 拍的音换把：只换弦 / 品，音高不变；修补写进小节注释
    ls = _sheet([_bar(0, [(0, 1, 64), (1, 1, 67), (2, 1, 69), (3, 1, 71)])], feel="straight")
    plain = generate_melody(ls, 0, MID, "T")
    calls = []

    def check(text):
        calls.append(text)
        return [Violation(1, 1.0, "shift", "x")] if len(calls) == 1 else []

    text = generate_melody(ls, 0, MID, "T", check=check)
    assert len(calls) == 2 and calls[0] == plain and calls[-1] == text

    def frets(t):
        return re.findall(r"(\d+)\.(\d)\.\d", _voice_bar(t, 1))

    before, after = frets(plain), frets(text)
    assert before[1] != after[1] and [x for i, x in enumerate(before) if i != 1] == [x for i, x in enumerate(after) if i != 1]
    from guitarize.arrange.shapes import OPEN_STRINGS
    assert {OPEN_STRINGS[int(s) - 1] + int(f) for f, s in (before[1], after[1])} == {67}
    assert re.search(r"// bar 1 · refret @2 \d+\.\d→\d+\.\d \(shift\)", text)


def test_straight_triplet_from_a_manual_note_is_written_as_a_triplet_beat(tmp_path):
    # 直拍里 overrides.melody 写的 ⅓ 拍位置：那一拍整拍写成三连音，别的拍还是十六分格
    ls = _sheet([_bar(0, [(0, T3, 64), (T3, T3, 66), (2 * T3, T3, 67), (1, 0.75, 69), (1.75, 0.25, 67), (2, 2, 64)])],
                feel="straight")
    text = generate_melody(ls, 0, MID, "T")
    assert _rhythm(_voice_bar(text, 1)) == ["n.8 {tu 3}", "n.8 {tu 3}", "n.8 {tu 3}", "n.8 {d}", "n.16", "n.2"]
    score = _build(text, tmp_path)
    assert [round(s, 4) for s, _ in _onsets(score)] == [0.0, 0.3333, 0.6667, 1.0, 1.75, 2.0]


# ---------- 连奏（legato）与尾滑（slide_out）：同一音节里滑过去的音不重新拨弦 ----------

def _flagged(i, mel, chord="N"):
    """mel：[(拍内偏移, 时值, 音高, 标记)]，标记如 {"legato": True} / {"slide_out": "down"}。"""
    bar = _bar(i, [(o, d, m) for o, d, m, _ in mel], chord)
    for note, (_, _, _, extra) in zip(bar["melody"], mel):
        note.update(extra)
    return bar


L = {"legato": True}


def _strip(ls):
    """去掉 legato / slide_out：旧 leadsheet 的样子。"""
    return {**ls, "bars": [{**b, "melody": [{k: v for k, v in m.items() if k not in ("legato", "slide_out")}
                                            for m in b["melody"]]} for b in ls["bars"]]}


def _built_effects(score):
    """谱面模型里每个音：(记谱位置, 弦, 品, 连音线, 击勾弦起点, 滑音)。"""
    return [(round(e.written_start, 4), n.string, n.fret, n.tie, n.hammer, n.slide)
            for e in score.events for n in e.notes]


def test_two_semitone_legato_stays_on_one_string_as_a_hammer_on(tmp_path):
    # D4 → E4（2 个半音）：平时 E4 弹 1 弦空弦；连奏时留在 2 弦 3 → 5 品，D4 上写 h（击弦）
    ls = _sheet([_flagged(0, [(0, 1, 62, {}), (1, 1, 64, L)])], feel="straight", tempo=100)
    plain = generate_melody(_strip(ls), 0, MID, "T", octave=0)
    assert _voice_bar(plain, 1) == "3.2.4 {dy f} 0.1.4 r.2 |"
    calls = []

    def check(text):
        calls.append(self_check(text, MID, "melody"))
        return calls[-1]

    text = generate_melody(ls, 0, MID, "T", check=check, octave=0)
    assert _voice_bar(text, 1) == "3.2{h}.4 {dy f} 5.2.4 r.2 |" and calls[-1] == []
    score = _build(text, tmp_path)
    assert _built_effects(score) == [(0.0, 2, 3, False, True, None), (1.0, 2, 5, False, False, None)]
    assert check_playability(score, MID, "melody") == []
    s = metrics.summarize(metrics.score_bars(score, ls, "melody"), [], Thresholds(), "melody")
    assert s["melody_f1"] == pytest.approx(1.0)                      # 击弦的音照样是旋律音


def test_five_semitone_legato_is_a_slide(tmp_path):
    # A3 唱 2 拍后滑到 D4（5 个半音）：3 弦 2 → 7 品，A3 上写 sl（连奏滑音，alphaTab 不为 D4 重新起音）
    ls = _sheet([_flagged(0, [(0, 2, 57, {}), (2, 2, 62, L)])], feel="straight", tempo=100)
    assert _voice_bar(generate_melody(_strip(ls), 0, MID, "T", octave=0), 1) == "2.3.2 {dy f} 3.2.2 |"
    text = generate_melody(ls, 0, MID, "T", octave=0)
    assert _voice_bar(text, 1) == "2.3{sl}.2 {dy f} 7.3.2 |"
    score = _build(text, tmp_path)
    assert _built_effects(score) == [(0.0, 3, 2, False, False, "legato"), (2.0, 3, 7, False, False, None)]
    assert check_playability(score, MID, "melody") == []


def test_legato_to_or_from_an_open_string_is_a_hammer_or_pull(tmp_path):
    # 空弦滑不过去：3 弦 4 品 → 空弦（4 个半音）写成勾弦 h
    ls = _sheet([_flagged(0, [(0, 2, 59, {}), (2, 2, 55, L)])], feel="straight", tempo=100)
    text = generate_melody(ls, 0, MID, "T", octave=0)
    assert _voice_bar(text, 1) == "4.3{h}.2 {dy f} 0.3.2 |"
    assert _built_effects(_build(text, tmp_path))[0][4] is True


def test_slide_out_marks_the_end_of_the_note(tmp_path):
    # E4 平时弹 1 弦空弦；空弦滑不出去，带尾滑时放到 2 弦 5 品
    ls = _sheet([_flagged(0, [(0, 1, 62, {"slide_out": "down"}), (2, 1, 64, {"slide_out": "up"})])],
                feel="straight", tempo=100)
    text = generate_melody(ls, 0, MID, "T", octave=0)
    assert _voice_bar(text, 1) == "3.2{sod}.4 {dy f} r.4 5.2{sou}.4 r.4 |"
    assert [x[5] for x in _built_effects(_build(text, tmp_path))] == ["out_down", "out_up"]
    # 指法还是落在空弦上时不写滑出
    line = [M.Sung(0, 0, 1, 59, (2, 0), slide_out="down"), M.Sung(1, 1, 1, 60, (2, 1))]
    assert M.note_effects(line) == {}
    # 连到下一个音（h / sl）时音高的变化已经写了：不再写滑出
    ls = _sheet([_flagged(0, [(0, 1, 62, {"slide_out": "up"}), (1, 1, 64, L)])], feel="straight", tempo=100)
    assert _voice_bar(generate_melody(ls, 0, MID, "T", octave=0), 1) == "3.2{h}.4 {dy f} 5.2.4 r.2 |"


def test_no_flags_no_effects():
    # 没有标记（旧 leadsheet）、legato: false、slide_out: null：与原来完全一样，没有任何音符效果
    mel = [(0, 1, 62, {}), (1, 1, 64, {}), (2, 2, 57, {}), (4, 2, 62, {})]
    ls = _sheet([_flagged(0, mel[:3]), _flagged(1, mel[3:])], feel="straight")
    explicit = _sheet([_flagged(0, [(o, d, m, {"legato": False, "slide_out": None}) for o, d, m, _ in mel[:3]]),
                       _flagged(1, [(o, d, m, {"legato": False}) for o, d, m, _ in mel[3:]])], feel="straight")
    text = generate_melody(ls, 0, MID, "T")
    assert generate_melody(explicit, 0, MID, "T") == text
    assert not re.search(r"\{(h|sl|sod|sou)\}", text)
    assert all(not n.legato and n.slide_out is None for n in M.melody_line(ls, 0)[0])


def test_hammer_on_sits_on_the_last_tied_piece_across_the_bar_line(tmp_path):
    # D4 从第 4 拍唱过小节线，第 2 小节第 2 拍连到 E4：h 挂在小节线后的连音线延续上（紧挨着 E4），
    # 第 2 小节的指型名照旧挂在那一拍
    ls = _sheet([_flagged(0, [(3, 2, 62, {})]), _flagged(1, [(1, 1, 64, L)], chord="G")], feel="straight")
    text = generate_melody(ls, 0, MID, "T", octave=0)
    assert _voice_bar(text, 1) == "r.2 {dy f d} 3.2.4 |"
    assert _voice_bar(text, 2) == '-.2{h}.4 {ch "G"} 5.2.4 r.2 |'
    assert text.count("{h}") == 1
    score = _build(text, tmp_path)
    assert [x for x in _built_effects(score) if x[4]] == [(4.0, 2, 3, True, True, None)]
    assert check_playability(score, MID, "melody") == []


def test_shuffle_and_triplet_splits_keep_one_effect_on_the_last_piece(tmp_path):
    # 第 1 小节：D4 从拍点唱到第 2 拍的 +⅔，连到 E4（摇摆八分）：D4 被八分对切成 4 分 + 连音线八分，h 只在后一段；
    # 第 2 小节：D4 唱到第 2 拍的 +⅓（三连音拍），h 挂在拍内的三连音连音线延续上
    ls = _sheet([_flagged(0, [(0, 1 + 2 * T3, 62, {}), (1 + 2 * T3, T3, 64, L)]),
                 _flagged(1, [(0, 1 + T3, 62, {}), (1 + T3, 2 * T3, 64, L)])])
    text = generate_melody(ls, 0, MID, "T", octave=0)
    assert _voice_bar(text, 1) == "3.2.4 {dy f} -.2{h}.8 5.2.8 r.2 |"
    assert _voice_bar(text, 2) == "3.2.4 -.2{h}.8 {tu 3} 5.2.4 {tu 3} r.2 |"
    assert text.count("{h}") == 2
    score = _build(text, tmp_path)
    assert [x[0] for x in _built_effects(score) if x[4]] == [1.0, 5.0]
    assert _onsets(score) == [(0.0, 62), (round(1 + 2 * T3, 4), 64), (4.0, 62), (round(5 + T3, 4), 64)]
    assert check_playability(score, MID, "melody") == []


def test_legato_after_a_dropped_or_reoctaved_note_is_plucked():
    # 前一个音被丢掉（起音落到同一格上的后一个音）：连奏音照常拨弦
    ls = _sheet([_flagged(0, [(0, 1, 62, {}), (0.02, 0.98, 63, {}), (1, 1, 64, L)])], feel="straight")
    line, dropped = M.melody_line(ls, 0, octave=0)
    assert [(n.pitch, n.legato) for n in line] == [(62, False), (64, False)] and len(dropped) == 1
    # 前后两个音移了不同的八度（overrides.melody_octave 从第 2 小节起升八度）：音程不对了，照常拨弦
    ls = _sheet([_flagged(0, [(3, 1, 62, {})]), _flagged(1, [(0, 1, 64, L), (1, 1, 65, L)])], feel="straight")
    line, _ = M.melody_line(ls, 0, octave={2: 12})
    assert [(n.pitch, n.legato) for n in line] == [(62, False), (76, False), (77, True)]
    # 连奏音把前一个音的音尾接上（原唱数据里有一点空隙也一样）
    ls = _sheet([_flagged(0, [(0, 0.9, 62, {}), (1, 1, 64, L)])], feel="straight")
    line, _ = M.melody_line(ls, 0, octave=0)
    assert [(n.pos, n.dur, n.legato) for n in line] == [(0.0, 1.0, False), (1.0, 1.0, True)]


def test_refret_keeps_a_legato_pair_on_one_string():
    # 第 2 个音换把违规：换它（排在前面）会把它与后面连奏的音拆到两根弦上；改换前一个音同样不违规，就换前一个
    line = [M.Sung(0, 0, 0.5, 60, (2, 1)), M.Sung(0.5, 0.5, 0.5, 64, (3, 9)),
            M.Sung(1.0, 1.0, 1.0, 66, (3, 11), legato=True)]
    assert M.shift_breaks(line, MID.max_shift) == [1]
    j, old, new = M.refret(line, 1, MID)
    assert j == 0 and old == (2, 1) and new[0] in (3, 4)
    assert M.shift_breaks(line, MID.max_shift) == [] and line[1].where == (3, 9)


def test_fingerstyle_ignores_the_flags():
    from guitarize.arrange.finger import generate_finger
    mel = [(0, 1, 62, {}), (1, 1, 64, L), (2, 2, 67, {"slide_out": "down"})]
    ls = _sheet([_flagged(0, mel, "G"), _flagged(1, [(0, 2, 69, {}), (2, 2, 71, L)], "D")], feel="straight")
    assert generate_finger(ls, 0, MID, "T") == generate_finger(_strip(ls), 0, MID, "T")
