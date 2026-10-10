"""旋律谱（arr = melody）：leadsheet 的每一个旋律音、按原唱的节奏（Shuffle 三连音位置精确）、单声部、可弹。"""
import re

from guitarize.arrange import melody as M
from guitarize.arrange.melody import generate_melody
from guitarize.arrange.pipeline import self_check
from guitarize.config import load_profile
from guitarize.render import alphatab
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


def test_single_voice_display_and_a_chord_shape_on_every_bar():
    mel = [(0, 1, 64), (1, 1, 66), (2, 2, 68)]
    ls = _sheet([_bar(0, mel, "B", "サビ"), _bar(1, mel, "B"), _bar(2, mel, "E")], feel="straight")
    text = generate_melody(ls, 4, MID, "T")
    assert "\\voice" not in text and "\\staff {score tabs}" in text and "\\ks B" in text
    assert "\\capo 4" in text and "dy f" in text and "// bar 1 [サビ]" in text
    for n, shape in ((1, "G"), (2, "G"), (3, "C")):                  # 实际 B / E 在 capo 4 下是指法调 G / C
        assert f'ch "{shape}"' in _voice_bar(text, n)
    assert '\\chord "G"' in text and '\\chord "C"' in text


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
