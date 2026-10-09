import pytest
from conftest import capo4_bar12_leadsheet

from guitarize.arrange import finger
from guitarize.arrange.finger import (bass_events, generate_finger, melody_floor, melody_line, place_fill,
                                      skeleton_notes, written_position)
from guitarize.config import Thresholds, load_profile
from guitarize.render import alphatab
from guitarize.verify import metrics
from guitarize.verify.model import load_score
from guitarize.verify.playability import Violation, check_playability

MID = load_profile("intermediate")


def leadsheet(feel="shuffle"):
    # B 大调两小节：B（4 拍）、E（2 拍）+ D#/G（2 拍）。旋律在演奏时间轴（Shuffle 反拍在 2/3）。
    mel1 = [(0, 1, 75), (1, 2 / 3, 73), (5 / 3, 1 / 3, 71), (2, 2, 70)]
    mel2 = [(4, 1, 68), (5, 1, 71), (6, 2, 70)]
    bars = [
        {"bar": 1, "section": "サビ", "t_start": 0.0, "t_end": 1.44,
         "chords": [{"beat": 1, "dur_beats": 4, "label": "B", "confidence": "high"}],
         "melody": [{"start_beats": s, "dur_beats": d, "midi": m, "conf": 1.0} for s, d, m in mel1]},
        {"bar": 2, "section": None, "t_start": 1.44, "t_end": 2.88,
         "chords": [{"beat": 1, "dur_beats": 2, "label": "E", "confidence": "high"},
                    {"beat": 3, "dur_beats": 2, "label": "D#/G", "confidence": "high"}],
         "melody": [{"start_beats": s, "dur_beats": d, "midi": m, "conf": 1.0} for s, d, m in mel2]},
    ]
    return {"key": "B", "mode": "major", "tempo_bpm": 167.0, "feel": feel, "beats_per_bar": 4, "bars": bars}


def test_written_position():
    assert written_position(2 / 3, "shuffle") == 0.5
    assert written_position(1 / 3, "shuffle") == 0.0
    assert written_position(1.0, "shuffle") == 1.0
    assert written_position(1.5, "straight") == 1.5
    assert written_position(1.2, "straight") == 1.0


def test_skeleton_notes_drop_short_offbeats():
    notes = skeleton_notes(leadsheet(), MID)
    assert (1.5, 71) not in notes          # 5/3 拍的短反拍音不是骨架音
    assert notes[0] == (0.0, 75) and (2.0, 70) in notes


def test_bass_events_follow_chords_and_slash():
    ev = bass_events(leadsheet(), 4, MID)
    # 实际 B / E / D#/G 在 capo 4 下为指法调 G / C / B/D#
    assert ev == [(0.0, 2.0, 43, 11), (2.0, 2.0, 43, 11),   # G：根音 G2，填充三音 B；第 3 拍重复根音
                  (4.0, 2.0, 48, 4),                        # C：根音 C3，填充三音 E
                  (6.0, 2.0, 51, 11)]                       # B/D#：低音 D#3；三音就是低音 → 填充根音 B


def test_place_fill():
    assert place_fill(11, 43, 66) == 59      # B3：在 G2 之上、F#4 之下，且落在 50–64
    assert place_fill(4, 48, 64) == 52       # E3（E4 = 64 不低于旋律）
    assert place_fill(4, 60, 62) is None     # 两者之间放不下 E


def test_generate_finger_is_playable_and_matches_reference(tmp_path):
    ls = leadsheet()
    text = generate_finger(ls, 4, MID, "Test")
    assert text.count("\\voice") == 2 and "\\capo 4" in text and "// bar 1 [サビ]" in text
    src = tmp_path / "f.atex"
    src.write_text(text, encoding="utf-8")
    score = load_score(alphatab.build(src, tmp_path / "b", "f")["json"])
    assert check_playability(score, MID, "finger") == []
    s = metrics.summarize(metrics.score_bars(score, ls, "finger"), [], Thresholds(), "finger")
    assert s["melody_f1"] == pytest.approx(1.0)
    assert s["chord_agreement"] >= 0.85
    v2 = [e for e in score.events if e.voice == 1 and not e.rest]
    assert all(len(e.notes) == 2 for e in v2)    # 每个低音都带一个填充音


def test_place_fill_keeps_three_semitones_below_melody():
    assert place_fill(11, 43, 62) == 59      # B3 比 D4 低 3 个半音：可以
    assert place_fill(11, 43, 61) == 47      # 比 C#4 只低 2 个半音 → 降到 B2
    assert place_fill(11, 43, None) == 59    # 无旋律：上限 E4


def test_melody_floor_spans_the_whole_sustain():
    items = [(0.0, 1.0, 71), (1.0, 1.0, 60), (2.0, 2.0, 67)]
    assert melody_floor(items, 0.0, 2.0) == 60     # 不只看起音时刻的 71
    assert melody_floor(items, 2.0, 2.0) == 67
    assert melody_floor(items, 4.0, 2.0) is None


def _bar(i, section, chord, mel):
    return {"bar": i + 1, "section": section, "t_start": i * 2.4, "t_end": (i + 1) * 2.4,
            "chords": [{"beat": 1, "dur_beats": 4, "label": chord, "confidence": "high"}],
            "melody": [{"start_beats": i * 4 + o, "dur_beats": d, "midi": m, "conf": 1.0} for o, d, m in mel]}


def _sheet(bars, feel="straight"):
    return {"key": "C", "mode": "major", "tempo_bpm": 100.0, "feel": feel, "beats_per_bar": 4, "bars": bars}


def test_melody_octave_is_chosen_per_section():
    mid = [(0, 1, 60), (1, 1, 62), (2, 1, 64), (3, 1, 65)]
    lo = [(0, 1, 50), (1, 1, 52), (2, 1, 53), (3, 1, 55)]
    ls = _sheet([_bar(0, "A", "C", mid), _bar(1, None, "C", mid), _bar(2, "B", "C", lo), _bar(3, None, "C", lo)])
    line = melody_line(ls, 0, MID)
    assert {p for pos, p in line if pos < 8} <= {60, 62, 64, 65}       # A 段已在音域中部：不移
    assert {p for pos, p in line if pos >= 8} <= {62, 64, 65, 67}      # B 段整段升八度（整曲统一移位做不到）


def test_fallbacks_are_surfaced_in_bar_comment():
    # G 和弦低音 G2 + 旋律 B4（1 弦 7 品）跨度放不下 → 低音升八度（填充也放不下）
    ls = _sheet([_bar(0, None, "G", [(0, 1, 71), (1, 1, 60), (2, 1, 60), (3, 1, 62)])])
    text = generate_finger(ls, 0, MID, "T")
    comment = next(line for line in text.splitlines() if line.startswith("// bar 1"))
    assert comment.startswith("// bar 1 · ") and "bass+12 @1" in comment


def _bar_comment(text, n):
    return next(line for line in text.splitlines() if line == f"// bar {n}" or line.startswith(f"// bar {n} "))


def test_repair_loop_with_scripted_check():
    calls = []

    def check(text):
        calls.append(text)
        # 第 2 小节第 1 拍反拍（演奏时间 4 + 2/3 = 记谱 4.5）：C 和弦低音 + 填充仍按着
        return [Violation(2, 4 + 2 / 3, "span", "x")] if len(calls) == 1 else []

    text = generate_finger(leadsheet(), 4, MID, "T", check=check)
    assert len(calls) == 2 and calls[-1] == text
    assert _bar_comment(text, 2) == "// bar 2 · bass cut @1.5 (span)"
    assert _bar_comment(calls[0], 2) == "// bar 2"
    v2 = text.split("\\voice")[2]
    line = v2.split("// bar 2\n")[1].splitlines()[0]
    assert line.startswith("(2.4 3.5).8 r.")      # 低音 + 填充只留半拍，余下是休止


def test_repair_loop_stops_when_nothing_applies():
    calls = []

    def check(text):
        calls.append(text)
        return [Violation(1, 0.0, "bar_length", "x")]

    text = generate_finger(leadsheet(), 4, MID, "T", check=check)
    assert len(calls) == 1 and calls[0] == text


def test_repair_rounds_are_capped(monkeypatch):
    monkeypatch.setattr(finger, "REPAIR_ROUNDS", 1)
    calls = []

    def check(text):
        calls.append(text)
        return [Violation(1, 2 / 3, "span", "x"), Violation(2, 6.0, "span", "x")]

    generate_finger(leadsheet(), 4, MID, "T", check=check)
    assert len(calls) == 1


def _voice2_bar(text, n):
    return text.split("\\voice")[2].split(f"// bar {n}\n")[1].splitlines()[0]


def test_fill_cut_keeps_the_bass_ringing_as_a_tie():
    text = generate_finger(capo4_bar12_leadsheet(), 4, MID, "T")
    assert _bar_comment(text, 1) == "// bar 1 [サビ] · fill cut @2 (same string)"
    assert _voice2_bar(text, 1) == "(0.2 1.4).4 -.4.4 (0.2 6.5).2 |"   # 填充截在第 2 拍，低音接着响到第 3 拍


def test_shift_with_nothing_at_t_is_repaired_at_the_previous_attack():
    # (a)：第 3 拍的新低音（5 弦 6 品）那一刻没有可截、可去的（旋律、填充都是空弦）。
    # 前一个起音（第 2 拍反拍）时第 1 拍的低音还按着 → 截到那里，第 3 拍的低音保留
    calls = []

    def check(text):
        calls.append(text)
        return [Violation(1, 2.0, "shift", "0.33 拍内换把 1→6 品")] if len(calls) == 1 else []

    text = generate_finger(capo4_bar12_leadsheet(), 4, MID, "T", check=check)
    assert len(calls) == 2
    assert _bar_comment(text, 1) == "// bar 1 [サビ] · fill cut @2 (same string) · bass cut @2.5 (shift)"
    assert _voice2_bar(text, 1) == "(0.2 1.4).4 -.4.8 r.8 (0.2 6.5).2 |"


def test_shift_between_two_melody_notes_moves_one_an_octave():
    # (b)：D5 只能在 1 弦 10 品，G4 在 2 弦 8 品，没有低音 / 填充。校验说这里换把 → G4 降八度落到 3 弦空弦
    ls = _sheet([_bar(0, None, "N", [(0, 1, 74), (1, 1, 67), (2, 2, 65)])])
    calls = []

    def check(text):
        calls.append(text)
        return [Violation(1, 1.0, "shift", "x")] if len(calls) == 1 else []

    text = generate_finger(ls, 0, MID, "T", check=check)
    assert calls[0].split("// bar 1\n")[1].splitlines()[0] == "10.1.4 8.2.4 6.2.2 |"
    assert _bar_comment(text, 1) == "// bar 1 · melody-12 @2 (shift)"
    assert text.split("// bar 1 · melody-12 @2 (shift)\n")[1].splitlines()[0] == "10.1.4 0.3.4 6.2.2 |"
