import re
from dataclasses import replace

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


def _plain(line: str) -> str:
    """去掉和弦指型 / 力度标记（{ch "X"}、{dy f}，或二者合在一起），只看音和节奏。"""
    line = re.sub(r' \{(?:ch "[^"]*"\s*)?(?:dy \w+)?\}', "", line)
    return line


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
    line = melody_line(ls, 0, MID, octave="section")
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
    assert _plain(line).startswith("(2.4 3.5).8 r.")      # 低音 + 填充只留半拍，余下是休止


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
    assert _plain(_voice2_bar(text, 1)) == "(0.2 1.4).4 -.4.4 (0.2 6.5).2 |"   # 填充截在第 2 拍，低音接着响到第 3 拍


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
    assert _plain(_voice2_bar(text, 1)) == "(0.2 1.4).4 -.4.8 r.8 (0.2 6.5).2 |"


def test_shift_between_two_melody_notes_moves_one_an_octave():
    # (b)：D5 只能在 1 弦 10 品，反拍的 G4 在 2 弦 8 品，没有低音 / 填充；max_shift = 1 时半拍内 10 → 8 品是真违规
    # （校验只看相隔不到 1 拍的起音）→ 用真实自检修补：G4 降八度落到 3 弦空弦
    from guitarize.arrange.pipeline import self_check
    tight = replace(MID, max_shift=1)
    ls = _sheet([_bar(0, None, "N", [(0, 0.5, 74), (0.5, 1.5, 67), (2, 2, 65)])])
    calls = []

    def check(text):
        found = self_check(text, tight, "finger")
        calls.append(found)
        return found

    text = generate_finger(ls, 0, tight, "T", check=check)
    assert [(v.rule, v.detail) for v in calls[0]] == [("shift", "0.50 拍内换把 10→8 品")]
    assert calls[-1] == []
    assert _bar_comment(text, 1) == "// bar 1 · melody-12 @1.5 (shift)"
    assert _plain(text.split("// bar 1 · melody-12 @1.5 (shift)\n")[1].splitlines()[0]) == "10.1.8 0.3.4 {d} 6.2.2 |"


def test_melody_line_default_keeps_contour_across_sections():
    mid = [(0, 1, 60), (1, 1, 62), (2, 1, 64), (3, 1, 65)]
    lo = [(0, 1, 50), (1, 1, 52), (2, 1, 53), (3, 1, 55)]
    ls = _sheet([_bar(0, "A", "C", mid), _bar(1, "B", "C", lo)])
    line = melody_line(ls, 0, MID)
    shifts = {p - q for (_, p), q in zip(line, [60, 62, 64, 65, 50, 52, 53, 55])}
    assert len(shifts) == 1 and shifts.pop() % 12 == 0                  # 整曲同一个移位：段落之间不换八度


def test_finger_draft_marks_chord_shapes_and_balances_voices(tmp_path):
    # 指弹谱也标出该按的和弦指型（指法调名称 + 和弦图），伴奏声部比旋律轻
    text = generate_finger(leadsheet(), 4, MID, "T")
    assert '\\chord "G" 3 0 0 0 2 3' in text and '\\chord "C"' in text and '\\chord "B7/D#"' in text
    v1, v2 = text.split("\\voice")[1:]
    both = v1 + v2
    assert 'ch "G"' in both and 'ch "C"' in both and 'ch "B7/D#"' in both
    assert both.count('ch "G"') == 1                     # 同一指型连续时只标一次
    assert 'ch "C"' in v1                                # 旋律声部在该处有起音时挂在旋律声部（黑色、醒目）
    assert 'ch "G"' in v2                                # 第 1 小节有段落名「サビ」：挂在低音声部，免得两行文字重叠
    assert "dy f" in v1 and "dy mf" in v2
    src = tmp_path / "f.atex"
    src.write_text(text, encoding="utf-8")
    built = alphatab.build(src, tmp_path / "b", "f")
    import pretty_midi
    score = load_score(built["json"])
    velocity = {(round(n.start, 2), n.pitch): n.velocity
                for inst in pretty_midi.PrettyMIDI(str(built["mid"])).instruments for n in inst.notes}
    voice = {v: [velocity[(round(score.seconds(s.start), 2), s.midi)] for s in score.sounding()
                 if s.voice == v and (round(score.seconds(s.start), 2), s.midi) in velocity] for v in (0, 1)}
    assert voice[0] and voice[1] and max(voice[1]) < min(voice[0])     # 伴奏（声部 2）比旋律轻
    assert check_playability(score, MID, "finger") == []


def test_every_bar_shows_its_chord_shape():
    # 每小节第 1 拍都标指型（和弦没变也标）；同一小节第 3 拍指型不变时不重复
    mel = [(0, 1, 67), (1, 1, 69), (2, 2, 71)]
    ls = _sheet([_bar(0, None, "C", mel), _bar(1, None, "C", mel), _bar(2, None, "C", mel)])
    marks, _ = finger.shape_marks(ls, 0, MID)
    assert marks == {0.0: "C", 4.0: "C", 8.0: "C"}
    text = generate_finger(ls, 0, MID, "T")
    assert text.count('ch "C"') == 3


def test_skeleton_keeps_both_notes_that_land_on_the_same_eighth():
    # なんもねえ 第 1 小节：B3（第 3 拍，1/3 拍）后紧跟长音 F#4（3+1/3）——Shuffle 记谱都落在第 3 拍；
    # 后一个顺延到下一个空着的八分位，不能丢掉进副歌的长音
    ls = _sheet([_bar(0, None, "N", [(1 + 2 / 3, 1 / 3, 66), (2, 1, 66), (3, 1 / 3, 59), (3 + 1 / 3, 4 / 3, 66)])],
                feel="shuffle")
    assert skeleton_notes(ls, MID) == [(1.5, 66), (2.0, 66), (3.0, 59), (3.5, 66)]


def test_held_melody_note_ties_across_the_bar_line():
    # 拍 3.5 起的长音唱到下一小节：写成连音线延续，下一小节不能以休止开头
    ls = _sheet([_bar(0, None, "N", [(3.5, 2.0, 67)]), _bar(1, None, "N", [(2, 2, 69)])])
    text = generate_finger(ls, 0, MID, "T")
    v1 = text.split("\\voice")[1]
    bar2 = v1.split("// bar 2\n")[1].splitlines()[0]
    assert _plain(bar2).startswith("-.")
