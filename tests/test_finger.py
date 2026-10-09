import pytest

from guitarize.arrange.finger import bass_events, generate_finger, place_fill, skeleton_notes, written_position
from guitarize.config import Thresholds, load_profile
from guitarize.render import alphatab
from guitarize.verify import metrics
from guitarize.verify.model import load_score
from guitarize.verify.playability import check_playability

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
