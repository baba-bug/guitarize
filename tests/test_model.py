import pytest

from conftest import TEX_HEAD
from guitarize.verify.model import load_score, swing


def test_swing_mapping():
    assert swing(0.5, "triplet8th") == pytest.approx(2 / 3)
    assert swing(1.25, "triplet8th") == pytest.approx(1 + 1 / 3)
    assert swing(0.75, "triplet8th") == pytest.approx(2 / 3 + 1 / 6)
    assert swing(0.5, "none") == 0.5


def test_demo_finger_model(demo_song, tmp_path):
    from guitarize.render import alphatab

    score = load_score(alphatab.build(demo_song.atex("finger"), tmp_path / "b", "f")["json"])
    assert score.tempo == 100 and len(score.bars) == 8
    assert score.bars[1].start == 4 and score.bars[1].length == 4
    mel = score.melody(voice=0)
    assert len(mel) == 28
    assert [m.midi for m in mel[:4]] == [60, 60, 67, 67]
    assert mel[6].start == 6 and mel[6].end == 8  # 第 2 小节的二分音符 G4
    assert score.seconds(4) == pytest.approx(2.4)


def test_triplet_feel_swings_offbeats(build_tex):
    score = build_tex("\\tf triplet-8th\n:8 0.1 0.1 0.1 0.1 :2 0.1 |")
    starts = [e.start for e in score.events]
    assert starts == pytest.approx([0, 2 / 3, 1, 5 / 3, 2])
    assert [e.written_start for e in score.events] == pytest.approx([0, 0.5, 1, 1.5, 2])


def test_explicit_triplets_not_swung(build_tex):
    score = build_tex("\\tf triplet-8th\n:8 0.1 {tu 3} 0.1 {tu 3} 0.1 {tu 3} :4 0.1 :2 0.1 |")
    assert [e.start for e in score.events][:3] == pytest.approx([0, 1 / 3, 2 / 3])


def test_ties_extend_sounding(build_tex):
    score = build_tex(":2 3.1 :4 -.1 r |")
    snd = score.sounding()
    assert len(snd) == 1 and snd[0].start == 0 and snd[0].end == 3
    assert len(score.attacks()) == 1


def test_capo_in_midi_and_attack_flags(build_tex):
    head = TEX_HEAD + "\\capo 4\n"
    score = build_tex(":4 (3.6 2.5) {sd} (0.1 0.2) {su} 0.3 r |", head=head)
    assert score.capo == 4
    attacks = score.attacks()
    assert [n.midi for n in attacks[0].notes] == [47, 51]
    assert attacks[0].strum and attacks[0].direction == "down"
    assert attacks[1].direction == "up"
    assert not attacks[2].strum and attacks[2].direction is None


def test_voice_lengths_detect_overfull_bar(build_tex):
    score = build_tex(":4 0.1 0.1 0.1 0.1 0.1 |")
    assert score.voice_lengths()[(0, 0)] == pytest.approx(5)
