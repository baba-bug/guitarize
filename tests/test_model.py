import pytest
import pretty_midi

from conftest import TEX_HEAD
from guitarize.verify.model import load_score, performed_times


def test_performed_times_pairs_only():
    e, q = 0.5, 1.0
    # 一对八分 → 2/3 + 1/3
    assert performed_times([(0, e, 1), (0.5, e, 1)], "triplet8th") == pytest.approx([(0, 2 / 3), (2 / 3, 1 / 3)])
    # 切分：后一个八分前面不是成对 → 原样
    assert performed_times([(0, q, 1), (1, e, 1), (1.5, q, 1)], "triplet8th") == [(0, q), (1, e), (1.5, q)]
    # 起点不在四分边界的一对（0.5 起）→ 原样
    assert performed_times([(0, e, 1), (0.5, e, 1), (1, e, 1)], "triplet8th")[1:] == \
        pytest.approx([(2 / 3, 1 / 3), (1, e)])
    assert performed_times([(0.5, e, 1), (1.0, e, 1)], "triplet8th") == [(0.5, e), (1.0, e)]
    # 连音组、其他 feel
    assert performed_times([(0, 1 / 3, 3), (1 / 3, 1 / 3, 3)], "triplet8th") == [(0, 1 / 3), (1 / 3, 1 / 3)]
    assert performed_times([(0, e, 1), (0.5, e, 1)], "none") == [(0, e), (0.5, e)]


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


def test_dynamics_palm_mute_and_accent_give_the_attack_velocity(build_tex):
    # 力度沿用到下一个写了 dy 的拍（alphaTab 的 alphaTex 导入如此）；重音 +16、强重音 +32，同 alphaTab 的 MIDI。
    # 音符效果要调弦写在括号里的文件头（TEX_HEAD 的旧写法下 '2.5{pm}' 解析不了）
    from guitarize.arrange.atex import header

    head = header("T", 120, 0, "straight", "tabs")
    score = build_tex("(2.5{pm} 4.4{pm}).4 {bd 60 dy mp} (2.5{ac} 4.4).4 {bu 40 dy f} (2.5 4.4{hac}).2 {bd 60} |",
                      head=head)
    attacks = score.attacks()
    assert [a.velocity for a in attacks] == [63, 111, 127]
    assert [a.palm_mute for a in attacks] == [True, False, False]
    assert [a.direction for a in attacks] == ["down", "up", "down"]
    first, second = score.events[0], score.events[1]
    assert first.dynamics == "MP" and first.notes[0].palm_mute and second.notes[0].accent == 1
    assert build_tex("(2.5 4.4).1 |", head=head).attacks()[0].velocity == 95   # 没写力度：alphaTab 默认 f


def test_voice_lengths_detect_overfull_bar(build_tex):
    score = build_tex(":4 0.1 0.1 0.1 0.1 0.1 |")
    assert score.voice_lengths()[(0, 0)] == pytest.approx(5)


def test_syncopation_is_not_swung(build_tex):
    s1 = build_tex("\\tf triplet-8th\n:8 0.1 :4 1.1 :8 2.1 :2 3.1 |")
    assert [e.start for e in s1.events] == pytest.approx([0, 0.5, 1.5, 2])
    s2 = build_tex("\\tf triplet-8th\n0.1.4 {d} 1.1.8 2.1.2 |")
    assert [e.start for e in s2.events] == pytest.approx([0, 1.5, 2])


def test_model_matches_real_midi_timing(tmp_path):
    from guitarize.render import alphatab

    body = ("\\tf triplet-8th\n:8 0.1 1.1 :4 2.1 :8 3.1 0.2 :4 r |\n"
            ":8 r 1.2 :4 2.2 :8 0.3 2.3 :4 0.4 |\n"
            ":8 0.1 :4 1.1 :8 2.1 :2 3.1 |")
    src = tmp_path / "m.atex"
    src.write_text(TEX_HEAD + body, encoding="utf-8")
    built = alphatab.build(src, tmp_path / "b", "m")
    score = load_score(built["json"])
    onsets = sorted(n.start for n in pretty_midi.PrettyMIDI(str(built["mid"])).instruments[0].notes)
    expected = sorted(score.seconds(a.start) for a in score.attacks())
    assert len(onsets) == len(expected)
    assert onsets == pytest.approx(expected, abs=0.005)


def test_triplet_beat_next_to_swung_pairs(tmp_path):
    # 三连音拍按记谱演奏（k/3），不参与、也不打断前后的八分对；模型与 alphaTab 的 MIDI 一致
    from guitarize.render import alphatab

    body = "\\tf triplet-8th\n:8 0.1 1.1 0.2 {tu 3} :4 1.2 {tu 3} :8 2.1 r :4 r {tu 3} :8 3.1 {tu 3} |"
    src = tmp_path / "t.atex"
    src.write_text(TEX_HEAD + body, encoding="utf-8")
    built = alphatab.build(src, tmp_path / "b", "t")
    score = load_score(built["json"])
    assert [e.start for e in score.events] == pytest.approx([0, 2 / 3, 1, 4 / 3, 2, 8 / 3, 3, 11 / 3])
    assert [e.written_start for e in score.events] == pytest.approx([0, 0.5, 1, 4 / 3, 2, 2.5, 3, 11 / 3])
    assert [e.tuplet for e in score.events] == [1, 1, 3, 3, 1, 1, 3, 3]
    onsets = sorted(n.start for n in pretty_midi.PrettyMIDI(str(built["mid"])).instruments[0].notes)
    assert onsets == pytest.approx([score.seconds(a.start) for a in score.attacks()], abs=0.005)
