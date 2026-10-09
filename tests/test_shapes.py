import pytest

from guitarize import theory
from guitarize.arrange.capo import choose_capo, effective_profile, melody_octave
from guitarize.arrange.shapes import OPEN_STRINGS, playable, sounding, voicings, with_bass
from guitarize.config import load_profile

MID = load_profile("intermediate")
BEG = load_profile("beginner")
ADV = load_profile("advanced")


def test_open_shape_first():
    v = voicings("G", MID)[0]
    assert v.name == "G" and v.frets == (3, 0, 0, 0, 2, 3) and v.cost == 0.0


def test_substitute_seventh_for_missing_open_major():
    v = voicings("B", MID)[0]
    assert v.name == "B7" and v.frets == (2, 0, 2, 1, 2, None)


def test_slash_chords():
    assert voicings("G/B", MID)[0].frets == (3, 0, 0, 0, 2, None)
    assert voicings("D/F#", MID)[0].frets == (2, 3, 2, 0, None, 2)
    assert voicings("C/E", MID)[0].frets == (0, 1, 0, 2, 3, 0)
    v = voicings("B/D#", MID)[0]
    assert v.name == "B7/D#" and v.frets == (2, 0, 2, 1, None, None)


def test_barre_only_when_allowed():
    assert voicings("F", MID)[0].barre is True
    assert all(not v.barre for v in voicings("A", BEG))
    half_dim = voicings("F#m7-5", BEG)   # 没有完整指型 → 只剩 1–4 弦的部分指型
    assert half_dim and all(not v.barre and v.frets[4:] == (None, None) for v in half_dim)
    with pytest.raises(ValueError):
        voicings("H7", BEG)


def test_power_chord_and_no_chord():
    assert voicings("E5", MID)[0].name == "E"
    assert voicings("N", MID) == []


def test_with_bass_moves_to_fifth_string_when_sixth_is_too_far():
    v = with_bass(voicings("G", MID)[0], 1, MID)   # G/C#：6 弦需要 9 品（跨度超限）→ 改用 5 弦 4 品
    assert v.name == "G/C#" and v.frets == (3, 0, 0, 0, 4, None) and playable(v.frets, MID)


def test_choose_capo_prefers_g_shapes_for_b_major_song():
    beats = {"B": 16, "E": 16, "D#/G": 8, "G#m": 8, "C#": 8, "D#": 8}
    best = choose_capo(beats, MID)[0]
    assert best.capo == 4
    assert best.shapes["D#/G"].name == "B7/D#"
    assert best.shapes["G#m"].name == "Em" and best.shapes["E"].name == "C"


def test_melody_octave():
    assert melody_octave([71, 73, 75, 76]) == (-12, 0.0)
    assert melody_octave([]) == (0, 0.0)


def test_no_chord_label_skipped_in_capo_choice():
    best = choose_capo({"N": 4, "C": 4}, MID)[0]
    assert best.capo == 0
    assert "N" not in best.shapes


def _partial(v):
    """只用 1–3 弦或 1–4 弦（且这几根弦都弹）。"""
    return any(None not in v.frets[:n] and set(v.frets[n:]) == {None} for n in (3, 4))


def test_partial_voicing_when_no_full_voicing():
    vs = voicings("B", BEG)      # 新手档：B 没有不横按的完整指型
    assert vs[0].name == "B" and vs[0].cost == 2.0
    assert vs[0].frets == (2, 4, 4, None, None, None)
    for v in vs:
        assert _partial(v) and playable(v.frets, BEG) and not v.barre
        assert {p % 12 for p in sounding(v.frets)} == theory.chord_tones("B")
        assert len([f for f in v.frets if f]) <= BEG.max_fingers


def test_partials_rank_after_full_voicings():
    vs = voicings("G", MID)
    full = [v for v in vs if not _partial(v)]
    part = [v for v in vs if _partial(v)]
    assert vs[0].frets == (3, 0, 0, 0, 2, 3) and part
    assert min(v.cost for v in part) == 1.5 and min(v.cost for v in part) > vs[0].cost
    assert all(v.name == "G" for v in part)
    assert vs.index(part[0]) > vs.index(full[0])


def test_seventh_partials_may_drop_the_fifth():
    for v in voicings("A#7", BEG):        # 1–3 弦：根音、三音、七音
        pcs = {p % 12 for p in sounding(v.frets)}
        assert {10, 2, 8} <= pcs <= theory.chord_tones("A#7")


def test_slash_partial_keeps_plain_name_with_penalty():
    v = voicings("F#/A#", BEG)[0]       # 高音弦上做不到斜杠低音：沿用原名，代价 +0.5
    assert v.name == "F#" and v.cost == 2.5 and _partial(v)


def test_every_chord_has_a_voicing_at_every_level_and_capo():
    suffixes = ["", "m", "7", "m7", "maj7", "sus4", "sus2", "dim", "aug", "m7-5", "5"]
    for prof in (BEG, MID, ADV):
        for capo in range(8):
            eff = effective_profile(prof, capo)
            for root in theory.PITCH_CLASSES:
                for suffix in suffixes:
                    vs = voicings(root + suffix, eff)
                    assert vs and all(playable(v.frets, eff) for v in vs), (prof.name, capo, root + suffix)
                    assert all(v.barre is False for v in vs) or eff.allow_barre


def test_real_song_capo_and_shapes_unchanged():
    """なんもねえ（B 大调）的逐拍和弦统计：中级仍选变调夹 4，指型不变（部分指型代价更高，不会顶替）。"""
    beats = {"F#7": 14, "B": 48, "E": 52, "D#/G": 16, "G#m": 32, "B/D#": 20, "C#": 12, "D#": 24,
             "C#m7": 8, "F#": 17, "A": 2, "D": 2}
    best = choose_capo(beats, MID)[0]
    assert best.capo == 4
    expected = {"F#7": "D7", "B": "G", "E": "C", "D#/G": "B7/D#", "G#m": "Em", "B/D#": "G/B", "C#": "A", "D#": "B7",
                "C#m7": "Am7", "F#": "D"}
    assert {k: best.shapes[k].name for k in expected} == expected
    beg = choose_capo(beats, BEG)
    assert all(set(o.shapes) == set(beats) for o in beg)   # 新手档每个和弦都有指型（不再按缺失罚分）
