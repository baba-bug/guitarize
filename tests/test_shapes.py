import pytest

from guitarize.arrange.capo import choose_capo, melody_octave
from guitarize.arrange.shapes import playable, voicings, with_bass
from guitarize.config import load_profile

MID = load_profile("intermediate")
BEG = load_profile("beginner")


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
    with pytest.raises(ValueError):
        voicings("F#m7-5", BEG)
    assert all(not v.barre for v in voicings("A", BEG))


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
