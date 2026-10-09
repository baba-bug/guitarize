import numpy as np

from guitarize import theory
from guitarize.skeleton import skeleton_mask


def test_pitch_names():
    assert theory.pc("C#") == 1 and theory.pc("Db") == 1
    assert theory.pc("Bb") == 10 and theory.pc("B") == 11 and theory.pc("E#") == 5
    assert theory.midi_name(60) == "C4" and theory.midi_name(71) == "B4"


def test_majmin():
    assert theory.majmin("B") == "B:maj"
    assert theory.majmin("G#m7") == "G#:min"
    assert theory.majmin("D#7") == "D#:maj"
    assert theory.majmin("Ebmaj7") == "D#:maj"
    assert theory.majmin("F#m7-5") == "F#:min"
    assert theory.majmin("E/G#") == "E:maj"
    assert theory.majmin("Asus4") == "A:maj"
    assert theory.majmin("N") == "N" and theory.majmin(None) == "N"


def test_transpose_label():
    assert theory.transpose_label("B", -4) == "G"
    assert theory.transpose_label("D#7", -4) == "B7"
    assert theory.transpose_label("G#m", -4) == "Em"
    assert theory.transpose_label("E/G#", -4) == "C/E"
    assert theory.transpose_label("N", 3) == "N"


def test_chord_tones():
    assert theory.chord_tones("G") == {7, 11, 2}
    assert theory.chord_tones("Am7") == {9, 0, 4, 7}
    assert theory.chord_tones("C/E") == {0, 4, 7}


def test_infer_majmin():
    w = np.zeros(12)
    w[[7, 11, 2]] = 1
    assert theory.infer_majmin(w, bass_pc=7) == {"G:maj"}
    w = np.zeros(12)
    w[[11, 6]] = 1  # 强力和弦：大小三并列
    assert theory.infer_majmin(w, bass_pc=11) == {"B:maj", "B:min"}
    assert theory.infer_majmin(np.zeros(12)) == {"N"}


def test_skeleton_mask():
    starts = [0, 5 / 3, 2, 7 / 3, 8 / 3, 5]
    durs = [1, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1]
    assert skeleton_mask(starts, durs) == [True, False, True, False, False, True]
    # 乐句开头（前面休止 ≥ 1 拍）即使在反拍也保留
    assert skeleton_mask([0, 8 / 3], [1, 1 / 3]) == [True, True]
    # 反拍长音保留
    assert skeleton_mask([0, 2 / 3], [2 / 3, 4 / 3]) == [True, True]
