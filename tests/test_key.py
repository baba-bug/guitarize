import numpy as np

from guitarize.analyze.key import MAJOR, MINOR, estimate_key, parse_key, scale_pcs


def test_estimate_key():
    assert estimate_key(np.roll(MAJOR, 11))[:2] == ("B", "major")
    assert estimate_key(np.roll(MINOR, 9))[:2] == ("A", "minor")


def test_scale_and_parse():
    assert scale_pcs("B", "major") == {11, 1, 3, 4, 6, 8, 10}
    assert scale_pcs("A", "minor") == {9, 11, 0, 2, 4, 5, 7}
    assert parse_key("B") == ("B", "major")
    assert parse_key("G#m") == ("G#", "minor")
    assert parse_key("Bb") == ("A#", "major")


def _segs(labels):
    return [{"bar": i + 1, "beat": 1, "dur_beats": 4, "label": l} for i, l in enumerate(labels)]


def test_diatonic_fraction_counts_triads_power_roots_and_skips_n():
    from guitarize.analyze.key import diatonic_fraction

    # G 大调：C#7 的三和弦 C#-F-G# 不在调内；B5 只看根音；N 不计
    segs = _segs(["G", "Em7", "D/F#", "B5", "C#7", "N"])
    assert diatonic_fraction(segs, "G", "major") == 4 / 5
    assert diatonic_fraction(_segs(["N"]), "G", "major") == 0.0


def test_cross_check_prefers_chord_key_over_wrong_chroma_key():
    from guitarize.analyze.key import cross_check_key

    segs = _segs(["G", "C", "D", "Em", "G", "C", "Am", "D7"] * 2)
    kc = cross_check_key(("D", "major"), segs)
    assert (kc["tonic"], kc["mode"]) == ("G", "major")  # E 小调同分：主和弦 G 出现得多，取 G 大调
    assert kc["score"] == 1.0 and kc["chroma_key_score"] == 0.625 and kc["used"] is True


def test_cross_check_keeps_chroma_key_when_close():
    from guitarize.analyze.key import cross_check_key

    segs = _segs(["G", "C", "D", "Em"] + ["D"] * 7)  # G 大调 1.0，D 大调 10/11：差不到 0.1
    kc = cross_check_key(("D", "major"), segs)
    assert kc["score"] - kc["chroma_key_score"] < 0.1 and kc["used"] is False


def test_chosen_key_and_report():
    from guitarize.analyze.key import chosen_key, key_report

    old = {"key": {"tonic": "D", "mode": "major", "r": 0.8}}
    assert chosen_key(old) == ("D", "major")  # 旧 chords.json 没有 key_from_chords
    kc = {"tonic": "G", "mode": "major", "score": 1.0, "chroma_key_score": 0.625, "used": True}
    new = {**old, "key_from_chords": kc}
    assert chosen_key(new) == ("G", "major")
    assert "G major from chords" in key_report(new, overridden=False)
    assert chosen_key({**old, "key_from_chords": {**kc, "used": False}}) == ("D", "major")
