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
