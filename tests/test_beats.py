import numpy as np
import pytest

from guitarize.analyze.beats import assemble_beats, bar_positions, beat_to_time, clean_beats, time_to_beat


def test_clean_beats_fills_and_drops():
    out = clean_beats([0, 0.5, 1.0, 2.0, 2.1, 2.5])
    assert out.tolist() == pytest.approx([0, 0.5, 1.0, 1.5, 2.0, 2.5])


def test_bar_positions_with_pickup():
    ev = np.zeros(16)
    ev[[3, 7, 11, 15]] = 1.0
    assert bar_positions(ev) == [1, 2, 3, 0] * 4


def test_bar_positions_allows_strong_short_bar():
    ev = np.zeros(14)
    ev[[0, 4, 8, 10]] = 5.0
    pos = bar_positions(ev)
    assert pos[8] == 0 and pos[9] == 1 and pos[10] == 0 and pos[13] == 3


def test_assemble_beats_pickup_bar_is_padded():
    times = [0.1 + 0.5 * k for k in range(11)]
    bj = assemble_beats(times, [1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], "shuffle")
    assert bj["tempo_bpm"] == 120.0 and bj["pad_beats"] == 1 and bj["pickup_beats"] == 3
    assert bj["beats"][0] == {"t": 0.1, "bar": 1, "beat": 2, "g": 1}
    assert bj["beats"][3] == {"t": 1.6, "bar": 2, "beat": 1, "g": 4}
    assert bj["bars"][0] == {"bar": 1, "t_start": -0.4, "t_end": 1.6, "beats": 3}
    assert bj["bars"][1] == {"bar": 2, "t_start": 1.6, "t_end": 3.6, "beats": 4}
    assert bj["short_bars"] == [] and bj["feel"] == "shuffle"


def test_time_beat_mapping_roundtrip_and_extrapolation():
    times = [0.1 + 0.5 * k for k in range(11)]
    bj = assemble_beats(times, [1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], "straight")
    assert float(time_to_beat(bj, 1.6)) == pytest.approx(4.0)
    assert float(beat_to_time(bj, 0.0)) == pytest.approx(-0.4)
    assert float(time_to_beat(bj, beat_to_time(bj, 7.25))) == pytest.approx(7.25)
    assert time_to_beat(bj, np.array([0.1, 0.35])).tolist() == pytest.approx([1.0, 1.5])
