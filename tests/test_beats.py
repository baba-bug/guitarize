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


def test_positions_with_shift_keeps_strict_cycle():
    from guitarize.analyze.beats import positions_with_shift

    ev = np.zeros(14)
    ev[[0, 4, 8, 10]] = 5.0
    pos = positions_with_shift(ev, shift=1)
    assert all((b - a) % 4 == 1 for a, b in zip(pos, pos[1:]))


def test_positions_with_shift_zero_matches_bar_positions():
    from guitarize.analyze.beats import positions_with_shift

    ev = np.zeros(14)
    ev[[0, 4, 8, 10]] = 5.0
    assert positions_with_shift(ev, shift=0) == bar_positions(ev)


def test_short_bar_reported_when_no_pickup():
    bj = assemble_beats([0.5 * k for k in range(11)], [0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 0], "straight")
    assert bj["short_bars"] == [1]


def _bars_after_override(times, activation, override):
    """模拟 track_beats 的顺序：先按 override 纠正拍点，再在纠正后的拍上定小节线。"""
    from guitarize.analyze.beats import apply_tempo_override

    fixed, notes = apply_tempo_override(times, override, activation)
    ev = np.zeros(len(fixed))
    ev[::4] = 1.0  # 纠正后每 4 拍一个强拍
    return fixed, notes, assemble_beats(fixed, bar_positions(ev), "straight")


def test_tempo_override_halves_double_time_beats():
    true = 0.1 + 0.5 * np.arange(32)               # 120 BPM，8 小节
    detected = np.sort(np.r_[true, true - 0.25])   # 模型报成 240 BPM，真拍在奇数位
    activation = np.where(np.isin(detected, true), 1.0, 0.2)
    fixed, notes, bj = _bars_after_override(detected, activation, 120)
    assert fixed.tolist() == pytest.approx(true.tolist())
    assert bj["tempo_bpm"] == 120.0 and len(bj["bars"]) == 8
    assert notes and "2" in notes[0]


def test_tempo_override_doubles_half_time_beats():
    true = 0.5 * np.arange(32)                     # 120 BPM，8 小节
    detected = true[::2]                           # 模型报成 60 BPM
    fixed, notes, bj = _bars_after_override(detected, None, 120)
    assert fixed.tolist() == pytest.approx(true[:-1].tolist())
    assert bj["tempo_bpm"] == 120.0 and len(bj["bars"]) == 8
    assert notes


def test_tempo_override_small_difference_keeps_beats():
    from guitarize.analyze.beats import apply_tempo_override

    t = 60 / 123.6 * np.arange(40)                 # 比 override 快 3%
    fixed, notes = apply_tempo_override(t, 120, None)
    assert np.array_equal(fixed, t) and notes == []
    t5 = 60 / 126 * np.arange(40)                  # 快 5%：保留拍点但警告
    fixed5, notes5 = apply_tempo_override(t5, 120, None)
    assert np.array_equal(fixed5, t5) and len(notes5) == 1 and "5.0%" in notes5[0]
    assert apply_tempo_override(t, None, None)[0] is t


def test_tempo_override_phase_tie_uses_first_strong_beat():
    from guitarize.analyze.beats import apply_tempo_override

    t = 0.25 * np.arange(16)
    act = np.zeros(16)
    act[[1, 2]] = 1.0                              # 两个相位总和相同：取第一个最强拍（下标 1）所在相位
    fixed, _ = apply_tempo_override(t, 120, act)
    assert fixed.tolist() == pytest.approx(t[1::2].tolist())


def test_tempo_stability_recorded_and_warned():
    from guitarize.analyze.beats import stability_warning

    steady = assemble_beats([0.5 * k for k in range(40)], [k % 4 for k in range(40)], "straight")
    stab = steady["tempo_stability"]
    assert set(stab) == {"local_min_bpm", "local_max_bpm", "cv"}
    assert stab["local_min_bpm"] == stab["local_max_bpm"] == 120.0 and stab["cv"] == 0.0
    assert stability_warning(stab, steady["tempo_bpm"]) is None
    ibi = np.linspace(0.5, 0.42, 63)               # 渐快 120 → 143 BPM
    drift = assemble_beats(np.r_[0, np.cumsum(ibi)], [k % 4 for k in range(64)], "straight")
    s = drift["tempo_stability"]
    assert s["local_max_bpm"] - s["local_min_bpm"] > 10
    assert "unstable" in stability_warning(s, drift["tempo_bpm"])
