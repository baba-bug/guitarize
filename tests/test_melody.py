import numpy as np
import pytest

from guitarize.analyze.melody import fix_octaves, quantize, snap_to_key

C_MAJOR = {0, 2, 4, 5, 7, 9, 11}


def test_snap_to_key():
    assert snap_to_key(61.3, C_MAJOR) == (62, False)
    assert snap_to_key(64.2, C_MAJOR) == (64, False)
    assert snap_to_key(61.0, C_MAJOR) == (61, True)


def frames(segments, per_beat=60):
    """segments: [(start_beat, end_beat, midi or None)] → 逐帧拍位置与音高。"""
    pos, midi = [], []
    for a, b, m in segments:
        n = int((b - a) * per_beat)
        pos += list(a + np.arange(n) / per_beat)
        midi += [np.nan if m is None else m] * n
    return np.array(pos), np.array(midi, dtype=float)


def test_quantize_merges_slots_and_splits_on_onsets():
    pos, midi = frames([(0, 1, 60.1), (1, 2, 62.0), (2, 3, None), (3, 4, 64.0), (4, 6, 60.0)])
    per = np.ones_like(midi)
    notes = quantize(pos, midi, per, onset_beats=np.array([0.0, 1.0, 3.0, 4.0, 5.0]), key_pcs=C_MAJOR, slot=1 / 3)
    got = [(n["start_beats"], n["dur_beats"], n["midi"]) for n in notes]
    assert got == pytest.approx([(0, 1, 60), (1, 1, 62), (3, 1, 64), (4, 1, 60), (5, 1, 60)], abs=1e-3)
    assert all(n["out_of_key"] is False and n["conf"] == 1.0 for n in notes)


def test_fix_octaves():
    notes = [{"midi": m} for m in [60, 62, 76, 64, 65]]
    assert [n["midi"] for n in fix_octaves(notes)] == [60, 62, 64, 64, 65]


def test_quantize_empty_input_returns_no_notes():
    empty = np.array([], dtype=float)
    assert quantize(empty, empty, empty, empty, C_MAJOR, slot=1 / 4) == []
