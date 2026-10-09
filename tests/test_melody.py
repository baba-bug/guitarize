import numpy as np
import pytest

from guitarize.analyze.melody import fix_octaves, merge_glides, quantize, snap_to_key

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


def test_fix_octaves_keeps_legitimate_leap():
    notes = [{"midi": m} for m in [60, 60, 60, 70, 60, 60, 60]]
    assert [n["midi"] for n in fix_octaves(notes)] == [60, 60, 60, 70, 60, 60, 60]


def N(start, dur, midi, out=False):
    return {"start_beats": start, "dur_beats": dur, "midi": midi, "out_of_key": out, "conf": 0.9}


def _sd(notes):
    return [(n["start_beats"], n["dur_beats"], n["midi"]) for n in notes]


def test_merge_glides_scoop_into_next():
    got = merge_glides([N(1.0, 0.3333, 67, True), N(1.3333, 0.6667, 68)])
    assert _sd(got) == [(1.0, 1.0, 68)]
    assert got[0]["conf"] == 0.9


def test_merge_glides_fall_off_into_previous():
    got = merge_glides([N(2.0, 1.0, 66), N(3.0, 0.3333, 67, True)])
    assert _sd(got) == [(2.0, 1.3333, 66)]


def test_merge_glides_prefers_next():
    got = merge_glides([N(1.0, 0.3333, 66), N(1.3333, 0.3333, 67, True), N(1.6667, 0.3333, 68)])
    assert _sd(got) == [(1.0, 0.3333, 66), (1.3333, 0.6667, 68)]


def test_merge_glides_not_adjacent_unchanged():
    notes = [N(0.0, 0.3333, 66), N(0.6667, 0.3333, 67, True), N(1.3333, 0.3333, 68)]
    assert merge_glides(notes) == notes


def test_merge_glides_too_far_unchanged():
    notes = [N(1.0, 0.3333, 62, True), N(1.3333, 0.6667, 70)]
    assert merge_glides(notes) == notes


def test_merge_glides_long_note_unchanged():
    notes = [N(1.0, 0.6667, 67, True), N(1.6667, 0.6667, 68)]
    assert merge_glides(notes) == notes


def test_merge_glides_out_of_key_neighbour_does_not_absorb():
    notes = [N(1.0, 0.3333, 67, True), N(1.3333, 0.3333, 68, True)]
    assert merge_glides(notes) == notes


def test_merge_glides_empty():
    assert merge_glides([]) == []


def test_merge_glides_does_not_mutate_input():
    notes = [N(1.0, 0.3333, 67, True), N(1.3333, 0.6667, 68)]
    snapshot = [dict(n) for n in notes]
    merge_glides(notes)
    assert notes == snapshot


def test_tidy_notes_merges_glides_when_key_fits():
    from guitarize.analyze.melody import tidy_notes

    notes = [N(1.0, 0.3333, 67, True), N(1.3333, 0.6667, 68), N(2.0, 1.0, 64), N(3.0, 1.0, 65), N(4.0, 1.0, 64)]
    assert _sd(tidy_notes(notes))[0] == (1.0, 1.0, 68)


def test_tidy_notes_skips_glide_merge_when_key_looks_wrong(capsys):
    from guitarize.analyze.melody import tidy_notes

    notes = [N(1.0, 0.3333, 67, True), N(1.3333, 0.6667, 68), N(2.0, 1.0, 61, True), N(3.0, 1.0, 65)]
    assert tidy_notes(notes) == notes  # 2/4 = 50% 调外 > 30%：调性多半错了，不合并
    assert "out of key" in capsys.readouterr().err


def test_tidy_notes_drops_notes_before_bar_one():
    from guitarize.analyze.melody import tidy_notes

    assert _sd(tidy_notes([N(-1.0, 1.0, 64), N(0.0, 1.0, 65)])) == [(0.0, 1.0, 65)]


def test_quantize_and_merge_never_overlap_on_triplet_grid():
    # 三连音格上起点与时值分别四舍五入会累积误差（0.6667 + 0.6667 = 1.3334 > 1.3333）：
    # 时值必须取 round(end) - round(start)，合并滑音后也不能与相邻音重叠
    slot_pitch = [None, None, 64, 64, 61, 62, 62, 64, 64]  # 每格音高；61 = C#，C 大调外
    pos = np.arange(len(slot_pitch) * 20) / 60
    midi = np.array([np.nan if slot_pitch[i // 20] is None else slot_pitch[i // 20] for i in range(len(pos))])
    notes = quantize(pos, midi, np.ones_like(midi), np.array([]), C_MAJOR, slot=1 / 3)
    merged = merge_glides(notes)
    for got in (notes, merged):
        for a, b in zip(got, got[1:]):
            assert a["start_beats"] + a["dur_beats"] <= b["start_beats"] + 1e-9
    assert _sd(merged) == [(0.6667, 0.6666, 64), (1.3333, 1.0, 62), (2.3333, 0.6667, 64)]


def test_merge_glides_fall_off_ends_where_glide_ended():
    # 前一个音的终点与滑音起点只在 1e-3 容差内相接：合并后的终点取滑音的终点，而不是两段时值相加
    got = merge_glides([N(2.0, 0.6667, 66), N(2.6663, 0.3337, 67, True), N(3.0, 1.0, 71)])
    assert _sd(got) == [(2.0, 1.0, 66), (3.0, 1.0, 71)]
