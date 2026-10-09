from guitarize.arrange.fretting import FretEvent, _position, _transition, assign_frets
from guitarize.config import load_profile

MID = load_profile("intermediate")


def test_first_position_melody_with_bass():
    evs = assign_frets([FretEvent(0, 59, 43), FretEvent(1, 60, 48), FretEvent(2, 62, None)], MID)
    assert [(e.melody_pos, e.bass_pos) for e in evs] == [((2, 0), (6, 3)), ((2, 1), (5, 3)), ((2, 3), None)]
    assert all(e.fallback is None for e in evs)


def test_unplayable_event_falls_back_without_crashing():
    ev = assign_frets([FretEvent(0, 71, 43)], MID)[0]
    assert ev.fallback == "bass+12" and ev.melody_pos == (1, 7) and ev.bass_pos == (3, 0)


def test_bass_above_melody_is_dropped_an_octave():
    ev = assign_frets([FretEvent(0, 57, 60)], MID)[0]
    assert ev.melody_pos is not None and ev.bass_pos is not None
    m_string, b_string = ev.melody_pos[0], ev.bass_pos[0]
    assert m_string < b_string


def test_fill_sits_between_melody_and_bass():
    ev = assign_frets([FretEvent(0, 66, 43, fill=59)], MID)[0]
    assert ev.fallback is None
    assert ev.melody_pos == (1, 2) and ev.fill_pos == (2, 0) and ev.bass_pos == (6, 3)


def test_fill_is_kept_when_bass_moves_up_an_octave():
    ev = assign_frets([FretEvent(0, 71, 43, fill=59)], MID)[0]
    assert ev.fallback == "bass+12"
    assert ev.melody_pos == (1, 7) and ev.fill_pos == (2, 0) and ev.bass_pos == (3, 0)


def test_melody_prefers_top_three_strings():
    evs = assign_frets([FretEvent(t, p, None) for t, p in enumerate([55, 57, 59, 60])], MID)
    assert all(e.melody_pos[0] <= 3 for e in evs)


def test_hand_box_intervals_and_transition():
    assert _position(((6, 3),), 3) == (1, 3)
    assert _position(((2, 1), (5, 3)), 3) == (1, 1)
    assert _position(((1, 0), (2, 0)), 3) is None
    assert _transition(((6, 3),), ((2, 1), (5, 3)), 1.0, 3) == 0
    assert _position(((2, 8),), 3) == (5, 8)
    assert _transition(((2, 1),), ((2, 8),), 1.0, 3) == 8.0
