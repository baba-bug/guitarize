from conftest import TEX_HEAD
from guitarize.config import load_profile
from guitarize.render import alphatab
from guitarize.verify.model import load_score
from guitarize.verify.playability import check_playability, fingers_needed, soft_stats

PROFILE = load_profile("intermediate")


def rules(score, kind="finger"):
    return [v.rule for v in check_playability(score, PROFILE, kind)]


def test_fingers_needed():
    assert fingers_needed([1, 1, 2, 3, 3, 1]) == 4  # F 大横按
    assert fingers_needed([0, 0, 0]) == 0
    assert fingers_needed([2, 3, 4, 4, 3]) == 5


def test_span_violation(build_tex):
    assert rules(build_tex(":4 (1.1 5.2) r r r |")) == ["span"]


def test_fingers_violation(build_tex):
    assert rules(build_tex(":4 (2.1 3.2 4.3 4.4 3.5) {bd} r r r |")) == ["fingers"]


def test_fast_shift_violation(build_tex):
    assert rules(build_tex(":8 1.1 8.1 :4 r :2 r |")) == ["shift"]


def test_slow_shift_ok(build_tex):
    assert rules(build_tex(":4 1.1 8.1 r r |")) == []


def test_swung_one_beat_gap_is_not_fast_shift(build_tex):
    assert rules(build_tex("\\tf triplet-8th\n:8 r 1.1 r 8.1 :2 r |")) == []


def test_held_note_counts_for_span(build_tex):
    assert rules(build_tex(":4 (1.5 0.1) (-.5 8.1) r r |")) == ["span"]


def test_plucks_only_for_finger_without_strum(build_tex):
    s = build_tex(":4 (0.1 0.2 0.3 2.4 3.5) r r r |")
    assert rules(s, "finger") == ["plucks"]
    assert rules(s, "strum") == []
    assert rules(build_tex(":4 (0.1 0.2 0.3 2.4 3.5) {bd} r r r |"), "finger") == []


def test_max_fret_violation(build_tex):
    assert rules(build_tex(":4 13.1 r r r |")) == ["max_fret"]


def test_bar_length_violation(build_tex):
    assert rules(build_tex(":4 0.1 0.1 0.1 0.1 0.1 |")) == ["bar_length"]


def test_demo_fixtures_clean(demo_song, tmp_path):
    for arr in ("finger", "strum"):
        score = load_score(alphatab.build(demo_song.atex(arr), tmp_path / "b", arr)["json"])
        assert check_playability(score, PROFILE, arr) == []
    stats = soft_stats(score)
    assert stats[2]["barre"] is True and stats[1]["barre"] is False
    assert stats[2]["chord_marks"] == ["F", "C"]


def test_straight_offbeat_in_triplet_feel(build_tex):
    s = build_tex("\\tf triplet-8th\n:8 0.1 :4 1.1 :8 2.1 :2 3.1 |")
    vs = check_playability(s, PROFILE, "finger")
    assert [(v.rule, v.start) for v in vs] == [("straight_offbeat", 0.5), ("straight_offbeat", 1.5)]
    s = build_tex("\\tf triplet-8th\n0.1.4 {d} 1.1.8 2.1.2 |")
    assert [(v.rule, v.start) for v in check_playability(s, PROFILE, "finger")] == [("straight_offbeat", 1.5)]


def test_paired_offbeat_with_tie_is_ok(build_tex):
    assert rules(build_tex("\\tf triplet-8th\n:8 0.1 1.1 2.1 -.1 :2 3.1 |")) == []


def test_straight_feel_has_no_offbeat_rule(build_tex):
    assert rules(build_tex(":8 0.1 :4 1.1 :8 2.1 :2 3.1 |")) == []


def test_same_string_two_voices(build_tex):
    s = build_tex("\\voice\n:1 3.1 |\n\\voice\n:1 5.1 |")
    assert "same_string" in rules(s)


def test_max_fret_relative_to_neck_with_capo(build_tex):
    head = TEX_HEAD + "\\capo 4\n"
    s = build_tex(":4 11.1 r r r |", head=head)
    vs = check_playability(s, PROFILE, "finger")
    assert [v.rule for v in vs] == ["max_fret"] and "变调夹 4" in vs[0].detail
    assert rules(build_tex(":4 10.1 r r r |", head=head)) == []
