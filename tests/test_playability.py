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
