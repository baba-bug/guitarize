import numpy as np

from guitarize.arrange.patterns import accent_vector, choose_pattern
from guitarize.arrange.strum import generate_strum
from guitarize.config import Thresholds, load_profile
from guitarize.render import alphatab
from guitarize.verify import metrics
from guitarize.verify.model import load_score
from guitarize.verify.playability import check_playability

MID = load_profile("intermediate")


def leadsheet(bars, feel="straight", tempo=120.0):
    out = []
    for i, chords in enumerate(bars):
        out.append({"bar": i + 1, "section": "A" if i == 0 else None, "t_start": i * 2.0, "t_end": (i + 1) * 2.0,
                    "chords": [{"beat": b, "dur_beats": d, "label": l, "confidence": "high"} for l, b, d in chords],
                    "melody": []})
    return {"key": "B", "mode": "major", "tempo_bpm": tempo, "feel": feel, "beats_per_bar": 4, "bars": out}


def build(text, tmp_path, name="s"):
    src = tmp_path / f"{name}.atex"
    src.write_text(text, encoding="utf-8")
    return load_score(alphatab.build(src, tmp_path / "b", name)["json"])


def test_accent_vector_and_choice():
    v = accent_vector("D-DU-UDU", "shuffle")
    assert v.shape == (12,) and list(np.flatnonzero(v)) == [0, 3, 5, 8, 9, 11]
    assert choose_pattern(accent_vector("D-DU-UDU", "shuffle"), "shuffle") == "pop"
    assert choose_pattern(None, "shuffle") == "push"
    assert choose_pattern(np.zeros(12), "shuffle") == "push"


def test_generate_strum_capo4_b_major(tmp_path):
    ls = leadsheet([[("B", 1, 4)], [("E", 1, 2), ("D#/G", 3, 2)], [("G#m", 1, 4)], [("N", 1, 4)]], feel="shuffle")
    text = generate_strum(ls, 4, MID, ["push", "pop", "push", "push"], "Test")
    assert '\\chord "G"' in text and '\\chord "B7/D#"' in text and "\\capo 4" in text
    assert "// bar 1 [A]" in text
    score = build(text, tmp_path)
    assert check_playability(score, MID, "strum") == []
    bars = metrics.score_bars(score, ls, "strum")
    s = metrics.summarize(bars, [], Thresholds(), "strum")
    assert s["chord_agreement"] == 1.0 and s["chord_mark_issues"] == 0
    attacks = [a for a in score.attacks() if a.bar == 0]
    assert len(attacks) == 6 and attacks[0].direction == "down"
    bar2 = [a for a in score.attacks() if a.bar == 1]
    assert any(abs(a.start - 6.0) < 1e-6 and a.direction == "down" for a in bar2)
    assert not [a for a in score.attacks() if a.bar == 3]
