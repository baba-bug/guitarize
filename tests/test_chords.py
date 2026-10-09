import numpy as np

from guitarize import theory
from guitarize.analyze.chords import (LABELS, align_webchart, reference_chords, same_chord,
                                      template_track, vote_beat)
from guitarize.analyze.webchart import WebBar


def vec(label):
    v = np.zeros(12)
    v[list(theory.chord_tones(label))] = 1
    return v / np.linalg.norm(v)


def beats_cb(labels, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    cb = np.stack([vec(l) + noise * rng.random(12) for l in labels], axis=1)
    return cb / np.linalg.norm(cb, axis=0, keepdims=True)


def test_labels_order_matches_theory():
    assert [l.replace("m", "") + (":min" if l.endswith("m") else ":maj") for l in LABELS] == theory.MAJMIN_LABELS


def test_template_track_recovers_progression():
    truth = ["C"] * 4 + ["G"] * 4 + ["Am"] * 4 + ["F"] * 4
    labels, post = template_track(beats_cb(truth))
    assert labels == truth
    assert (post > 0.7).mean() > 0.8


def test_align_webchart_with_jump_and_transpose():
    web = [WebBar([(c, 1.0)]) for c in ["C", "G", "Am", "F", "C", "G", "F", "C"]]
    audio = np.stack([vec(c) for c in ["C", "G", "Am", "F", "F", "C"]])
    out = align_webchart(audio, web)
    assert out["index"] == [0, 1, 2, 3, 6, 7] and out["transpose"] == 0
    up2 = np.stack([vec(c) for c in ["D", "A", "Bm", "G", "G", "D"]])
    out2 = align_webchart(up2, web)
    assert out2["transpose"] == 2 and out2["index"] == [0, 1, 2, 3, 6, 7]


def test_same_chord_and_vote():
    assert same_chord("G#m", "G#5") and same_chord("B", "B/D#") and not same_chord("Em", "D#/G")
    assert vote_beat("G#m", 0.9, "G#5")[:2] == ("G#m", "high")
    assert vote_beat("B", 0.9, "B/D#")[:2] == ("B/D#", "high")
    assert vote_beat("Em", 0.9, "D#/G")[:2] == ("D#/G", "low")
    assert vote_beat("C", 0.8, None)[:2] == ("C", "medium")
    assert vote_beat("C", 0.5, None)[:2] == ("C", "low")
    assert vote_beat("C", 0.9, "N")[:2] == ("N", "medium")
    assert vote_beat("Em", 0.9, "D#/G")[2] == {"template": "Em", "web": "D#/G"}


def test_reference_chords_end_to_end():
    truth = ["C"] * 4 + ["G"] * 4 + ["Am"] * 4 + ["F"] * 4
    cb = beats_cb(truth)
    beats_json = {"time_signature": [4, 4],
                  "beats": [{"t": 0.5 * i, "bar": i // 4 + 1, "beat": i % 4 + 1, "g": i} for i in range(16)]}
    web = [WebBar([(c, 1.0)]) for c in ["C", "G", "Am", "F"]]
    out = reference_chords(cb, beats_json, web)
    segs = [(s["bar"], s["beat"], s["dur_beats"], s["label"], s["confidence"]) for s in out["segments"]]
    assert segs == [(1, 1, 4, "C", "high"), (2, 1, 4, "G", "high"), (3, 1, 4, "Am", "high"), (4, 1, 4, "F", "high")]
    assert [a["web_index"] for a in out["web_alignment"]] == [0, 1, 2, 3] and out["web_transpose"] == 0
    no_web = reference_chords(cb, beats_json)
    assert "web_alignment" not in no_web
    assert {s["confidence"] for s in no_web["segments"]} <= {"medium", "low"}


def test_all_n_web_bar_accepted_when_consecutive():
    cb = beats_cb(["C"] * 8 + ["F"] * 4)
    beats_json = {"time_signature": [4, 4],
                  "beats": [{"t": 0.5 * i, "bar": i // 4 + 1, "beat": i % 4 + 1, "g": i} for i in range(12)]}
    web = [WebBar([("C", 1.0)]), WebBar([("N", 1.0)]), WebBar([("F", 1.0)])]
    out = reference_chords(cb, beats_json, web)
    by_bar = {s["bar"]: s for s in out["segments"]}
    assert by_bar[2]["label"] == "N" and by_bar[2]["confidence"] == "medium"
    assert by_bar[2]["votes"]["web"] == "N"
    assert by_bar[1]["confidence"] == "high" and by_bar[3]["confidence"] == "high"
