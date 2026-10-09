import numpy as np

from guitarize import theory
from guitarize.analyze.chords import (LABELS, align_webchart, power_quality_table, reference_chords,
                                      resolve_power, template_track, vote_beat)
from guitarize.analyze.key import scale_pcs
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


def test_vote_beat():
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


def test_power_quality_table_and_resolve():
    web = [WebBar([(l, 1.0)]) for l in ["B", "E", "G#m", "C#", "C#m7", "C#7", "B5"]]
    assert power_quality_table(web) == {11: "", 4: "", 8: "m", 1: ""}
    major = scale_pcs("B", "major")
    assert resolve_power("G#5", {}, major, "G#") == "G#m"
    assert resolve_power("E5", {4: ""}, None, "Em") == "E"
    assert resolve_power("A5", {}, major, "Am") == "Am"


def test_vote_quality_disagreement_is_medium():
    assert vote_beat("Bm", 0.9, "B5", {11: ""}, None)[:2] == ("B", "medium")
    assert vote_beat("G#", 0.9, "G#m")[:2] == ("G#m", "medium")


def test_resolve_power_minor_key_and_audio_fallback():
    minor = scale_pcs("G#", "minor")
    major_b = scale_pcs("B", "major")
    assert resolve_power("G#5", {}, minor, "G#") == "G#m"
    assert resolve_power("E5", {}, minor, "Em") == "E"
    assert resolve_power("C#5", {}, minor, "C#") == "C#m"
    assert resolve_power("F5", {}, None, "Fm") == "Fm"
    assert resolve_power("F5", {}, None, "F") == "F"
    assert resolve_power("A5", {}, major_b, "F#m") == "A"
    assert resolve_power("A5", {}, major_b, "Am") == "Am"
    assert resolve_power("F5", {}, None, "Dm") == "F"
    assert resolve_power("F5", {}, None, "N") == "F"
    assert resolve_power("F5", {}, None, "") == "F"


def test_reference_chords_rotates_power_table_with_transpose():
    # 网上谱在 C 调，音频高 2 个半音：Am 的小三性质必须随之移到 B，才能解析 A5 -> Bm
    truth = ["D"] * 4 + ["A"] * 4 + ["Bm"] * 4 + ["B"] * 4
    cb = beats_cb(truth)
    beats_json = {"time_signature": [4, 4],
                  "beats": [{"t": 0.5 * i, "bar": i // 4 + 1, "beat": i % 4 + 1, "g": i} for i in range(16)]}
    web = [WebBar([(c, 1.0)]) for c in ["C", "G", "Am", "A5"]]
    out = reference_chords(cb, beats_json, web)
    assert out["web_transpose"] == 2
    last = [s for s in out["segments"] if s["bar"] == 4]
    assert [(s["label"], s["confidence"]) for s in last] == [("Bm", "medium")]


def test_tail_aligns_to_n_run_not_repeat():
    # 网上谱：段落 + N.C. 段 + 段落重复；音频：段落 + 无和弦的安静尾巴。
    # 全 N 小节的相似度若是 0，DP 会跳去重复段；对齐时按 MIN_ALIGN_SIM 计，尾巴应落在 N 段上。
    sec = ["C", "G", "Am", "F"]
    web = ([WebBar([(c, 1.0)]) for c in sec] + [WebBar([("N", 1.0)]) for _ in range(4)]
           + [WebBar([(c, 1.0)]) for c in sec])
    rng = np.random.default_rng(1)
    tail = np.ones((12, 16)) + 0.2 * rng.random((12, 16))  # 近乎平坦的色度：没有和弦
    cb = np.concatenate([beats_cb([c for c in sec for _ in range(4)]), tail / np.linalg.norm(tail, axis=0)], axis=1)
    beats_json = {"time_signature": [4, 4],
                  "beats": [{"t": 0.5 * i, "bar": i // 4 + 1, "beat": i % 4 + 1, "g": i} for i in range(32)]}
    out = reference_chords(cb, beats_json, web)
    assert [a["web_index"] for a in out["web_alignment"]] == list(range(8))
    assert all(a["sim"] == 0.0 for a in out["web_alignment"][4:])  # 报告的仍是真实相似度
    tail_segs = [s for s in out["segments"] if s["bar"] >= 5]
    assert {(s["label"], s["confidence"]) for s in tail_segs} == {("N", "medium")}
