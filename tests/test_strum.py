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


BEG = load_profile("beginner")


def _check(profile, tmp_path):
    """真实校验：alphaTab 构建 → load_score → check_playability（与 verify 相同）。"""
    calls = []
    tmp_path.mkdir(parents=True, exist_ok=True)

    def check(text):
        calls.append(text)
        return check_playability(build(text, tmp_path, f"c{len(calls)}"), profile, "strum")
    return check, calls


def test_voicings_chosen_for_transitions_not_alone(tmp_path):
    # 变调夹 0：D# 单独看最好是 6 品横按，接开放 E（1 品）在 1 拍内换把 5 品 → 必须换个组合
    ls = leadsheet([[("D#", 1, 4)], [("E", 1, 4)]] * 3, feel="shuffle")
    text = generate_strum(ls, 0, MID, ["pop"] * 6, "T")
    score = build(text, tmp_path)
    assert check_playability(score, MID, "strum") == []
    assert "x." not in text                      # 靠选指型解决，没有闷音
    assert text.count("\\chord ") == 2           # 每个和弦一个指型、一张和弦图


def _fake_voicings(table):
    from guitarize.arrange.shapes import Voicing
    return lambda label, profile: [Voicing(label, table[label], 0.0)]


def test_unavoidable_shift_mutes_offbeat_before_change(tmp_path, monkeypatch):
    from guitarize.arrange import strum
    monkeypatch.setattr(strum, "voicings", _fake_voicings({"X": (6, 8, 8, 8, 6, None), "Y": (0, 1, 0, 2, 3, None)}))
    ls = leadsheet([[("X", 1, 4)], [("Y", 1, 2), ("X", 3, 2)], [("Y", 1, 4)]], feel="shuffle")
    text = generate_strum(ls, 0, MID, ["pop"] * 3, "T")
    score = build(text, tmp_path)
    assert check_playability(score, MID, "strum") == []
    dead = sorted(round(a.start, 3) for a in score.attacks() if all(n.dead for n in a.notes))
    assert dead == [3.667, 5.667, 7.667]          # 每次换和弦前 1/3 拍的反拍上扫（演奏时刻）
    for t in (3.0, 5.0, 7.0):                    # 换和弦前恰好 1 拍的正拍仍然发声
        a = next(a for a in score.attacks() if abs(a.start - t) < 1e-6)
        assert not any(n.dead for n in a.notes)
    ups = [a for a in score.attacks() if abs(a.start - 3.667) < 1e-3]
    assert ups[0].direction == "up" and [n.string for n in ups[0].notes] == [1, 2, 3]


def test_check_driven_repair_mutes_strums_before_violation(tmp_path):
    from guitarize.verify.playability import Violation
    ls = leadsheet([[("G", 1, 4)], [("C", 1, 4)]])
    calls = []

    def check(text):
        calls.append(text)
        return [Violation(2, 4.0, "shift", "fake")] if len(calls) == 1 else []
    text = generate_strum(ls, 0, MID, ["pop", "pop"], "T", check=check)
    assert len(calls) == 2 and "x." not in calls[0]
    bar1 = text.split("// bar 1")[1].split("// bar 2")[0]
    assert bar1.strip().endswith("(x.1 x.2 x.3 x.4 x.5 x.6).8 {sd} (x.1 x.2 x.3).8 {su} |")
    assert bar1.count("x.1") == 2
    assert "x." not in text.split("// bar 2")[1]


def test_repair_loop_is_bounded():
    from guitarize.verify.playability import Violation
    ls = leadsheet([[("G", 1, 4)]] * 8)
    calls = []

    def check(text):
        calls.append(text)
        return [Violation(len(calls) + 1, 4.0 * len(calls), "shift", "fake")]
    generate_strum(ls, 0, MID, ["pop"] * 8, "T", check=check)
    assert len(calls) == 5


def test_up_strum_falls_back_to_top_three_sounding_strings(tmp_path):
    from guitarize.arrange.shapes import Voicing
    from guitarize.arrange.strum import bar_tokens
    low = Voicing("E5", (None, None, None, 2, 2, 0), 0.0)
    tokens, _ = bar_tokens(["E5"] * 4, "D-DU-UDU", {"E5": low}, None, "straight")
    ups = [t for t in tokens if "su" in t]
    assert ups and all(t.startswith("(2.4 2.5 0.6)") for t in ups)
    head = "\\title \"T\"\n\\tempo 120\n\\track \"Guitar\"\n\\staff {tabs}\n\\tuning (E4 B3 G3 D3 A2 E2)\n\\ts 4 4\n"
    score = build(head + " ".join(tokens) + " |\n", tmp_path, "up")
    assert all(len(a.notes) == 3 for a in score.attacks())


def test_beginner_b_major_song_at_every_capo(tmp_path):
    bars = [[("B", 1, 4)], [("E", 1, 2), ("F#", 3, 2)], [("G#m", 1, 4)], [("D#m", 1, 2), ("C#m7", 3, 2)],
            [("B/D#", 1, 2), ("E", 3, 2)], [("F#7", 1, 4)], [("D#", 1, 4)], [("B", 1, 4)]]
    ls = leadsheet(bars, feel="shuffle")
    for capo in range(8):
        check, _ = _check(BEG, tmp_path / str(capo))
        text = generate_strum(ls, capo, BEG, ["pop"] * len(bars), "T", check=check)
        assert check_playability(build(text, tmp_path, f"beg{capo}"), BEG, "strum") == [], capo
