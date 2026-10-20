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
    text = generate_strum(ls, 0, MID, ["pop", "pop"], "T", check=check, rhythm="drums")
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
    tokens, _ = bar_tokens(["E5"] * 4, "D-DU-UDU", {"E5": low}, None, "straight", rhythm="drums")
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


def test_every_bar_shows_its_chord_shape(tmp_path):
    # 用户要求：每小节上方都标出指型（和弦没变也标），小节中间换和弦再标；同一小节内不重复
    ls = leadsheet([[("C", 1, 4)], [("C", 1, 4)], [("C", 1, 2), ("G", 3, 2)]])
    text = generate_strum(ls, 0, MID, ["pop"] * 3, "T")
    bars = [line for line in text.splitlines() if line.endswith("|")]
    assert [b.count('ch "C"') for b in bars] == [1, 1, 1] and bars[2].count('ch "G"') == 1
    assert check_playability(build(text, tmp_path), MID, "strum") == []


# ---------- guitar 节奏（默认）：吉他分轨的节奏型 + 开放和弦的扫法 ----------

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from guitarize.arrange.patterns import POWER_PATTERNS  # noqa: E402
from guitarize.arrange.shapes import Voicing  # noqa: E402

OPEN = {"G": Voicing("G", (3, 0, 0, 0, 2, 3), 0.0), "C": Voicing("C", (0, 1, 0, 2, 3, None), 0.0),
        "D": Voicing("D", (2, 3, 2, 0, None, None), 0.0)}
FULL_G = "(3.1 0.2 0.3 0.4 2.5 3.6)"


def _guitar(chords, pattern, prev=None, feel="shuffle", mute=()):
    from guitarize.arrange.strum import bar_tokens
    return bar_tokens(chords, POWER_PATTERNS.get(pattern, pattern), OPEN, prev, feel, mute, rhythm="guitar")


def test_drums_rhythm_is_the_old_strum_byte_for_byte():
    # strum_rhythm: drums 是改动前的弹唱：鼓 + 吉他选的 PATTERNS、sd / su 箭头、没有力度（fixture 是改动前生成的）
    bars = [[("B", 1, 4)], [("E", 1, 2), ("D#/G", 3, 2)], [("G#m", 1, 4)], [("B/D#", 1, 2), ("C#m7", 3, 2)],
            [("F#7", 1, 4)], [("A", 1, 2), ("N", 3, 2)], [("D#", 1, 4)], [("B", 1, 4)]]
    ls = leadsheet(bars, feel="shuffle")
    text = generate_strum(ls, 4, MID, ["pop", "push", "eighth", "quarter", "half", "whole", "pop", "push"], "Old",
                          rhythm="drums")
    assert text == (Path(__file__).parent / "fixtures" / "strum_drums.atex").read_text(encoding="utf-8")


def test_verse_palm_mutes_the_three_lowest_strings_and_accents_the_open_chord():
    tokens, last = _guitar(["G"] * 4, "verse")
    assert tokens == ['(0.4{pm} 2.5{pm} 3.6{pm}).4 {ch "G" bd 60 dy f}',     # 1：闷低音 3 根弦
                      "(3.1{ac} 0.2{ac} 0.3{ac} 0.4{ac} 2.5{ac} 3.6{ac}).4 {bd 60 dy f}",   # 2：整个和弦 + 重音
                      "(0.4{pm} 2.5{pm} 3.6{pm}).8 {bd 60 dy f}",               # 3：闷音
                      "(3.1 0.2 0.3).8 {bu 60 dy mp}",                          # 3 的反拍：上扫 1–3 弦（闷音之后低音不响）
                      f"{FULL_G}.4 {{bd 60 dy f}}"]                             # 4：下扫
    assert last == "G"


def test_palm_mute_takes_the_lowest_three_sounding_strings_of_the_shape():
    tokens, _ = _guitar(["C", "C", "D", "D"], "M-M-M-M-")
    assert tokens == ['(0.3{pm} 2.4{pm} 3.5{pm}).4 {ch "C" bd 60 dy f}', "(0.3{pm} 2.4{pm} 3.5{pm}).4 {bd 60 dy f}",
                      '(3.2{pm} 2.3{pm} 0.4{pm}).4 {ch "D" bd 60 dy f}', "(3.2{pm} 2.3{pm} 0.4{pm}).4 {bd 60 dy f}"]


def test_up_strum_lets_the_bass_of_the_down_strum_ring():
    # verse_end：1、2（重音）、2 的反拍上扫 1–3 弦，低音弦接着响（连音线），一直响到小节末
    tokens, _ = _guitar(["G"] * 4, "verse_end", prev="G")
    assert tokens == [f'{FULL_G}.4 {{ch "G" bd 60 dy f}}',
                      "(3.1{ac} 0.2{ac} 0.3{ac} 0.4{ac} 2.5{ac} 3.6{ac}).8 {bd 60 dy f}",
                      "(3.1 0.2 0.3 -.4 -.5 -.6).8 {bu 60 dy mp}",
                      "(-.1 -.2 -.3 -.4 -.5 -.6).2"]
    # 旧的型在 guitar 节奏里同样写刷弦和力度：连着的上扫都让低音接着响
    tokens, _ = _guitar(["G"] * 4, "D-DU-UDU", prev="G", feel="straight")
    assert tokens == [f'{FULL_G}.4 {{ch "G" bd 60 dy f}}', f"{FULL_G}.8 {{bd 60 dy f}}",
                      "(3.1 0.2 0.3 -.4 -.5 -.6).4 {bu 60 dy mp}", "(3.1 0.2 0.3 -.4 -.5 -.6).8 {bu 60 dy mp}",
                      f"{FULL_G}.8 {{bd 60 dy f}}", "(3.1 0.2 0.3 -.4 -.5 -.6).8 {bu 60 dy mp}"]


def test_chorus_light_strums_and_the_hit():
    tokens, _ = _guitar(["G"] * 4, "chorus", prev="G")
    accent = "(3.1{ac} 0.2{ac} 0.3{ac} 0.4{ac} 2.5{ac} 3.6{ac})"
    ties = "(-.1 -.2 -.3 -.4 -.5 -.6)"
    assert tokens == [f'{accent}.4 {{ch "G" bd 60 dy f}}', f"{FULL_G}.4 {{bd 60 dy mp}}", f"{ties}.8",
                      f"{FULL_G}.8 {{bd 60 dy mp}}", f"{ties}.4"]
    tokens, _ = _guitar(["G"] * 4, "hit", prev="G")
    assert tokens == [f'{accent}.1 {{ch "G" bd 60 dy f}}']


def test_up_strum_holds_the_bass_of_its_own_chord_only():
    # 换和弦后接着响的是新和弦的低音（C 指型 4、5 弦）；改成闷音的下扫之后，上扫没有低音可接
    tokens, last = _guitar(["G", "G", "C", "C"], "D-DU-UDU", feel="straight")
    assert tokens[2:] == ["(3.1 0.2 0.3 -.4 -.5 -.6).8 {bu 60 dy mp}", '(0.1 1.2 0.3 2.4 3.5).8 {ch "C" bd 60 dy f}',
                          "(0.1 1.2 0.3 -.4 -.5).8 {bu 60 dy mp}", "(0.1 1.2 0.3 2.4 3.5).8 {bd 60 dy f}",
                          "(0.1 1.2 0.3 -.4 -.5).8 {bu 60 dy mp}"] and last == "C"
    tokens, _ = _guitar(["G"] * 4, "D-DU-UDU", feel="straight", mute={2})
    assert tokens[1:4] == ["(x.1 x.2 x.3 x.4 x.5 x.6).8 {bd 60 dy f}", "(3.1 0.2 0.3).4 {bu 60 dy mp}",
                           "(3.1 0.2 0.3).8 {bu 60 dy mp}"]


GUITAR_PROGRESSION = [[("B", 1, 4)], [("E", 1, 2), ("D#/G", 3, 2)], [("G#m", 1, 4)], [("B/D#", 1, 2), ("C#m7", 3, 2)],
                      [("F#7", 1, 4)], [("A", 1, 2), ("N", 3, 2)], [("D#", 1, 4)], [("B", 1, 4)]]


def _guitar_song(tmp_path, patterns=None, profile=MID, check=None):
    ls = leadsheet(GUITAR_PROGRESSION, feel="shuffle")
    ls["bars"][4]["section"] = "B"
    patterns = patterns or ["chorus"] * 4 + ["verse", "verse", "verse_end", "hit"]
    text = generate_strum(ls, 4, profile, patterns, "T", check=check)
    return ls, text, build(text, tmp_path)


def test_guitar_rhythm_builds_plays_and_verifies(tmp_path):
    ls, text, score = _guitar_song(tmp_path)
    assert "\\capo 4" in text and '\\chord "G" 3 0 0 0 2 3' in text and " sd" not in text and " su" not in text
    assert "// bar 1 [A] · 节奏型 chorus" in text and "// bar 5 [B] · 节奏型 verse" in text
    assert "// bar 7 · 节奏型 verse_end" in text and "// bar 8 · 节奏型 hit" in text and "// bar 2 ·" not in text
    assert check_playability(score, MID, "strum") == []
    bars = [line for line in text.splitlines() if line.endswith("|")]
    assert all('ch "' in b for b in bars)                                       # 每小节都标指型
    attacks = score.attacks()
    muted = [a for a in attacks if a.palm_mute]
    assert muted and all(a.bar in (4, 5) for a in muted)
    for a in muted:   # 闷音只扫指型最低的 3 根发声弦：小节里后面那一下整个和弦的下扫的最粗 3 根弦
        full = next(b for b in attacks if b.start > a.start and b.direction == "down" and not b.palm_mute)
        assert sorted(n.string for n in a.notes) == sorted(n.string for n in full.notes)[-3:]
    ups = [a for a in attacks if a.direction == "up"]
    assert ups and all({n.string for n in a.notes} == {1, 2, 3} and a.velocity == 63 for a in ups)
    assert max(a.velocity for a in attacks) == 111 and {a.direction for a in attacks} == {"down", "up"}
    assert all(e.brush in ("down", "up") for e in score.events if not e.rest and any(not n.tie for n in e.notes))
    # verse_end 第 7 小节：2 的重音下扫之后，反拍上扫重扫 1–3 弦，低音弦接着响到小节末
    ends = sorted(round(s.end, 3) for s in score.sounding() if abs(s.start - (6 * 4 + 1)) < 1e-6)
    assert ends.count(round(6 * 4 + 5 / 3, 3)) == 3 and ends.count(28.0) >= 2 and len(ends) >= 5
    s = metrics.summarize(metrics.score_bars(score, ls, "strum"), [], Thresholds(), "strum")
    assert s["chord_agreement"] >= 0.9 and s["chord_mark_issues"] == 0


def test_guitar_rhythm_is_playable_at_every_capo_for_beginners(tmp_path):
    ls = leadsheet(GUITAR_PROGRESSION, feel="shuffle")
    for capo in range(8):
        check, _ = _check(BEG, tmp_path / str(capo))
        text = generate_strum(ls, capo, BEG, ["verse", "verse", "chorus", "verse_end", "pop", "hit", "verse", "push"],
                              "T", check=check)
        assert check_playability(build(text, tmp_path, f"g{capo}"), BEG, "strum") == [], capo


def test_guitar_rhythm_mutes_before_an_unavoidable_shift(tmp_path, monkeypatch):
    from guitarize.arrange import strum
    monkeypatch.setattr(strum, "voicings", _fake_voicings({"X": (6, 8, 8, 8, 6, None), "Y": (0, 1, 0, 2, 3, None)}))
    ls = leadsheet([[("X", 1, 4)], [("Y", 1, 2), ("X", 3, 2)], [("Y", 1, 4)]], feel="shuffle")
    text = generate_strum(ls, 0, MID, ["pop"] * 3, "T")
    score = build(text, tmp_path)
    assert check_playability(score, MID, "strum") == []
    dead = [a for a in score.attacks() if all(n.dead for n in a.notes)]
    assert dead and all(a.direction == "up" and a.velocity == 63 for a in dead)


def test_strum_rhythm_and_pattern_overrides():
    from guitarize.config import strum_pattern, strum_rhythm
    assert strum_rhythm({}) == "guitar" and strum_rhythm({"strum_rhythm": "drums"}) == "drums"
    with pytest.raises(ValueError, match="strum_rhythm"):
        strum_rhythm({"strum_rhythm": "bass"})
    # guitar 节奏：两套型都能整首指定（旧的型也写刷弦和力度）；drums 节奏只有旧的型
    assert strum_pattern({"strum_pattern": "verse"}) == "verse"
    assert strum_pattern({"strum_pattern": "quarter"}) == "quarter"
    assert strum_pattern({"strum_pattern": "quarter", "strum_rhythm": "drums"}) == "quarter"
    with pytest.raises(ValueError, match="strum_pattern"):
        strum_pattern({"strum_pattern": "verse", "strum_rhythm": "drums"})


def _overrides(paths, **values):
    import yaml
    d = yaml.safe_load(paths.config.read_text(encoding="utf-8"))
    d["overrides"] = values
    paths.config.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")


def test_arrange_strum_rhythm_guitar_by_default_drums_on_request(demo_song, capsys):
    from guitarize.cli import main

    songs = str(demo_song.root.parent)
    assert main(["--songs-dir", songs, "arrange", "demo", "--arr", "strum", "--force"]) == 0
    assert "自检：0 处违规" in capsys.readouterr().out
    text = demo_song.atex("strum").read_text(encoding="utf-8")
    assert "bd 60 dy f" in text and " sd" not in text and "· 节奏型 verse" in text   # 没有吉他分轨：默认型
    _overrides(demo_song, strum_pattern="quarter")                                  # 旧的型：整首同一个，照样刷弦
    assert main(["--songs-dir", songs, "arrange", "demo", "--arr", "strum", "--force"]) == 0
    text = demo_song.atex("strum").read_text(encoding="utf-8")
    assert "// bar 1 · 节奏型 quarter" in text and "bd 60 dy f" in text and "bu 60" not in text
    _overrides(demo_song, strum_rhythm="drums")
    assert main(["--songs-dir", songs, "arrange", "demo", "--arr", "strum", "--force"]) == 0
    text = demo_song.atex("strum").read_text(encoding="utf-8")
    assert " sd}" in text and "bd 60" not in text and "节奏型" not in text and "dy " not in text


def test_verify_strum_guitar_rhythm_against_the_guitar_stem_only(demo_song):
    import json
    import shutil

    from guitarize.cli import main
    from guitarize.render.synth import midi_to_wav

    songs = str(demo_song.root.parent)
    assert main(["--songs-dir", songs, "arrange", "demo", "--arr", "strum", "--force"]) == 0
    # 拿编配自己的合成当"吉他分轨"（没有鼓分轨）：guitar 节奏只对照吉他分轨、按力度加权
    built = alphatab.build(demo_song.atex("strum"), demo_song.build_dir, "strum")
    demo_song.stem("guitar").parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(midi_to_wav(built["mid"], demo_song.build("strum", "wav")), demo_song.stem("guitar"))
    assert main(["--songs-dir", songs, "verify", "demo", "--arr", "strum"]) == 0
    it = demo_song.report_root("strum") / "iter-01"
    s = json.loads((it / "scores.json").read_text(encoding="utf-8"))["summary"]
    assert s["violations"] == 0 and s["rhythm_correlation"] > 0.8 and s["rhythm_reference"] == "guitar"
    assert "## 节奏相关度（按段落，对照吉他分轨）" in (it / "report.md").read_text(encoding="utf-8")
    _overrides(demo_song, strum_rhythm="drums")                                     # 旧的弹唱：要鼓 + 吉他分轨
    assert main(["--songs-dir", songs, "verify", "demo", "--arr", "strum"]) == 0
    s = json.loads((demo_song.report_root("strum") / "iter-02" / "scores.json").read_text(encoding="utf-8"))["summary"]
    assert s["rhythm_correlation"] is None and s["rhythm_reference"] == "drums+guitar"
