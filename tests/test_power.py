"""强力和弦弹唱（arr = power）：实际音高（不夹变调夹）的强力和弦、按吉他分轨选的分段节奏型（闷音 / 刷弦 / 力度）、
校验按根音对和弦、节奏相关度按力度加权。"""
import numpy as np
import pytest

from guitarize.arrange import power as P
from guitarize.arrange.patterns import POWER_PATTERNS, accent_vector
from guitarize.config import Thresholds, load_profile, power_patterns
from guitarize.render import alphatab
from guitarize.verify import metrics
from guitarize.verify.model import load_score
from guitarize.verify.playability import check_playability

MID = load_profile("intermediate")
BEG = load_profile("beginner")


def _bar(i, chords, section=None):
    """chords：[(标签, 起始拍(1 起), 拍数)]。"""
    return {"bar": i + 1, "section": section, "t_start": i * 1.44, "t_end": (i + 1) * 1.44,
            "chords": [{"beat": b, "dur_beats": d, "label": lb, "confidence": "high"} for lb, b, d in chords],
            "melody": []}


def _sheet(bars, feel="shuffle", sections=None):
    sections = sections or {}
    return {"key": "B", "mode": "major", "tempo_bpm": 167.0, "feel": feel, "beats_per_bar": 4,
            "bars": [_bar(i, c, sections.get(i)) for i, c in enumerate(bars)]}


def _build(text, tmp_path, name="p"):
    src = tmp_path / f"{name}.atex"
    src.write_text(text, encoding="utf-8")
    return load_score(alphatab.build(src, tmp_path / "b", name)["json"])


def _frets(name, profile=MID):
    return [v.frets for v in P.power_shapes(name, profile)]


# ---------- 和弦 → 强力和弦 ----------

def test_power_name_is_the_chord_root():
    assert P.power_name("B") == "B5" and P.power_name("G#m") == "G#5" and P.power_name("C#m7") == "C#5"
    assert P.power_name("F#7") == "F#5" and P.power_name("E5") == "E5" and P.power_name("Bb") == "A#5"
    # 斜杠和弦按和弦根音，不按低音：D#/G → D#5，B/D# → B5（不是 D#5）
    assert P.power_name("D#/G") == "D#5" and P.power_name("B/D#") == "B5"
    assert P.power_name("N") is None and P.power_name("N.C.") is None


def test_power_shapes_root_on_six_or_five_with_fifth_and_octave():
    #           1 弦 → 6 弦
    assert _frets("B5") == [(None, None, 4, 4, 2, None), (None, None, None, 9, 9, 7)]
    assert _frets("E5") == [(None, None, None, 2, 2, 0), (None, None, 9, 9, 7, None)]   # 开放 E5 在前
    assert _frets("A5") == [(None, None, 2, 2, 0, None), (None, None, None, 7, 7, 5)]
    assert _frets("D#5") == [(None, None, 8, 8, 6, None)]          # 6 弦 11 品超出 0–9 品
    assert _frets("G#5") == [(None, None, None, 6, 6, 4)]          # 5 弦 11 品超出 0–9 品
    assert all(v.name == "C#5" for v in P.power_shapes("C#5", MID))


def test_power_shapes_follow_the_level():
    # 初级档最高 7 品：6 弦 7 品的 B5（五音、八度在 9 品）弹不了，只剩 5 弦 2 品
    assert _frets("B5", BEG) == [(None, None, 4, 4, 2, None)]
    with pytest.raises(ValueError, match="D#5"):
        P.power_shapes("D#5", BEG)                                 # 5 弦 6 品的五音在 8 品，初级档弹不了


def test_shapes_minimise_hand_movement():
    # 只有 B5 / E5：都放在低把位（5 弦 2 品的 B5 + 开放 E5），不换把
    chords = [["B5"] * 4, ["E5"] * 4] * 4
    shapes = P.choose_shapes(chords, {n: P.power_shapes(n, MID) for n in ("B5", "E5")})
    assert shapes["B5"].frets == (None, None, 4, 4, 2, None) and shapes["E5"].frets == (None, None, None, 2, 2, 0)
    # D#5 只能在 5 弦 6 品：反复 D#5 ↔ E5 时 E5 用 5 弦 7 品（移 1 品），不用开放 E5（移 4 品）
    chords = [["D#5"] * 4, ["E5"] * 4] * 4
    shapes = P.choose_shapes(chords, {n: P.power_shapes(n, MID) for n in ("D#5", "E5")})
    assert shapes["E5"].frets == (None, None, 9, 9, 7, None)


def test_shapes_for_the_nanmonee_chorus():
    # 只看副歌 16 小节：全放在 4–9 品一带（B5 6 弦 7 品、E5 5 弦 7 品……）共移 20 品，比低把位那一套（开放 E5）的 28 品少
    labels = ["B", "E", "D#/G", "G#m", "E", "B/D#", "C#", "D#", "B", "E", "D#/G", "G#m", "E", "B/D#", "C#m7", "F#"]
    chords = [[P.power_name(lb)] * 4 for lb in labels]
    cands = {n: P.power_shapes(n, MID) for n in dict.fromkeys(c for bar in chords for c in bar)}
    shapes = P.choose_shapes(chords, cands)
    assert {n: v.frets for n, v in shapes.items()} == {
        "B5": (None, None, None, 9, 9, 7), "E5": (None, None, 9, 9, 7, None), "D#5": (None, None, 8, 8, 6, None),
        "G#5": (None, None, None, 6, 6, 4), "C#5": (None, None, None, 11, 11, 9), "F#5": (None, None, 11, 11, 9, None)}
    assert P.movement(chords, shapes) == 20 and P.movement(chords, {n: c[0] for n, c in cands.items()}) == 28


# ---------- 节奏型 → 记号 ----------

B5 = {"B5": P.power_shapes("B5", MID)[0], "A5": P.power_shapes("A5", MID)[0]}


def test_verse_tokens_palm_mute_accent_brush_and_dynamics():
    tokens, last = P.bar_tokens(["B5"] * 4, POWER_PATTERNS["verse"], B5, None, "shuffle")
    assert tokens == ['(4.3{pm} 4.4{pm} 2.5{pm}).4 {ch "B5" bd 60 dy f}',     # 1：闷音
                      "(4.3{ac} 4.4{ac} 2.5{ac}).4 {bd 60 dy f}",              # 2：空弦重音
                      "(4.3{pm} 4.4{pm} 2.5{pm}).8 {bd 60 dy f}",              # 3：闷音
                      "(4.3 4.4 2.5).8 {bu 60 dy mp}",                         # 3 的反拍：短促的空弦上扫
                      "(4.3 4.4 2.5).4 {bd 60 dy f}"]                          # 4：空弦
    assert last == "B5"


def test_verse_end_chorus_and_hit_tokens():
    tokens, _ = P.bar_tokens(["B5"] * 4, POWER_PATTERNS["verse_end"], B5, "B5", "shuffle")
    assert tokens == ['(4.3 4.4 2.5).4 {ch "B5" bd 60 dy f}', "(4.3{ac} 4.4{ac} 2.5{ac}).8 {bd 60 dy f}",
                      "(4.3 4.4 2.5).8 {bu 60 dy mp}", "(-.3 -.4 -.5).2"]
    tokens, _ = P.bar_tokens(["B5"] * 4, POWER_PATTERNS["chorus"], B5, "B5", "shuffle")
    assert tokens == ['(4.3{ac} 4.4{ac} 2.5{ac}).4 {ch "B5" bd 60 dy f}', "(4.3 4.4 2.5).4 {bd 60 dy mp}",
                      "(-.3 -.4 -.5).8", "(4.3 4.4 2.5).8 {bd 60 dy mp}", "(-.3 -.4 -.5).4"]
    tokens, _ = P.bar_tokens(["B5"] * 4, POWER_PATTERNS["hit"], B5, "B5", "shuffle")
    assert tokens == ['(4.3{ac} 4.4{ac} 2.5{ac}).1 {ch "B5" bd 60 dy f}']


def test_chord_change_and_rest_in_a_bar():
    tokens, last = P.bar_tokens(["B5", "B5", "A5", "A5"], POWER_PATTERNS["verse"], B5, "B5", "shuffle")
    assert tokens[2] == '(2.3{pm} 2.4{pm} 0.5{pm}).8 {ch "A5" bd 60 dy f}' and last == "A5"
    tokens, last = P.bar_tokens(["N"] * 4, POWER_PATTERNS["verse"], B5, "B5", "shuffle")
    assert tokens == ["r.1"] and last is None


# ---------- 选节奏型（只看吉他分轨） ----------

def _onsets(names, feel="shuffle"):
    return np.array([accent_vector(POWER_PATTERNS[n], feel) if n else np.r_[0, 1, np.zeros(10)] for n in names])


def test_patterns_follow_the_guitar_onsets_per_section():
    # 4 小节副歌 | Bメロ：3 小节 verse + 5 小节 verse_end（段落内切一刀）| 无和弦 | 2 小节没有起伏的音墙 → 一击
    rows = ["chorus"] * 4 + ["verse"] * 3 + ["verse_end"] * 5 + ["hit", None, None]
    chords = [[("B", 1, 4)]] * 12 + [[("N", 1, 4)]] + [[("E", 1, 4)]] * 2
    ls = _sheet(chords, sections={0: "サビ", 4: "Bメロ", 12: "間奏"})
    got = P.choose_patterns(_onsets(rows), ls)
    assert got == ["chorus"] * 4 + ["verse"] * 3 + ["verse_end"] * 5 + [None, "hit", "hit"]
    # 没有吉他分轨：整首默认型
    assert P.choose_patterns(None, ls) == [P.POWER_DEFAULT] * 12 + [None, P.POWER_DEFAULT, P.POWER_DEFAULT]


def test_no_split_when_both_halves_pick_the_same_pattern():
    ls = _sheet([[("B", 1, 4)]] * 8, sections={0: "サビ"})
    assert P.choose_patterns(_onsets(["chorus"] * 8), ls) == ["chorus"] * 8


def test_pattern_overrides_run_to_the_end_of_the_phrase():
    ls = _sheet([[("B", 1, 4)]] * 6 + [[("N", 1, 4)]] + [[("B", 1, 4)]] * 2, sections={0: "A", 3: "B"})
    got = P.choose_patterns(None, ls, {2: "hit", 4: "verse_end"})
    assert got == ["verse", "hit", "hit", "verse_end", "verse_end", "verse_end", None, "verse", "verse"]
    with pytest.raises(ValueError, match="bar 7"):
        P.choose_patterns(None, ls, {7: "hit"})                    # 无和弦小节
    with pytest.raises(ValueError, match="bar 10"):
        P.choose_patterns(None, ls, {10: "hit"})


def test_power_patterns_override_is_validated():
    assert power_patterns({}) == {}
    assert power_patterns({"power_patterns": {"33": "verse_end", 22: "verse"}}) == {33: "verse_end", 22: "verse"}
    with pytest.raises(ValueError, match="overrides.power_patterns"):
        power_patterns({"power_patterns": {33: "pop"}})
    with pytest.raises(ValueError, match="overrides.power_patterns"):
        power_patterns({"power_patterns": {"x": "verse"}})


def test_palm_mute_and_accent_are_note_effects(tmp_path):
    # alphaTab 1.8.4：pm / ac 只能写成音符效果（'2.5{pm}'）；写成拍效果 '{pm}' 报 Unrecognized property
    from guitarize.render.alphatab import AlphaTexError

    head = '\\title "T"\n\\track "Guitar"\n\\staff {tabs}\n\\tuning (E4 B3 G3 D3 A2 E2)\n\\ts 4 4\n'
    for fx in ("pm", "ac"):
        with pytest.raises(AlphaTexError, match=f"Unrecognized property '{fx}'"):
            _build(head + f"(2.5 4.4).4 {{{fx}}} r.4 r.2 |\n", tmp_path, fx)
    score = _build(head + "(2.5{pm} 4.4{pm}).4 {bd 60} (2.5{ac} 4.4{ac}).4 {bu 60} r.2 |\n", tmp_path, "ok")
    assert [a.palm_mute for a in score.attacks()] == [True, False]


# ---------- 整谱：alphaTab 构建、可弹性、校验 ----------

PROGRESSION = [[("B", 1, 4)], [("E", 1, 2), ("D#/G", 3, 2)], [("G#m", 1, 4)], [("B/D#", 1, 2), ("C#m7", 3, 2)],
               [("F#7", 1, 4)], [("A", 1, 2), ("N", 3, 2)], [("D#", 1, 4)], [("B", 1, 4)]]


def _power(tmp_path, patterns=None, profile=MID):
    ls = _sheet(PROGRESSION, sections={0: "サビ", 4: "Aメロ"})
    patterns = patterns or ["chorus", "chorus", "chorus", "chorus", "verse", "verse", "verse_end", "hit"]
    text = P.generate_power(ls, profile, patterns, "Test")
    return ls, text, _build(text, tmp_path)


def test_generate_power_builds_without_capo_and_is_playable(tmp_path):
    ls, text, score = _power(tmp_path)
    assert "\\capo" not in text and score.capo == 0 and "\\chordDiagramsInScore" in text
    for name in ("B5", "E5", "D#5", "G#5", "C#5", "F#5", "A5"):
        assert f'\\chord "{name}" ' in text and f'ch "{name}"' in text
    assert "// bar 1 [サビ] · 节奏型 chorus" in text and "// bar 5 [Aメロ] · 节奏型 verse" in text
    assert "// bar 7 · 节奏型 verse_end" in text and "// bar 2 ·" not in text
    assert check_playability(score, MID, "power") == []
    assert all(e.brush in ("down", "up") for e in score.events if not e.rest and any(not n.tie for n in e.notes))
    muted = [a for a in score.attacks() if a.palm_mute]
    assert muted and all(a.bar in (4, 5) for a in muted)        # 闷音只在 verse 小节
    assert max(a.velocity for a in score.attacks()) == 111       # f + 重音
    assert min(a.velocity for a in score.attacks()) == 63        # mp：上扫 / 轻扫
    assert not [a for a in score.attacks() if a.bar == 5 and a.start >= 5 * 4 + 2]   # 第 6 小节后半无和弦


def test_beginner_power_chords_are_playable(tmp_path):
    ls = _sheet([[("B", 1, 4)], [("E", 1, 4)], [("G#m", 1, 4)], [("F#", 1, 4)], [("A", 1, 4)], [("C#m", 1, 4)]])
    text = P.generate_power(ls, BEG, ["verse"] * 6, "T")
    assert check_playability(_build(text, tmp_path), BEG, "power") == []


def test_verify_power_matches_chords_by_root(tmp_path):
    ls, _, score = _power(tmp_path)
    bars = metrics.score_bars(score, ls, "power")
    s = metrics.summarize(bars, [], Thresholds(), "power")
    assert s["chord_agreement"] == 1.0 and s["chord_mark_issues"] == 0 and s["chord_target"] == 0.9
    assert set(s["checks"]) == {"playability", "chord", "reference", "chord_marks"} and s["pass"] is True
    wrong = _sheet([[("E", 1, 4)]] + PROGRESSION[1:])               # 第 1 小节参考 E，谱上是 B5
    bars = metrics.score_bars(score, wrong, "power")
    assert bars[0].chord_agree == 0 and "参考 E" in bars[0].chord_issues[0]
    assert metrics.problem_bars(bars, [], Thresholds(), "power") == [1]


def test_power_chord_marks_match_by_root(tmp_path):
    # riff 小节：G#5 的 hit 后 2/3 拍是 B5，这一拍听起来是 G#m——强力和弦名没有三音，G#5 标记仍然相符（弹唱谱照旧报）
    head = P.generate_power(_sheet([[("B", 1, 4)]]), MID, ["hit"], "T").split("// bar 1")[0]
    body = ('// bar 1\nr.2 (6.4 6.5 4.6).4 {ch "G#5" tu 3} (4.3 4.4 2.5).4 {tu 3} (6.4 6.5 4.6).4 {tu 3} |\n')
    score = _build(head + body, tmp_path)
    ls = _sheet([[("G#m", 1, 4)]])
    assert [b.mark_issues for b in metrics.score_bars(score, ls, "power")] == [[]]
    assert metrics.score_bars(score, ls, "strum")[0].mark_issues == ["拍 3 标记 G#5，实际音听起来是 G#:min"]


def test_power_rhythm_vector_weighs_palm_mutes_and_dynamics(tmp_path):
    _, _, score = _power(tmp_path, ["verse"] * 8)
    v = metrics.strum_vector(score.attacks(), score.bars[0], dynamic=True)
    assert list(np.flatnonzero(v)) == [0, 3, 6, 8, 9]
    assert v[0] == pytest.approx(0.3) and v[6] == pytest.approx(0.3)         # 闷音
    assert v[3] == 1.0 and v[9] == 1.0 and v[8] == pytest.approx(63 / 95)    # 重音 / f / 上扫 mp
    plain = metrics.strum_vector(score.attacks(), score.bars[0])              # 弹唱谱的算法不变：↓ 1、↑ 0.6
    assert plain[0] == 1.0 and plain[8] == pytest.approx(0.6)


def test_arrange_verify_render_cli(demo_song, capsys):
    import json
    import shutil

    from guitarize.cli import main
    from guitarize.render.synth import midi_to_wav

    songs = str(demo_song.root.parent)
    assert main(["--songs-dir", songs, "arrange", "demo", "--arr", "power"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].endswith("arrangements/power.atex") and "capo" not in out and "自检：0 处违规" in out
    text = demo_song.atex("power").read_text(encoding="utf-8")
    assert "\\capo" not in text and '\\chord "C5" ' in text and "· 节奏型 verse" in text   # 没有吉他分轨：默认型

    # 拿编配自己的合成当"吉他分轨"：节奏相关度应很高（只看吉他分轨，不看鼓）
    built = alphatab.build(demo_song.atex("power"), demo_song.build_dir, "power")
    demo_song.stem("guitar").parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(midi_to_wav(built["mid"], demo_song.build("power", "wav")), demo_song.stem("guitar"))
    assert main(["--songs-dir", songs, "verify", "demo", "--arr", "power"]) == 0
    it = demo_song.report_root("power") / "iter-01"
    s = json.loads((it / "scores.json").read_text(encoding="utf-8"))["summary"]
    assert s["pass"] is True and s["chord_agreement"] == 1.0 and s["violations"] == 0
    assert s["rhythm_correlation"] > 0.8 and s["rhythm_by_section"][0]["bars"] == [1, 8]
    report = (it / "report.md").read_text(encoding="utf-8")
    assert "# 强力和弦校验报告 · iter-01" in report and "✅ 达标" in report and "| 和弦一致率（按根音） | 100.0% |" in report
    assert "## 节奏相关度（按段落，对照吉他分轨）" in report

    assert main(["--songs-dir", songs, "render", "demo", "--arr", "power"]) == 0
    txt = demo_song.out("power", "txt").read_text(encoding="utf-8")
    assert "Capo" not in txt and "C5" in txt and "PM  >   PM" in txt and "↑" in txt
    page = demo_song.out("power", "html").read_text(encoding="utf-8")
    assert "<title>Demo Twinkle · 强力和弦</title>" in page
    assert demo_song.out("power", "pdf").read_bytes()[:4] == b"%PDF"
