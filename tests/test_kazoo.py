"""卡祖笛独奏谱（arr = kazoo）：确认过的旋律按实际音高、与旋律谱同一个节奏；连奏写连线；五线谱 + 简谱；和弦名（实际音高）。"""
import re

import pytest

from guitarize.arrange import atex
from guitarize.arrange import kazoo as K
from guitarize.arrange.kazoo import generate_kazoo
from guitarize.arrange.melody import generate_melody
from guitarize.config import Thresholds, load_profile
from guitarize.render import alphatab
from guitarize.verify import metrics
from guitarize.verify.model import load_score

MID = load_profile("intermediate")
T3 = 1 / 3
L = {"legato": True}


def _bar(i, mel, chord="N", section=None):
    """mel：[(拍内偏移, 时值, 音高)] 或 [(拍内偏移, 时值, 音高, 标记)]，标记如 {"legato": True}。"""
    chords = chord if isinstance(chord, list) else [{"beat": 1, "dur_beats": 4, "label": chord, "confidence": "high"}]
    notes = []
    for o, d, m, *extra in mel:
        notes.append({"start_beats": round(i * 4 + o, 4), "dur_beats": round(d, 4), "midi": m, "conf": 0.9,
                      "out_of_key": False, **(extra[0] if extra else {})})
    return {"bar": i + 1, "section": section, "t_start": i * 1.44, "t_end": (i + 1) * 1.44, "chords": chords,
            "melody": notes}


def _sheet(bars, feel="shuffle", tempo=167.0, key="B"):
    return {"key": key, "mode": "major", "tempo_bpm": tempo, "feel": feel, "beats_per_bar": 4, "bars": bars}


def _build(text, tmp_path, name="k"):
    src = tmp_path / f"{name}.atex"
    src.write_text(text, encoding="utf-8")
    return load_score(alphatab.build(src, tmp_path / "b", name)["json"])


def _voice_bar(text, n):
    return text.split(f"// bar {n}")[1].split("\n", 1)[1].splitlines()[0]


def _onsets(score):
    return [(round(s.start, 4), s.midi) for s in score.melody(voice=0)]


# なんもねえ 第 1–2 小节（进副歌的拾音 + 副歌第一句），Shuffle 167；第 3 小节一拍三个三连音
BAR1 = [(1 + 2 * T3, T3, 66), (2, 1, 66), (3, T3, 59), (3 + T3, 4 * T3, 66)]
BAR2 = [(2, T3, 64), (2 + T3, 2 * T3, 63), (3, 2 * T3, 66), (3 + 2 * T3, T3, 66)]
BAR3 = [(0, T3, 61), (T3, T3, 63), (2 * T3, T3, 64), (1, 1, 66), (2, 2, 68)]


def test_every_melody_note_at_sounding_pitch(tmp_path):
    # 每一个确认过的旋律音、原唱的八度（不加变调夹、不移八度）：低到 D#3 也照写
    low = [(0, 1, 51), (1, 1, 54), (2, 2, 56)]
    ls = _sheet([_bar(0, BAR1), _bar(1, BAR2), _bar(2, BAR3), _bar(3, low)])
    score = _build(generate_kazoo(ls, "T"), tmp_path)
    want = [(round(m["start_beats"], 4), m["midi"]) for b in ls["bars"] for m in b["melody"]]
    assert _onsets(score) == want
    bars = metrics.score_bars(score, ls, "kazoo")
    s = metrics.summarize(bars, [], Thresholds(), "kazoo", arrangement_bars=4, reference_bars=4)
    assert s["melody_f1"] == pytest.approx(1.0) and s["melody_octave_exact"] == pytest.approx(1.0)
    assert s["pass"] is True


def test_same_rhythm_as_the_melody_arrangement(tmp_path):
    # 起音、时值、三连音、休止、跨小节线的连音线都和旋律谱一样（旋律谱不加变调夹、不移八度时连音高也一样）
    ls = _sheet([_bar(0, BAR1, "B", "サビ"), _bar(1, BAR2, "E"), _bar(2, BAR3, "B"),
                 _bar(3, [(0, 1, 64), (2, 0.25, 66), (2.5, 0.75, 67), (3.25, 0.25, 69)])])
    kazoo = _build(generate_kazoo(ls, "T"), tmp_path, "k")
    melody = _build(generate_melody(ls, 0, MID, "T", octave=0), tmp_path, "m")

    def sounding(score):
        return [(round(s.start, 4), round(s.end, 4), s.midi) for s in score.sounding()]

    assert sounding(kazoo) == sounding(melody)
    assert [(round(e.start, 4), e.tuplet) for e in kazoo.events if not e.rest] == \
           [(round(e.start, 4), e.tuplet) for e in melody.events if not e.rest]


def test_pitched_tokens_triplets_and_ties_across_the_bar_line():
    text = generate_kazoo(_sheet([_bar(0, BAR1, "B"), _bar(1, BAR2, "E"), _bar(2, BAR3, "B")]), "T")
    # 第 1 小节：2+⅔ 的 F#4 是摇摆八分对的后一个（记在 2.5），第 4 拍有 +⅓ 起音 → 三连音；F#4 连到下一小节
    assert _voice_bar(text, 1) == 'r.4 {ch "B" dy f} r.8 F#4.8 F#4.4 B3.8 {tu 3} F#4.4 {tu 3} |'
    assert _voice_bar(text, 2).startswith('F#4{-}.8 {ch "E"} r.8 r.4 E4.8 {tu 3} D#4.4 {tu 3} ')
    assert _voice_bar(text, 3).startswith('C#4.8 {ch "B" tu 3} D#4.8 {tu 3} E4.8 {tu 3} F#4.4 G#4.2')
    assert not re.search(r"(^|\s)-\.", text)                         # 没有吉他谱的 '-.弦' 连音线


def test_flat_keys_are_spelled_with_flats():
    ls = _sheet([_bar(0, [(0, 1, 63), (1, 1, 70), (2, 2, 68)], "Eb")], feel="straight", key="Eb")
    text = generate_kazoo(ls, "T")
    assert "\\ks Eb" in text and _voice_bar(text, 1) == 'Eb4.4 {ch "Eb" dy f} Bb4.4 Ab4.2 |'


def test_header_staff_and_labels(tmp_path):
    mel = [(0, 1, 64), (1, 1, 66), (2, 2, 68)]
    split = [{"beat": 1, "dur_beats": 2, "label": "E", "confidence": "high"},
             {"beat": 3, "dur_beats": 2, "label": "F#", "confidence": "high"}]
    ls = _sheet([_bar(0, mel, "N", "イントロ"), _bar(1, mel, "B", "サビ"), _bar(2, mel, "B"),
                 _bar(3, [(0, 4, 68)], split), _bar(4, mel, "N")])
    text = generate_kazoo(ls, "なんもねえ")
    head = text.split("// bar 1")[0]
    for line in ('\\track "Kazoo"', f"\\instrument {K.KAZOO_PROGRAM}", "\\staff {score numbered}",
                 "\\tuning (piano)", "\\ts 4 4", "\\ks B", "\\tf triplet-8th"):
        assert line in head.splitlines()
    assert K.KAZOO_PROGRAM == 59
    assert "\\capo" not in text and "\\chord" not in text and "\\section" not in text and "\\voice" not in text
    # 和弦名（实际音高，不画和弦图）：每小节第 1 拍都标，小节中间换和弦再标，转入无和弦标 N.C.；段落名写成文字
    assert _voice_bar(text, 1).startswith('E4.4 {ch "N.C." txt "イントロ" dy f}')
    assert _voice_bar(text, 2).startswith('E4.4 {ch "B" txt "サビ"}')
    assert _voice_bar(text, 3).startswith('E4.4 {ch "B"}')
    assert _voice_bar(text, 4) == 'G#4.2 {ch "E"} G#4{-}.2 {ch "F#"} |'      # 长音在换和弦处切开、连音线接上
    assert _voice_bar(text, 5).startswith('E4.4 {ch "N.C."}')
    assert "// bar 1 [イントロ]" in text and "// bar 2 [サビ]" in text
    score = _build(text, tmp_path)
    assert score.tuning == [] and score.capo == 0 and score.bars[0].key == 5
    assert [e.chord for e in score.events if e.chord] == ["N.C.", "B", "B", "E", "F#", "N.C."]
    assert [e.text for e in score.events if e.text] == ["イントロ", "サビ"]


def test_legato_notes_are_slurred_from_the_previous_note(tmp_path):
    # 连奏（同一音节里滑过去）：不重新吐音，从前一个音起画连线；连成一串的几个音一条连线罩住整串。
    # 前一个音跨小节线时连线从它的起音开始（音头），连音线延续上不挂
    mel1 = [(0, 1, 63), (1, 1, 66, L), (2, 1, 68), (3, 2, 66)]
    mel2 = [(1, 1, 64, L), (2, 1, 63, L), (3, 1, 61)]
    text = generate_kazoo(_sheet([_bar(0, mel1), _bar(1, mel2)], feel="straight"), "T")
    assert _voice_bar(text, 1) == 'D#4{slur s1}.4 {ch "N.C." dy f} F#4{slur s1}.4 G#4.4 F#4{slur s2}.4 |'
    assert _voice_bar(text, 2) == "F#4{-}.4 E4.4 D#4{slur s2}.4 C#4.4 |"
    score = _build(text, tmp_path)
    notes = [(n.midi, n.tie, n.slur_start, n.slur_end) for e in score.events for n in e.notes]
    assert notes == [(63, False, True, False), (66, False, False, True), (68, False, False, False),
                     (66, False, True, False), (66, True, False, False), (64, False, False, False),
                     (63, False, False, True), (61, False, False, False)]
    assert K.legato_slurs([]) == {}


def test_display_octave_puts_most_notes_in_the_middle_octave():
    # 简谱不加点的八度：alphaTab 默认从第 4 组的主音起（B 大调 B4），男声旋律大多在它下面 → 8vb 谱号
    # （记谱高八度，简谱 B3–A#4 不加点）；本来就在中音区的旋律不加记号
    fifths = atex.key_fifths("B", "major")
    assert atex.numbered_one(fifths, False) == 71 and atex.numbered_one(0, False) == 60
    assert atex.numbered_one(atex.key_fifths("A", "minor"), True) == 69
    male = [51, 54, 56, 59, 61, 63, 64, 66, 66, 68, 70, 71]
    assert K.display_ottava(male, fifths, False) == "8vb"
    assert K.display_ottava([71, 73, 75, 76, 78], fifths, False) is None
    assert K.display_ottava([], fifths, False) is None
    assert K.display_ottava([36, 38, 40, 43], 0, False) == "15mb"
    ls = _sheet([_bar(0, [(0, 1, 59), (1, 1, 63), (2, 2, 66)])])
    assert "\\ottava 8vb" in generate_kazoo(ls, "T").split("// bar 1")[0]


# ---------- 文本简谱（out/kazoo.txt） ----------

def test_text_jianpu_for_a_small_known_melody(tmp_path):
    from guitarize.render.jianpu import render_jianpu
    ls = _sheet([_bar(0, BAR1, "B", "サビ"), _bar(1, BAR2, "E"), _bar(2, BAR3, "B")])
    text = render_jianpu(_build(generate_kazoo(ls, "なんもねえ"), tmp_path))
    lines = text.splitlines()
    assert lines[0] == "1=B 4/4"
    assert lines[1].startswith("なんもねえ · ♩ = 167 · Shuffle")
    assert "不加记号的中音区是 B3–A#4" in text and "³5-6" in text            # 记法写在文件开头
    # 8vb 谱号：B3 = 1；摇摆八分对写成两个半拍；+⅓ 的拍写成三连音（³）；- 是接着响，0 是休止
    assert lines[-3:] == ["  1 | 0    05   5    ³15- |  [サビ] B",
                          "  2 | -0   0    ³43- 55   |  E",
                          "  3 | ³234 5    6    -    |  B"]


def test_text_jianpu_accidentals_octaves_and_sixteenths(tmp_path):
    # F 大调（降号调）：Bb4 是 4、B4 是 #4；E4 在中音区（F4–E5）下面是 7,；F5 是 1'；¾ 拍 + ¼ 拍写成 #4--5
    from guitarize.render.jianpu import render_jianpu
    ls = _sheet([_bar(0, [(0, 1, 64), (1, 0.5, 65), (1.5, 0.5, 70), (2, 0.75, 71), (2.75, 0.25, 72), (3, 1, 77)],
                      "F"), _bar(1, [], "N")], feel="straight", tempo=100, key="F")
    text = render_jianpu(_build(generate_kazoo(ls, "T"), tmp_path))
    assert text.splitlines()[0] == "1=F 4/4" and "Shuffle" not in text and "中音区是 F4–E5" in text
    assert text.splitlines()[-2:] == ["  1 | 7,    14    #4--5 1'    |  F", "  2 | 0     0     0     0     |  N.C."]


def test_a_silent_bar_is_two_half_rests(tmp_path):
    # alphaTab 1.8.4 的简谱谱表画不了全休止 r.1（渲染时报错，网页和 PDF 都出不来）：休止在第 3 拍切开
    ls = _sheet([_bar(0, [(0, 1, 64)], "B"), _bar(1, [], "B"), _bar(2, [(1, 1, 66)], "B")], feel="straight")
    text = generate_kazoo(ls, "T")
    assert _voice_bar(text, 1) == 'E4.4 {ch "B" dy f} r.4 r.2 |'
    assert _voice_bar(text, 2) == 'r.2 {ch "B"} r.2 |' and _voice_bar(text, 3) == 'r.4 {ch "B"} F#4.4 r.2 |'
    assert "r.1" not in text
    src = tmp_path / "k.atex"
    src.write_text(text, encoding="utf-8")
    assert alphatab.build(src, tmp_path / "b", "k")["svg"].stat().st_size > 1000   # 渲染得出来


# ---------- 校验：旋律 F1、音域、换气 ----------

def test_range_and_breath_warnings(tmp_path):
    # ♩ = 60：一拍一秒。第 1–3 小节连着唱 11 秒没有休止 → 换气提醒；第 4 小节 3 秒、前面有休止 → 不提醒
    ls = _sheet([_bar(0, [(0, 4, 60)]), _bar(1, [(0, 4, 62)]), _bar(2, [(0, 2, 64), (2, 1, 65)]),
                 _bar(3, [(0, 2, 67), (2, 1, 69)])], feel="straight", tempo=60, key="C")
    score = _build(generate_kazoo(ls, "T"), tmp_path)
    stats = metrics.kazoo_stats(score, Thresholds())
    assert (stats["range_low"], stats["range_high"], stats["range_span"]) == ("C4", "A4", 9)
    assert stats["range"] == "C4–A4（9 个半音）"
    assert stats["breath_warnings"] == 1 and stats["breath_seconds"] == 8.0
    assert stats["breaths"] == [{"bars": [1, 3], "seconds": 11.0, "start": "1"}]
    assert metrics.kazoo_stats(score, Thresholds(breath_seconds=12.0))["breaths"] == []
    # 降号调的音名写降号
    flat = _build(generate_kazoo(_sheet([_bar(0, [(0, 1, 63), (2, 1, 70)])], key="Eb", feel="straight"), "T"),
                  tmp_path, "f")
    assert metrics.kazoo_stats(flat, Thresholds())["range"] == "Eb4–Bb4（7 个半音）"


def test_kazoo_summary_counts_every_note_and_no_chords(tmp_path):
    # 漏一个音：F1 0.8 < 0.9 → 未达标、标出问题小节；不评和弦、不查可弹性（记谱问题另算）
    ls = _sheet([_bar(0, [(0, 1, 60), (1.5, 0.5, 62), (2, 2, 64)], "C")], feel="straight", tempo=100, key="C")
    text = generate_kazoo(ls, "T").replace("D4.8 ", "r.8 ")
    score = _build(text, tmp_path)
    bars = metrics.score_bars(score, ls, "kazoo")
    assert bars[0].mel_issues == ["漏掉旋律音：拍 2& D4"]                   # 卡祖笛没有指法调
    s = metrics.summarize(bars, [], Thresholds(), "kazoo", arrangement_bars=1, reference_bars=1)
    assert s["melody_f1"] == pytest.approx(0.8) and s["melody_target"] == 0.9
    assert set(s["checks"]) == {"notation", "reference", "bar_count", "melody"} and s["pass"] is False
    assert s["chord_agreement"] is None and s["chord_target"] is None
    assert metrics.problem_bars(bars, [], Thresholds(), "kazoo") == [1]


def test_arrange_verify_render_cli(demo_song, capsys):
    import csv
    import json

    from guitarize.cli import main

    songs = str(demo_song.root.parent)
    assert main(["--songs-dir", songs, "arrange", "demo", "--arr", "kazoo"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].endswith("arrangements/kazoo.atex") and "capo" not in out and "自检：0 处违规" in out
    text = demo_song.atex("kazoo").read_text(encoding="utf-8")
    assert "\\staff {score numbered}" in text and "\\tuning (piano)" in text and "\\capo" not in text

    assert main(["--songs-dir", songs, "verify", "demo", "--arr", "kazoo"]) == 0
    it = demo_song.report_root("kazoo") / "iter-01"
    s = json.loads((it / "scores.json").read_text(encoding="utf-8"))["summary"]
    assert s["pass"] is True and s["melody_f1"] == 1.0 and s["violations"] == 0
    assert s["range"] == "C4–A4（9 个半音）"
    # 小星星 8 小节一口气唱完（♩ = 100，32 拍 19.2 秒，中间没有休止）
    assert s["breaths"] == [{"bars": [1, 8], "seconds": 19.2, "start": "1"}]
    report = (it / "report.md").read_text(encoding="utf-8")
    assert "# 卡祖笛校验报告 · iter-01" in report and "✅ 达标" in report
    assert "| 旋律 F1 | 1.000 | — | ≥ 0.90 |" in report and "| 小节数（编配 / 参考） | 8 / 8 | — | 相等 |" in report
    assert "| 音域 | C4–A4（9 个半音） | — | 仅报告 |" in report
    assert "## 换气提醒（连续超过 8 秒没有休止，1 处）\n- 第 1–8 小节：19.2 秒没有休止（第 1 小节拍 1 起）" in report
    assert "可弹性" not in report and "和弦" not in report and not (it / "snippets").exists()
    rows = list(csv.DictReader((demo_song.report_root("kazoo") / "history.csv").open(encoding="utf-8")))
    assert rows[-1]["range_span"] == "9" and rows[-1]["breath_warnings"] == "1" and rows[-1]["pass"] == "True"

    assert main(["--songs-dir", songs, "render", "demo", "--arr", "kazoo"]) == 0
    txt = demo_song.out("kazoo", "txt").read_text(encoding="utf-8")
    assert txt.splitlines()[0] == "1=C 4/4" and "  1 | 1 1 5 5 |" in txt and "e|" not in txt
    page = demo_song.out("kazoo", "html").read_text(encoding="utf-8")
    assert "<title>Demo Twinkle · 卡祖笛</title>" in page and "window.REF_AUDIO = " in page and "\\staff {score numbered}" in page
    assert demo_song.out("kazoo", "pdf").read_bytes()[:4] == b"%PDF"
