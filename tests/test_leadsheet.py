import copy
import json

import numpy as np
import pytest
import soundfile as sf

from guitarize.analyze.leadsheet import build_leadsheet, summary_table
from guitarize.analyze.stems import mix_harmonic

BEATS = {"tempo_bpm": 120.0, "feel": "straight", "time_signature": [4, 4], "pickup_beats": 0, "pad_beats": 0,
         "short_bars": [], "beats": [], "bars": [{"bar": 1, "t_start": 0.0, "t_end": 2.0, "beats": 4},
                                                  {"bar": 2, "t_start": 2.0, "t_end": 4.0, "beats": 4}]}
CHORDS = {"segments": [
    {"bar": 1, "beat": 1, "dur_beats": 4, "label": "C", "confidence": "high", "votes": {"template": "C", "web": "C"}},
    {"bar": 2, "beat": 1, "dur_beats": 2, "label": "G", "confidence": "low", "votes": {"template": "Em", "web": "G"}},
    {"bar": 2, "beat": 3, "dur_beats": 2, "label": "N", "confidence": "medium", "votes": {"template": "C", "web": "N"}},
]}
MELODY = {"grid": "sixteenth", "notes": [
    {"start_beats": 0.0, "dur_beats": 1.0, "midi": 60, "out_of_key": False, "conf": 0.9},
    {"start_beats": 4.5, "dur_beats": 0.5, "midi": 61, "out_of_key": True, "conf": 0.8},
]}


def test_build_leadsheet_matches_phase1_contract():
    ls = build_leadsheet(BEATS, CHORDS, MELODY, ("C", "major"), {2: "サビ"})
    assert ls["key"] == "C" and ls["mode"] == "major" and ls["beats_per_bar"] == 4
    b1, b2 = ls["bars"]
    assert (b1["bar"], b1["section"], b1["t_start"], b1["t_end"]) == (1, None, 0.0, 2.0)
    assert b2["section"] == "サビ"
    assert [c["label"] for c in b2["chords"]] == ["G", "N"]
    assert b1["melody"] == [{"start_beats": 0.0, "dur_beats": 1.0, "midi": 60, "conf": 0.9, "out_of_key": False}]
    assert b2["melody"][0]["midi"] == 61


def test_build_leadsheet_rejects_short_bars():
    import pytest

    with pytest.raises(ValueError, match="short bars"):
        build_leadsheet({**BEATS, "short_bars": [5]}, CHORDS, MELODY, ("C", "major"), {})


def test_summary_table_shows_confidence_and_shapes():
    ls = build_leadsheet(BEATS, CHORDS, MELODY, ("C", "major"), {2: "サビ"})
    text = summary_table(ls, capo=2)
    lines = text.splitlines()
    assert "capo 2" in lines[0]
    assert "1:C(A#)" in lines[1]
    assert "G(F)??" in lines[2] and "template=Em" in lines[2] and "[サビ]" in lines[2]
    assert "C#4*" in lines[2]


def test_mix_harmonic(tmp_path):
    for name, v in {"bass": 0.1, "guitar": 0.2, "piano": 0.0, "other": 0.05}.items():
        sf.write(tmp_path / f"{name}.wav", np.full((100, 2), v), 22050)
    out = mix_harmonic(tmp_path)
    y, sr = sf.read(out)
    assert sr == 22050 and y.shape == (100, 2) and np.allclose(y, 0.35, atol=1e-3)


def _paths_cfg(tmp_path):
    from guitarize.config import SongConfig
    from guitarize.paths import SongPaths

    return SongPaths(tmp_path / "song"), SongConfig(slug="song", title="t", artist="a")


def test_partial_steps_name_missing_prerequisite(tmp_path):
    import pytest

    from guitarize.analyze.pipeline import run_analysis

    paths, cfg = _paths_cfg(tmp_path)
    with pytest.raises(ValueError, match="--steps beats"):
        run_analysis(paths, cfg, ["chords"])
    with pytest.raises(ValueError, match="--steps beats"):
        run_analysis(paths, cfg, ["melody"])
    paths.beats_json.parent.mkdir(parents=True)
    paths.beats_json.write_text("{}")
    with pytest.raises(ValueError, match="--steps chords"):
        run_analysis(paths, cfg, ["leadsheet"])


def test_separate_surfaces_stderr(tmp_path, monkeypatch):
    import subprocess

    import pytest

    from guitarize.analyze import stems

    monkeypatch.setattr(stems, "_device", lambda: "cpu")
    monkeypatch.setattr(stems.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 1, b"", b"boom"))
    with pytest.raises(RuntimeError, match="boom"):
        stems.separate(tmp_path / "m.wav", tmp_path / "stems")


def test_separate_missing_stems_leaves_dir_untouched(tmp_path, monkeypatch):
    import subprocess

    import pytest

    from guitarize.analyze import stems

    monkeypatch.setattr(stems, "_device", lambda: "cpu")

    def fake_run(cmd, **k):
        out = tmp_path.__class__(cmd[cmd.index("-o") + 1]) / "htdemucs_6s" / "m"
        out.mkdir(parents=True)
        for n in ("vocals", "drums"):
            (out / f"{n}.wav").write_bytes(b"x")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(stems.subprocess, "run", fake_run)
    sd = tmp_path / "stems"
    with pytest.raises(RuntimeError, match="bass.*guitar"):
        stems.separate(tmp_path / "m.wav", sd)
    assert not list(sd.glob("*.wav"))


def test_mix_harmonic_is_float(tmp_path):
    for name in ("bass", "guitar", "piano", "other"):
        sf.write(tmp_path / f"{name}.wav", np.full((10, 2), 0.9), 22050)
    assert sf.info(mix_harmonic(tmp_path)).subtype == "FLOAT"


def _write_analysis(paths, chords=None, melody=None, beats=None):
    import json

    paths.analysis_dir.mkdir(parents=True, exist_ok=True)
    for path, data in ((paths.beats_json, beats or BEATS), (paths.chords_json, chords or CHORDS),
                       (paths.melody_json, melody or MELODY)):
        path.write_text(json.dumps(data), encoding="utf-8")


def test_leadsheet_step_uses_key_from_chords_when_chosen(tmp_path):
    import json

    from guitarize.analyze.pipeline import run_analysis

    paths, cfg = _paths_cfg(tmp_path)
    kc = {"tonic": "G", "mode": "major", "score": 1.0, "chroma_key_score": 0.6, "used": True}
    _write_analysis(paths, chords={**CHORDS, "key": {"tonic": "D", "mode": "major", "r": 0.7}, "key_from_chords": kc})
    run_analysis(paths, cfg, ["leadsheet"])
    ls = json.loads(paths.leadsheet_json.read_text(encoding="utf-8"))
    assert (ls["key"], ls["mode"]) == ("G", "major")


def test_short_bar_message_says_unsupported_not_shift():
    import pytest

    with pytest.raises(ValueError) as e:
        build_leadsheet({**BEATS, "short_bars": [5]}, CHORDS, MELODY, ("C", "major"), {})
    msg = str(e.value)
    assert "2/4 or 3/4" in msg and "not supported" in msg
    assert "mis-detected" in msg and "downbeat_shift" in msg


SIG = {"n_beats": 0, "pad_beats": 0, "pickup_beats": 0, "feel": "straight"}


def test_stale_chords_or_melody_rejected():
    import pytest

    stale = {**SIG, "n_beats": 99}
    with pytest.raises(ValueError, match=r"chords\.json was computed from an older beats\.json; re-run --steps chords"):
        build_leadsheet(BEATS, {**CHORDS, "beats_signature": stale}, {**MELODY, "beats_signature": SIG},
                        ("C", "major"), {})
    with pytest.raises(ValueError, match=r"melody\.json was computed from an older beats\.json; re-run --steps melody"):
        build_leadsheet(BEATS, {**CHORDS, "beats_signature": SIG}, {**MELODY, "beats_signature": stale},
                        ("C", "major"), {})
    ls = build_leadsheet(BEATS, {**CHORDS, "beats_signature": SIG}, {**MELODY, "beats_signature": SIG},
                         ("C", "major"), {})
    assert len(ls["bars"]) == 2


def test_signature_includes_feel():
    import pytest

    from guitarize.analyze.beats import beats_signature

    assert beats_signature(BEATS) == SIG
    # feel 变了（beats 重跑后直拍 ↔ shuffle）：melody 的量化格（十六分 / 三连音）都已过期
    shuffled, shuffle_sig = {**BEATS, "feel": "shuffle"}, {**SIG, "feel": "shuffle"}
    with pytest.raises(ValueError, match=r"chords\.json was computed from an older beats\.json; re-run --steps chords"):
        build_leadsheet(shuffled, {**CHORDS, "beats_signature": SIG}, {**MELODY, "beats_signature": shuffle_sig},
                        ("C", "major"), {})
    with pytest.raises(ValueError, match=r"melody\.json was computed from an older beats\.json; re-run --steps melody"):
        build_leadsheet(shuffled, {**CHORDS, "beats_signature": shuffle_sig}, {**MELODY, "beats_signature": SIG},
                        ("C", "major"), {})


def test_signature_without_feel_only_warns(capsys):
    old_sig = {k: v for k, v in SIG.items() if k != "feel"}  # feel 加进签名之前写出的文件
    ls = build_leadsheet(BEATS, {**CHORDS, "beats_signature": old_sig}, {**MELODY, "beats_signature": old_sig},
                         ("C", "major"), {})
    assert len(ls["bars"]) == 2
    err = capsys.readouterr().err
    assert "chords.json" in err and "melody.json" in err and "feel" in err
    # 没有 feel 但其它字段对不上：照样报错
    import pytest

    with pytest.raises(ValueError, match=r"chords\.json was computed from an older beats\.json"):
        build_leadsheet(BEATS, {**CHORDS, "beats_signature": {**old_sig, "n_beats": 99}},
                        {**MELODY, "beats_signature": old_sig}, ("C", "major"), {})


def test_missing_signature_only_warns(capsys):
    ls = build_leadsheet(BEATS, CHORDS, MELODY, ("C", "major"), {})
    assert len(ls["bars"]) == 2
    err = capsys.readouterr().err
    assert "chords.json" in err and "melody.json" in err and "beats_signature" in err


def _padded_beats():
    from guitarize.analyze.beats import assemble_beats

    # 弱起 3 拍：第 1 小节补 1 拍休止（pad 1），之后两个完整小节
    return assemble_beats([0.5 + 0.5 * k for k in range(11)], [1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], "straight")


def test_pipeline_stamps_beats_signature(tmp_path, monkeypatch):
    import json

    import numpy as np

    from guitarize import theory
    from guitarize.analyze import pipeline
    from guitarize.analyze.beats import beats_signature

    paths, cfg = _paths_cfg(tmp_path)
    beats = _padded_beats()
    _write_analysis(paths, beats=beats)
    cb = np.zeros((12, 11))
    cb[list(theory.chord_tones("C")), :] = 1 / np.sqrt(3)
    monkeypatch.setattr(pipeline.features, "load_mix", lambda wavs: np.zeros(10))
    monkeypatch.setattr(pipeline.features, "beat_sync_chroma", lambda y, times: cb)
    monkeypatch.setattr(pipeline, "extract_melody", lambda wav, bj, pcs, **kw: {"grid": "sixteenth", "notes": []})
    pipeline.run_analysis(paths, cfg, ["chords", "melody", "leadsheet"])
    sig = beats_signature(beats)
    assert sig == {"n_beats": 11, "pad_beats": 1, "pickup_beats": 3, "feel": "straight"}
    for path in (paths.chords_json, paths.melody_json):
        assert json.loads(path.read_text(encoding="utf-8"))["beats_signature"] == sig


def _score_from_events(events_spec):
    """events_spec: [(起点拍, 时值拍, [midi...])] → 声部 0 的 Score（3 个 4/4 小节，不用 alphaTab）。"""
    from guitarize.verify.model import Bar, Event, Note, Score

    events = [Event(bar=int(s // 4), voice=0, start=s, dur=d, written_start=s, written_dur=d,
                    notes=tuple(Note(i + 1, 0, m, False, False) for i, m in enumerate(sorted(ms, reverse=True))),
                    chord=None, pick=None, brush=None, rest=False)
              for s, d, ms in events_spec]
    bars = [Bar(i, 4.0 * i, 4.0, None, "none") for i in range(3)]
    return Score("t", 120.0, 0, [64, 59, 55, 50, 45, 40], bars, events)


def test_padded_leadsheet_lines_up_with_verify_metrics(tmp_path):
    """契约：带弱起（pad 1、pickup 3）的 beats.json → leadsheet.json → verify 读参考并评分，
    第 2 小节第 1 拍的参考和弦 / 旋律对上谱面第 2 小节第 1 拍。"""
    from guitarize.analyze.beats import beats_signature
    from guitarize.analyze.pipeline import run_analysis
    from guitarize.config import Thresholds
    from guitarize.verify import metrics
    from guitarize.verify.report import _load_json

    beats = _padded_beats()
    assert (beats["pad_beats"], beats["pickup_beats"]) == (1, 3)
    sig = beats_signature(beats)

    def seg(bar, beat, dur, label):
        return {"bar": bar, "beat": beat, "dur_beats": dur, "label": label, "confidence": "high",
                "votes": {"template": label, "web": label}}

    chords = {"segments": [seg(1, 2, 3, "G"), seg(2, 1, 1, "C"), seg(2, 2, 3, "F"), seg(3, 1, 4, "G")],
              "key": {"tonic": "C", "mode": "major", "r": None}, "beats_signature": sig}
    melody = {"grid": "sixteenth", "beats_signature": sig, "key": {"tonic": "C", "mode": "major"}, "notes": [
        {"start_beats": s, "dur_beats": d, "midi": m, "out_of_key": False, "conf": 0.9}
        for s, d, m in ((1.0, 2.0, 67), (3.0, 1.0, 71), (4.0, 1.0, 72), (5.0, 3.0, 69), (8.0, 4.0, 71))]}
    paths, cfg = _paths_cfg(tmp_path)
    _write_analysis(paths, chords=chords, melody=melody, beats=beats)
    run_analysis(paths, cfg, ["leadsheet"])
    leadsheet = _load_json(paths.leadsheet_json)  # run_verify 读参考的同一函数
    assert [b["bar"] for b in leadsheet["bars"]] == [1, 2, 3]
    assert [c["beat"] for c in leadsheet["bars"][0]["chords"]] == [2]
    assert [[m["start_beats"] for m in b["melody"]] for b in leadsheet["bars"]] == [[1.0, 3.0], [4.0, 5.0], [8.0]]

    # 谱面：第 1 小节第 1 拍是补的休止；第 2 小节第 1 拍 C（旋律 C5），第 2–4 拍 F（旋律 A4）
    score = _score_from_events([(1.0, 2.0, [55, 59, 62, 67]), (3.0, 1.0, [55, 59, 62, 71]),
                                (4.0, 1.0, [60, 64, 67, 72]), (5.0, 3.0, [53, 57, 60, 69]),
                                (8.0, 4.0, [55, 59, 62, 71])])
    bars = metrics.score_bars(score, leadsheet, "finger")
    assert [(b.chord_agree, b.chord_total) for b in bars] == [(3, 3), (4, 4), (4, 4)]
    assert [(b.mel_hit, b.mel_est) for b in bars] == [(2, 2), (2, 2), (1, 1)]
    s = metrics.summarize(bars, [], Thresholds(), "finger")
    assert s["chord_agreement"] == 1.0 and s["melody_f1"] == 1.0

    # 把第 2 小节第 1 拍的参考改成 Am：问题恰好记在谱面第 2 小节的第 1 拍
    leadsheet["bars"][1]["chords"][0]["label"] = "Am"
    bars = metrics.score_bars(score, leadsheet, "finger")
    assert (bars[1].chord_agree, bars[1].chord_total) == (3, 4)
    assert bars[1].chord_issues[0].startswith("拍 1：参考 Am")


def _sig_chords_melody(**melody_extra):
    return {**CHORDS, "beats_signature": SIG}, {**MELODY, "beats_signature": SIG, **melody_extra}


def test_melody_computed_in_another_key_is_rejected():
    import pytest

    chords, melody = _sig_chords_melody(key={"tonic": "G", "mode": "major"})
    with pytest.raises(ValueError, match=r"melody\.json was computed in G major; re-run --steps melody"):
        build_leadsheet(BEATS, chords, melody, ("C", "major"), {})
    with pytest.raises(ValueError, match=r"melody\.json was computed in G major; re-run --steps melody"):
        build_leadsheet(BEATS, chords, melody, ("G", "minor"), {})  # 同主音不同调式也算不同
    ls = build_leadsheet(BEATS, chords, melody, ("G", "major"), {})
    assert (ls["key"], ls["mode"]) == ("G", "major")
    flat, melody_flat = _sig_chords_melody(key={"tonic": "Bb", "mode": "minor"})  # 升降号写法不同不算不同
    assert build_leadsheet(BEATS, flat, melody_flat, ("A#", "minor"), {})["key"] == "A#"


def test_melody_without_key_only_warns(capsys):
    chords, melody = _sig_chords_melody()
    ls = build_leadsheet(BEATS, chords, melody, ("C", "major"), {})
    assert len(ls["bars"]) == 2
    err = capsys.readouterr().err
    assert "melody.json" in err and "no key" in err and "--steps melody" in err
    chords, melody = _sig_chords_melody(key={"tonic": "C", "mode": "major"})
    build_leadsheet(BEATS, chords, melody, ("C", "major"), {})
    assert capsys.readouterr().err == ""


def test_pipeline_records_melody_key_and_catches_a_later_key_change(tmp_path, monkeypatch):
    import json

    import pytest

    from guitarize import theory
    from guitarize.analyze import pipeline

    paths, cfg = _paths_cfg(tmp_path)
    beats = _padded_beats()
    _write_analysis(paths, beats=beats)
    cb = np.zeros((12, 11))
    cb[list(theory.chord_tones("C")), :] = 1 / np.sqrt(3)
    monkeypatch.setattr(pipeline.features, "load_mix", lambda wavs: np.zeros(10))
    monkeypatch.setattr(pipeline.features, "beat_sync_chroma", lambda y, times: cb)
    seen = []
    monkeypatch.setattr(pipeline, "extract_melody",
                        lambda wav, bj, pcs, **kw: seen.append(pcs) or {"grid": "sixteenth", "notes": []})
    pipeline.run_analysis(paths, cfg, ["chords", "melody", "leadsheet"])
    chords = json.loads(paths.chords_json.read_text(encoding="utf-8"))
    melody = json.loads(paths.melody_json.read_text(encoding="utf-8"))
    tonic, mode = pipeline.chosen_key(chords)
    assert melody["key"] == {"tonic": tonic, "mode": mode}
    assert seen == [pipeline.scale_pcs(tonic, mode)]

    # 之后 chords.json 的调变了（例如改了 overrides.key 重跑 chords），只重跑 leadsheet → 报错并指明要重跑 melody
    other = "G" if (tonic, mode) != ("G", "major") else "D"
    chords["key"] = {"tonic": other, "mode": "major", "r": None}
    chords.pop("key_from_chords", None)
    paths.chords_json.write_text(json.dumps(chords), encoding="utf-8")
    with pytest.raises(ValueError, match=rf"melody\.json was computed in {tonic} {mode}; re-run --steps melody"):
        pipeline.run_analysis(paths, cfg, ["leadsheet"])


# ---------- overrides.chords：人耳纠正的和弦，在 leadsheet 步骤套用 ----------

def _seg(beat, dur, label):
    return {"beat": beat, "dur_beats": dur, "label": label, "confidence": "manual", "votes": {"manual": label}}


def _override_bar(value, bar=1):
    ls = build_leadsheet(BEATS, CHORDS, MELODY, ("C", "major"), {}, {bar: value})
    return ls["bars"][bar - 1]["chords"]


@pytest.mark.parametrize("value, expected", [
    ("N", [_seg(1, 4, "N")]),                                    # 整小节无和弦
    ("B", [_seg(1, 4, "B")]),                                    # 1 个标签 = 整小节
    ("E F#", [_seg(1, 2, "E"), _seg(3, 2, "F#")]),               # 2 个标签各 2 拍
    ("  E   F#  ", [_seg(1, 2, "E"), _seg(3, 2, "F#")]),         # 多余空白无所谓
    ("C G Am F", [_seg(1, 1, "C"), _seg(2, 1, "G"), _seg(3, 1, "Am"), _seg(4, 1, "F")]),  # 4 个标签各 1 拍
    ("E N", [_seg(1, 2, "E"), _seg(3, 2, "N")]),
    ("NC", [_seg(1, 4, "N")]),                                   # N.C. / NC 都归一成 N
    ("N.C. E", [_seg(1, 2, "N"), _seg(3, 2, "E")]),
    ("E5", [_seg(1, 4, "E5")]),                                  # 强力和弦
    ("C#m7/G#", [_seg(1, 4, "C#m7/G#")]),                        # 转位
    (["C#m7", "C#m7", "F#", "F#"], [_seg(1, 2, "C#m7"), _seg(3, 2, "F#")]),        # 列表 = 逐拍
    (["C", "G", "G", "C"], [_seg(1, 1, "C"), _seg(2, 2, "G"), _seg(4, 1, "C")]),
    (["N", "N", "E", "E"], [_seg(1, 2, "N"), _seg(3, 2, "E")]),
])
def test_chord_override_value_forms(value, expected):
    assert _override_bar(value) == expected


def test_chord_override_merges_consecutive_equal_labels():
    assert _override_bar("C C") == [_seg(1, 4, "C")]
    assert _override_bar("C C C C") == [_seg(1, 4, "C")]
    assert _override_bar(["F", "F", "F", "F"]) == [_seg(1, 4, "F")]
    # 只合并相邻的：C G C G 不会合成一段
    assert [s["label"] for s in _override_bar("C G C G")] == ["C", "G", "C", "G"]


def test_chord_override_bar_keys_int_or_numeric_str():
    for key in (2, "2", " 2 "):
        ls = build_leadsheet(BEATS, CHORDS, MELODY, ("C", "major"), {}, {key: "E F#"})
        assert [s["label"] for s in ls["bars"][1]["chords"]] == ["E", "F#"]


@pytest.mark.parametrize("bar, shown", [(3, "3"), (0, "0"), (-1, "-1"), ("7", "7"), ("x", "x"), (1.5, "1.5"), (True, "True")])
def test_chord_override_unknown_bar_names_the_bar(bar, shown):
    with pytest.raises(ValueError, match=rf"overrides\.chords.*bar {shown}\b"):
        build_leadsheet(BEATS, CHORDS, MELODY, ("C", "major"), {}, {bar: "C"})


def test_chord_override_duplicate_bar_rejected():
    with pytest.raises(ValueError, match=r"bar 2.*twice"):
        build_leadsheet(BEATS, CHORDS, MELODY, ("C", "major"), {}, {2: "C", "2": "G"})


@pytest.mark.parametrize("value, bad", [("H", "H"), ("C Zz", "Zz"), (["C", "C", "Q#", "C"], "Q#"), ("", "")])
def test_chord_override_bad_label_names_bar_and_label(value, bad):
    with pytest.raises(ValueError) as e:
        build_leadsheet(BEATS, CHORDS, MELODY, ("C", "major"), {}, {2: value})
    msg = str(e.value)
    assert "bar 2" in msg and repr(bad) in msg and "overrides.chords" in msg


@pytest.mark.parametrize("value", [None, 5, 5.5, True, {"a": "C"}, ["C", None, "C", "C"], ["C", "C", 5, "C"]])
def test_chord_override_non_label_values_rejected(value):
    with pytest.raises(ValueError, match=r"overrides\.chords.*bar 2"):
        build_leadsheet(BEATS, CHORDS, MELODY, ("C", "major"), {}, {2: value})


@pytest.mark.parametrize("value", ["C D E", "C D E F G", "C D E F G A B C"])
def test_chord_override_string_count_must_divide_beats(value):
    with pytest.raises(ValueError, match=r"bar 2.*labels.*4 beats"):
        build_leadsheet(BEATS, CHORDS, MELODY, ("C", "major"), {}, {2: value})


@pytest.mark.parametrize("value", [["C"], ["C", "G"], ["C", "G", "Am"], ["C", "G", "Am", "F", "C"], []])
def test_chord_override_list_length_must_equal_beats_per_bar(value):
    with pytest.raises(ValueError, match=r"bar 2.*4 beats"):
        build_leadsheet(BEATS, CHORDS, MELODY, ("C", "major"), {}, {2: value})


def test_chord_overrides_must_be_a_mapping():
    with pytest.raises(ValueError, match=r"overrides\.chords must be a mapping"):
        build_leadsheet(BEATS, CHORDS, MELODY, ("C", "major"), {}, ["C", "G"])


def test_chord_override_beats_per_bar_follows_time_signature():
    beats3 = {**BEATS, "time_signature": [3, 4], "bars": [{**b, "beats": 3} for b in BEATS["bars"]]}

    def build(value):
        return build_leadsheet(beats3, CHORDS, MELODY, ("C", "major"), {}, {1: value})["bars"][0]["chords"]

    assert build("E F# G") == [_seg(1, 1, "E"), _seg(2, 1, "F#"), _seg(3, 1, "G")]
    assert build("E") == [_seg(1, 3, "E")]
    with pytest.raises(ValueError, match=r"bar 1.*3 beats"):
        build("E F#")           # 2 个标签分不了 3 拍
    with pytest.raises(ValueError, match=r"bar 1.*3 beats"):
        build(["E", "F#", "G", "A"])


def _four_bar_inputs():
    beats = {**BEATS, "bars": [{"bar": n, "t_start": 2.0 * (n - 1), "t_end": 2.0 * n, "beats": 4} for n in range(1, 5)]}
    chords = {"segments": [
        {"bar": n, "beat": 1, "dur_beats": 4, "label": lab, "confidence": "high", "votes": {"template": lab, "web": lab}}
        for n, lab in enumerate(["C", "F", "G", "Am"], start=1)]}
    melody = {"grid": "sixteenth", "notes": [
        {"start_beats": 4.0 * k, "dur_beats": 1.0, "midi": 60 + k, "out_of_key": False, "conf": 0.9} for k in range(4)]}
    return beats, chords, melody


def test_chord_overrides_change_only_the_named_bars():
    beats, chords, melody = _four_bar_inputs()
    frozen = copy.deepcopy(chords)
    base = build_leadsheet(beats, chords, melody, ("C", "major"), {1: "イントロ"})
    got = build_leadsheet(beats, chords, melody, ("C", "major"), {1: "イントロ"}, {2: "N", "4": ["Am", "Am", "E", "E"]})
    assert chords == frozen                                           # 不改输入的 chords.json 内容
    assert {k: v for k, v in got.items() if k != "bars"} == {k: v for k, v in base.items() if k != "bars"}
    for n in (0, 2):                                                  # 小节 1、3 原样
        assert got["bars"][n] == base["bars"][n]
    for n in (1, 3):                                                  # 小节 2、4 只有 chords 变了
        assert {k: v for k, v in got["bars"][n].items() if k != "chords"} == \
               {k: v for k, v in base["bars"][n].items() if k != "chords"}
    assert got["bars"][1]["chords"] == [_seg(1, 4, "N")]              # 原先的 F（high）整个被替换
    assert got["bars"][3]["chords"] == [_seg(1, 2, "Am"), _seg(3, 2, "E")]
    assert all(set(c) == {"beat", "dur_beats", "label", "confidence", "votes"} for b in got["bars"] for c in b["chords"])


def test_no_or_empty_chord_overrides_change_nothing():
    beats, chords, melody = _four_bar_inputs()
    base = build_leadsheet(beats, chords, melody, ("C", "major"), {})
    assert build_leadsheet(beats, chords, melody, ("C", "major"), {}, None) == base
    assert build_leadsheet(beats, chords, melody, ("C", "major"), {}, {}) == base


def test_chord_override_notes_go_to_stderr(capsys):
    beats, chords, melody = _four_bar_inputs()
    build_leadsheet(beats, chords, melody, ("C", "major"), {}, {"3": "E F#", 4: ["C#m7", "C#m7", "F#", "F#"]})
    err = capsys.readouterr().err
    notes = [line for line in err.splitlines() if "overridden" in line]   # 夹具没有签名，另有无关警告
    assert notes == ["[leadsheet] bar 3 chords overridden: E F#",
                     "[leadsheet] bar 4 chords overridden: C#m7 C#m7 F# F#"]
    build_leadsheet(beats, chords, melody, ("C", "major"), {})
    assert "overridden" not in capsys.readouterr().err


def test_chord_override_errors_are_raised_before_any_note(capsys):
    beats, chords, melody = _four_bar_inputs()
    with pytest.raises(ValueError, match="bar 9"):
        build_leadsheet(beats, chords, melody, ("C", "major"), {}, {2: "E", 9: "G"})
    assert "overridden" not in capsys.readouterr().err


def test_chord_override_covers_only_beats_that_exist():
    beats, chords, melody = _four_bar_inputs()
    beats["bars"][3]["beats"] = 2                       # 末尾小节只有 2 拍
    got = build_leadsheet(beats, chords, melody, ("C", "major"), {}, {4: ["Am", "Am", "E", "E"]})
    assert got["bars"][3]["chords"] == [_seg(1, 2, "Am")]
    got = build_leadsheet(beats, chords, melody, ("C", "major"), {}, {4: "Am E"})
    assert got["bars"][3]["chords"] == [_seg(1, 2, "Am")]


def test_chord_override_in_a_padded_first_bar_skips_the_pad_beat():
    beats = _padded_beats()                              # pad 1：第 1 小节只有第 2–4 拍
    assert beats["bars"][0]["beats"] == 3 and beats["pad_beats"] == 1
    chords = {"segments": [{"bar": n, "beat": b, "dur_beats": d, "label": "C", "confidence": "high", "votes": {}}
                           for n, b, d in ((1, 2, 3), (2, 1, 4), (3, 1, 4))]}
    got = build_leadsheet(beats, chords, MELODY, ("C", "major"), {}, {1: "E F#", 2: "G"})
    assert got["bars"][0]["chords"] == [_seg(2, 1, "E"), _seg(3, 2, "F#")]   # 第 1 拍（补的休止）没有和弦
    assert got["bars"][1]["chords"] == [_seg(1, 4, "G")]
    got = build_leadsheet(beats, chords, MELODY, ("C", "major"), {}, {1: ["A", "B", "B", "B"]})
    assert got["bars"][0]["chords"] == [_seg(2, 3, "B")]


def test_summary_table_marks_manual_chords():
    beats, chords, melody = _four_bar_inputs()
    ls = build_leadsheet(beats, chords, melody, ("C", "major"), {}, {2: "E F#"})
    lines = summary_table(ls, capo=2).splitlines()
    assert "1:E(D)![manual]" in lines[2] and "3:F#(E)![manual]" in lines[2]
    assert "votes" not in lines[2] and "template=" not in lines[2]
    assert "!" not in lines[1] and "[manual]" not in lines[1]           # 没覆盖的小节照旧
    assert "template=" in summary_table(build_leadsheet(BEATS, CHORDS, MELODY, ("C", "major"), {}))


def test_leadsheet_step_applies_chord_overrides_without_touching_chords_json(tmp_path, capsys):
    import json

    from guitarize.analyze.pipeline import run_analysis
    from guitarize.config import SongConfig
    from guitarize.paths import SongPaths

    paths = SongPaths(tmp_path / "song")
    cfg = SongConfig(slug="song", title="t", artist="a", overrides={"chords": {2: "E F#", "1": "N"}})
    _write_analysis(paths, chords={**CHORDS, "key": {"tonic": "C", "mode": "major", "r": None}})
    before = paths.chords_json.read_text(encoding="utf-8")
    run_analysis(paths, cfg, ["leadsheet"])
    ls = json.loads(paths.leadsheet_json.read_text(encoding="utf-8"))
    assert ls["bars"][0]["chords"] == [_seg(1, 4, "N")]
    assert ls["bars"][1]["chords"] == [_seg(1, 2, "E"), _seg(3, 2, "F#")]
    assert paths.chords_json.read_text(encoding="utf-8") == before      # 重跑 chords 前后改动都不落到 chords.json
    assert "[leadsheet] bar 2 chords overridden: E F#" in capsys.readouterr().err
    # 没有 overrides.chords 的歌：leadsheet 里仍是分析出的和弦
    run_analysis(paths, SongConfig(slug="song", title="t", artist="a"), ["leadsheet"])
    ls = json.loads(paths.leadsheet_json.read_text(encoding="utf-8"))
    assert [c["label"] for c in ls["bars"][1]["chords"]] == ["G", "N"]


def test_song_yaml_chord_overrides_round_trip(tmp_path):
    """song.yaml 里按文档写的各种形式（N、带 # 的标签、流式列表）经 load_song_config 后都能被 build_leadsheet 接受。"""
    from guitarize.config import load_song_config
    from guitarize.paths import SongPaths

    paths = SongPaths(tmp_path / "song")
    paths.root.mkdir()
    paths.config.write_text(
        "slug: song\ntitle: t\nartist: a\noverrides:\n  key: C\n  chords:\n"
        "    1: N              # 整小节\n    2: E F#\n    3: B\n    4: [C#m7, C#m7, F#, F#]\n",
        encoding="utf-8")
    cfg = load_song_config(paths)
    assert cfg.overrides["chords"] == {1: "N", 2: "E F#", 3: "B", 4: ["C#m7", "C#m7", "F#", "F#"]}
    beats, chords, melody = _four_bar_inputs()
    got = build_leadsheet(beats, chords, melody, ("C", "major"), {}, cfg.overrides["chords"])
    assert [[(c["beat"], c["dur_beats"], c["label"]) for c in b["chords"]] for b in got["bars"]] == [
        [(1, 4, "N")], [(1, 2, "E"), (3, 2, "F#")], [(1, 4, "B")], [(1, 2, "C#m7"), (3, 2, "F#")]]


def test_melody_step_reads_the_lead_vocal(tmp_path, monkeypatch):
    # 旋律只从居中的主唱识别（伴唱摆在左右会把旋律带偏）；lead.wav 不在时从人声分轨抽出来；
    # overrides.melody_source: vocals 可改回整条人声
    import dataclasses

    import numpy as np
    import soundfile as sf

    from guitarize.analyze import pipeline

    paths, cfg = _paths_cfg(tmp_path)
    _write_analysis(paths, beats=_padded_beats(), chords={**CHORDS, "key": {"tonic": "C", "mode": "major", "r": 0.9}})
    paths.stems_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(22050) / 22050
    sf.write(paths.stem("vocals"), np.stack([np.sin(2 * np.pi * 220 * t)] * 2, axis=1), 22050)
    used = []
    monkeypatch.setattr(pipeline, "extract_melody", lambda wav, bj, pcs, **kw: used.append(wav) or {"grid": "sixteenth", "notes": []})
    pipeline.run_analysis(paths, cfg, ["melody"])
    assert used[-1] == paths.stem("lead") and paths.stem("lead").exists()
    assert json.loads(paths.melody_json.read_text(encoding="utf-8"))["source"] == "lead"   # 校验据此选对照的分轨
    cfg2 = dataclasses.replace(cfg, overrides={**cfg.overrides, "melody_source": "vocals"})
    pipeline.run_analysis(paths, cfg2, ["melody"])
    assert used[-1] == paths.stem("vocals")
    assert json.loads(paths.melody_json.read_text(encoding="utf-8"))["source"] == "vocals"
    cfg3 = dataclasses.replace(cfg, overrides={**cfg.overrides, "melody_source": "mix"})
    with pytest.raises(ValueError, match=r'overrides\.melody_source must be "lead" or "vocals", got \'mix\''):
        pipeline.run_analysis(paths, cfg3, ["melody"])


# ---------- overrides.melody：人耳纠正的旋律，在 leadsheet 步骤套用 ----------

def _notes(bar):
    return [(round(m["start_beats"], 4), round(m["dur_beats"], 4), m["midi"]) for m in bar["melody"]]


def test_melody_override_replaces_the_bar_snapped_to_twelfths():
    beats, chords, melody = _four_bar_inputs()
    base = build_leadsheet(beats, chords, melody, ("C", "major"), {})
    got = build_leadsheet(beats, chords, melody, ("C", "major"), {}, None,
                          {1: [[2.67, 0.33, "F#4"], [3, 1, "F#4"], [4, 0.33, "B3"], [4.33, 1.33, "F#4"]], "3": []})
    assert _notes(got["bars"][0]) == [(1.6667, 0.3333, 66), (2.0, 1.0, 66), (3.0, 0.3333, 59), (3.3333, 1.3333, 66)]
    assert all(m["manual"] is True and m["conf"] == 1.0 for m in got["bars"][0]["melody"])
    assert [m["out_of_key"] for m in got["bars"][0]["melody"]] == [True, True, False, True]   # C 大调里 F# 是调外音
    assert got["bars"][2]["melody"] == [] and got["bars"][2]["melody_manual"] is True        # 这一小节没有人声旋律
    assert got["bars"][0]["melody_manual"] is True
    for n in (1, 3):                                                   # 没覆盖的小节原样
        assert got["bars"][n] == base["bars"][n] and "melody_manual" not in got["bars"][n]
    assert {k: v for k, v in got["bars"][0].items() if k not in ("melody", "melody_manual")} == \
           {k: v for k, v in base["bars"][0].items() if k != "melody"}
    quarter = build_leadsheet(beats, chords, melody, ("C", "major"), {}, None, {2: [[1.25, 0.25, "Bb3"], [1.5, 2.5, "c4"]]})
    assert _notes(quarter["bars"][1]) == [(4.25, 0.25, 58), (4.5, 2.5, 60)]           # ¼ 拍同样能写


@pytest.mark.parametrize("name, midi", [("F#4", 66), ("Bb3", 58), ("C4", 60), ("c4", 60), ("B#3", 60), ("Cb4", 59),
                                        ("E♭4", 63), ("G♯2", 44), ("A0", 21)])
def test_note_names_with_octave(name, midi):
    from guitarize import theory
    assert theory.note_midi(name) == midi


@pytest.mark.parametrize("raw, bar, what", [
    ({9: []}, "9", "no such bar"), ({0: []}, "0", "no such bar"), ({"x": []}, "x", "not a bar number"),
    ({2: [[1, 1, "H4"]]}, "2", "bad pitch 'H4'"), ({2: [[1, 1, "F#"]]}, "2", "bad pitch 'F#'"),
    ({2: [[1, 1, 66]]}, "2", "bad pitch 66"), ({2: [[1, 1, "F##4"]]}, "2", "bad pitch"),
    ({2: [[1, -1, "C4"]]}, "2", "duration must be positive"), ({2: [[1, 0, "C4"]]}, "2", "duration must be positive"),
    ({2: [[1, 0.01, "C4"]]}, "2", "duration must be positive"),
    ({2: [[5, 1, "C4"]]}, "2", "outside the bar"), ({2: [[0.5, 1, "C4"]]}, "2", "outside the bar"),
    ({2: [[1, 1]]}, "2", "each note is"), ({2: ["C4"]}, "2", "each note is"), ({2: [[True, 1, "C4"]]}, "2", "beat must be a number"),
    ({2: [[1, "1", "C4"]]}, "2", "duration must be a number"),
    ({2: "C4"}, "2", "expected a list"), ({2: None}, "2", "expected a list"),
    ({2: [[1, 2, "C4"], [2, 1, "D4"]]}, "2", "overlap"),
    ({2: [], "2": []}, "2", "twice"),
])
def test_melody_override_errors_name_the_bar(raw, bar, what):
    beats, chords, melody = _four_bar_inputs()
    with pytest.raises(ValueError) as e:
        build_leadsheet(beats, chords, melody, ("C", "major"), {}, None, raw)
    msg = str(e.value)
    assert msg.startswith(f"overrides.melody bar {bar}") and what in msg, msg


@pytest.mark.parametrize("text, what", [("[[.nan, 1, C4]]", "beat must be a finite number, got nan"),
                                        ("[[1, .inf, C4]]", "duration must be a finite number, got inf"),
                                        ("[[-.inf, 1, C4]]", "beat must be a finite number, got -inf")])
def test_melody_override_nan_and_inf_name_the_bar(text, what):
    # YAML 的 .nan / .inf 是合法的浮点数：报错要指出小节，而不是取整时冒出一个不知在哪的 ValueError / OverflowError
    import yaml

    beats, chords, melody = _four_bar_inputs()
    with pytest.raises(ValueError) as e:
        build_leadsheet(beats, chords, melody, ("C", "major"), {}, None, yaml.safe_load(f"2: {text}"))
    msg = str(e.value)
    assert msg.startswith("overrides.melody bar 2") and what in msg, msg


def test_melody_overrides_must_be_a_mapping():
    beats, chords, melody = _four_bar_inputs()
    with pytest.raises(ValueError, match=r"overrides\.melody must be a mapping"):
        build_leadsheet(beats, chords, melody, ("C", "major"), {}, None, [[1, 1, "C4"]])


def test_melody_override_errors_are_raised_before_anything_is_applied(capsys):
    beats, chords, melody = _four_bar_inputs()
    with pytest.raises(ValueError, match="bar 9"):
        build_leadsheet(beats, chords, melody, ("C", "major"), {}, {2: "E"}, {1: [[1, 1, "C4"]], 9: []})
    assert "overridden" not in capsys.readouterr().err                   # 和弦覆盖也还没套用、没打印


def test_melody_override_notes_go_to_stderr(capsys):
    beats, chords, melody = _four_bar_inputs()
    build_leadsheet(beats, chords, melody, ("C", "major"), {}, None, {2: [[1, 1, "C4"], [2.5, 0.5, "Eb4"]], 4: []})
    notes = [line for line in capsys.readouterr().err.splitlines() if "overridden" in line]
    assert notes == ["[leadsheet] bar 2 melody overridden: C4 D#4", "[leadsheet] bar 4 melody overridden: (no melody)"]


def test_summary_table_marks_manual_melody_bars():
    beats, chords, melody = _four_bar_inputs()
    ls = build_leadsheet(beats, chords, melody, ("C", "major"), {}, None, {2: [[1, 1, "F#4"]], 3: []})
    lines = summary_table(ls).splitlines()
    assert lines[2].endswith("| F#4* !") and lines[3].endswith("| !")
    assert not lines[1].endswith("!") and not lines[4].endswith("!")


def test_leadsheet_step_applies_melody_overrides_from_song_yaml(tmp_path, capsys):
    import json

    from guitarize.analyze.pipeline import run_analysis
    from guitarize.config import load_song_config
    from guitarize.paths import SongPaths

    paths = SongPaths(tmp_path / "song")
    beats, chords, melody = _four_bar_inputs()
    _write_analysis(paths, beats=beats, chords={**chords, "key": {"tonic": "C", "mode": "major", "r": None}},
                    melody=melody)
    paths.config.write_text(
        "slug: song\ntitle: t\nartist: a\noverrides:\n  melody:\n"
        "    1: [[2.67, 0.33, F#4], [3, 1, F#4], [4, 0.33, B3], [4.33, 1.33, F#4]]\n"
        "    3: []          # 这一小节没有人声旋律\n", encoding="utf-8")
    cfg = load_song_config(paths)
    before = paths.melody_json.read_text(encoding="utf-8")
    run_analysis(paths, cfg, ["leadsheet"])
    ls = json.loads(paths.leadsheet_json.read_text(encoding="utf-8"))
    assert [m["midi"] for m in ls["bars"][0]["melody"]] == [66, 66, 59, 66] and ls["bars"][2]["melody"] == []
    assert [m["midi"] for m in ls["bars"][1]["melody"]] == [61]           # 没覆盖的小节仍是分析出的旋律
    assert paths.melody_json.read_text(encoding="utf-8") == before        # melody.json 不动：重跑 melody 也不丢纠正
    assert "[leadsheet] bar 1 melody overridden: F#4 F#4 B3 F#4" in capsys.readouterr().err


# ---------- 连奏（legato）与尾滑（slide_out）：melody.json 的可选标记，原样带进 leadsheet ----------

def _legato_melody():
    """四小节，一拍一个音：第 2 拍的 D4 从 C4 滑上来（legato），第 3 拍的 E4 结尾下滑（slide_out），
    第 5 拍（第 2 小节第一个音）、第 9 拍（第 3 小节第一个音）也是连奏。"""
    beats, chords, _ = _four_bar_inputs()
    spec = [(0, 60, {}), (1, 62, {"legato": True}), (2, 64, {"slide_out": "down"}), (3, 65, {"legato": False}),
            (4, 67, {"legato": True, "slide_out": "up"}), (8, 69, {"legato": True}), (9, 71, {"slide_out": None})]
    notes = [{"start_beats": float(s), "dur_beats": 1.0, "midi": m, "out_of_key": False, "conf": 0.9, **extra}
             for s, m, extra in spec]
    return beats, chords, {"grid": "sixteenth", "notes": notes}


def _flags(ls):
    return [(m["midi"], m.get("legato"), m.get("slide_out")) for b in ls["bars"] for m in b["melody"]]


def test_legato_and_slide_out_pass_through_to_the_leadsheet():
    beats, chords, melody = _legato_melody()
    ls = build_leadsheet(beats, chords, melody, ("C", "major"), {})
    assert _flags(ls) == [(60, None, None), (62, True, None), (64, None, "down"), (65, None, None),
                          (67, True, "up"), (69, True, None), (71, None, None)]
    # 没有标记的音与旧文件一样只有这五个键（legato: false、slide_out: null 不写）
    assert set(ls["bars"][0]["melody"][0]) == set(ls["bars"][0]["melody"][3]) == \
           {"start_beats", "dur_beats", "midi", "conf", "out_of_key"}


def test_old_melody_json_without_flags_is_unchanged():
    beats, chords, melody = _four_bar_inputs()
    ls = build_leadsheet(beats, chords, melody, ("C", "major"), {})
    assert all(set(m) == {"start_beats", "dur_beats", "midi", "conf", "out_of_key"} for b in ls["bars"] for m in b["melody"])


def test_legato_needs_its_sung_predecessor_in_the_leadsheet():
    # 覆盖的音不是连奏；前一个音被 overrides.melody 换掉了（第 2 小节被覆盖 → 第 3 小节第一个音），
    # 或者根本没有前一个音（第一个音）：照常拨弦，legato 去掉；slide_out 只属于这个音自己，保留
    beats, chords, melody = _legato_melody()
    melody["notes"][0]["legato"] = True
    ls = build_leadsheet(beats, chords, melody, ("C", "major"), {}, None, {2: [[1, 1, "G4"], [2, 3, "A4"]]})
    assert _flags(ls) == [(60, None, None), (62, True, None), (64, None, "down"), (65, None, None),
                          (67, None, None), (69, None, None), (69, None, None), (71, None, None)]
    assert all("legato" not in m and "slide_out" not in m for m in ls["bars"][1]["melody"])


def test_bad_slide_out_names_the_note():
    beats, chords, melody = _legato_melody()
    melody["notes"][2]["slide_out"] = "sideways"
    with pytest.raises(ValueError, match=r"melody\.json note at start_beats 2 \(E4\): "
                                         r"slide_out must be \"down\" or \"up\", got 'sideways'"):
        build_leadsheet(beats, chords, melody, ("C", "major"), {})
