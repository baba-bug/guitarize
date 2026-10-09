import numpy as np
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
    monkeypatch.setattr(pipeline, "extract_melody", lambda wav, bj, pcs: {"grid": "sixteenth", "notes": []})
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
                        lambda wav, bj, pcs: seen.append(pcs) or {"grid": "sixteenth", "notes": []})
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
