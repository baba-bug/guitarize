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
