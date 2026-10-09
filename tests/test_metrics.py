import copy

import numpy as np
import pytest

from conftest import demo_leadsheet
from guitarize.config import Thresholds
from guitarize.render import alphatab
from guitarize.render.synth import midi_to_wav
from guitarize.verify import metrics
from guitarize.verify.model import load_score


@pytest.fixture
def demo_scores(demo_song, tmp_path):
    out = {}
    for arr in ("finger", "strum"):
        built = alphatab.build(demo_song.atex(arr), tmp_path / "b", arr)
        out[arr] = (load_score(built["json"]), built["mid"])
    return out


def test_f1():
    assert metrics.f1(0, 0, 0, 0) is None
    assert metrics.f1(4, 4, 4, 4) == 1.0
    assert metrics.f1(2, 4, 1, 2) == pytest.approx(0.5)


def test_demo_finger_perfect(demo_scores):
    score, _ = demo_scores["finger"]
    bars = metrics.score_bars(score, demo_leadsheet(), "finger")
    s = metrics.summarize(bars, [], Thresholds(), "finger")
    assert s["chord_agreement"] == 1.0
    assert s["melody_precision"] == 1.0 and s["melody_recall"] == 1.0 and s["melody_f1"] == 1.0
    assert s["melody_octave_exact"] == 1.0
    assert s["pass"] is True
    assert metrics.problem_bars(bars, [], Thresholds(), "finger") == []


def test_wrong_reference_chord_flagged(demo_scores):
    score, _ = demo_scores["strum"]
    ls = demo_leadsheet()
    ls["bars"][0]["chords"] = [{"beat": 1, "dur_beats": 4, "label": "Am", "confidence": "high"}]
    bars = metrics.score_bars(score, ls, "strum")
    assert bars[0].chord_agree == 0 and bars[0].chord_total == 4
    assert "参考 Am" in bars[0].chord_issues[0]
    assert metrics.problem_bars(bars, [], Thresholds(), "strum") == [1]


def test_low_confidence_beats_excluded(demo_scores):
    score, _ = demo_scores["strum"]
    ls = demo_leadsheet()
    ls["bars"][0]["chords"] = [{"beat": 1, "dur_beats": 4, "label": "Am", "confidence": "low"}]
    bars = metrics.score_bars(score, ls, "strum")
    assert bars[0].chord_total == 0 and bars[0].low_conf_beats == [1, 2, 3, 4]


def test_melody_wrong_note(demo_scores):
    score, _ = demo_scores["finger"]
    ls = demo_leadsheet()
    ls["bars"][0]["melody"][2]["midi"] = 69  # 参考第 1 小节第 3 拍改成 A4
    bars = metrics.score_bars(score, ls, "finger")
    assert (bars[0].mel_hit, bars[0].mel_est) == (3, 4)
    assert (bars[0].mel_found, bars[0].mel_ref) == (3, 4)
    assert any("漏掉骨架音" in x for x in bars[0].mel_issues)
    assert any("多余/错音" in x for x in bars[0].mel_issues)


def test_melody_octave_tolerated(demo_scores):
    score, _ = demo_scores["finger"]
    ls = copy.deepcopy(demo_leadsheet())
    for b in ls["bars"]:
        for n in b["melody"]:
            n["midi"] -= 12
    s = metrics.summarize(metrics.score_bars(score, ls, "finger"), [], Thresholds(), "finger")
    assert s["melody_f1"] == 1.0 and s["melody_octave_exact"] == 0.0


def test_bars_without_reference_melody_not_scored(demo_scores):
    score, _ = demo_scores["finger"]
    ls = demo_leadsheet()
    ls["bars"][0]["melody"] = []
    bars = metrics.score_bars(score, ls, "finger")
    assert bars[0].mel_est == 0 and bars[0].mel_f1 is None


def test_harmonic_similarity_self(demo_scores, tmp_path):
    score, mid = demo_scores["finger"]
    wav = midi_to_wav(mid, tmp_path / "f.wav")
    bars = metrics.score_bars(score, demo_leadsheet(), "finger", synth_wav=wav, harmonic_wav=wav)
    sims = [b.harmonic_sim for b in bars]
    assert all(s is not None and s > 0.99 for s in sims)


def test_chord_mark_mismatch(build_tex):
    from conftest import TEX_HEAD

    score = build_tex('\\chord "G" 3 0 0 0 2 3\n:1 (0.1 1.2 0.3 2.4 3.5) {ch "G" sd} |', head=TEX_HEAD)
    bars = metrics.score_bars(score, None, "strum")
    assert bars[0].mark_issues and "标记 G" in bars[0].mark_issues[0]
    s = metrics.summarize(bars, [], Thresholds(), "strum")
    assert s["chord_mark_issues"] == 1 and s["checks"]["chord_marks"] is False
    assert metrics.problem_bars(bars, [], Thresholds(), "strum") == [1]


def test_demo_strum_marks_consistent(demo_scores):
    score, _ = demo_scores["strum"]
    bars = metrics.score_bars(score, demo_leadsheet(), "strum")
    assert all(not b.mark_issues for b in bars)
    assert metrics.summarize(bars, [], Thresholds(), "strum")["pass"] is True


def test_strum_vector_and_pearson(demo_scores):
    score, _ = demo_scores["strum"]
    v = metrics.strum_vector(score.attacks(), score.bars[0])
    # 每半小节：↓ 在 0，↓ 在 1 拍（slot 3），↑ 在 1.67 拍（slot 5）
    assert list(np.flatnonzero(v)) == [0, 3, 5, 6, 9, 11]
    assert v[5] == pytest.approx(0.6) and v[0] == 1.0
    assert metrics.pearson(v, v) == pytest.approx(1.0)
    assert metrics.pearson(np.ones(12), v) is None
