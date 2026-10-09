"""闭环：合成歌曲 → 分析 → 与真值比较。验证"耳朵"本身是否可靠。"""
import numpy as np
import pytest

import synth_song as S
from guitarize import theory
from guitarize.analyze import features
from guitarize.analyze.beats import time_to_beat, track_beats
from guitarize.analyze.chords import reference_chords, template_track
from guitarize.analyze.key import estimate_key, scale_pcs
from guitarize.analyze.melody import extract_melody
from guitarize.analyze.stems import STEMS, mix_harmonic, separate
from guitarize.analyze.webchart import parse_webchart

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def song(tmp_path_factory):
    out = S.render_song(tmp_path_factory.mktemp("synth"))
    mix_harmonic(out["stems"])
    out["beats"] = track_beats(out["mix"], out["stems"])
    times = [b["t"] for b in out["beats"]["beats"]]
    out["cb"] = features.beat_sync_chroma(features.load_mix([out["stems"] / "harmonic.wav"]), times)
    return out


def truth_chord_at(t: float) -> str | None:
    k = int((t + 0.05 - S.OFFSET) // (4 * S.SPB))
    return S.PROGRESSION[k] if 0 <= k < len(S.PROGRESSION) else None


def test_beats_tempo_and_downbeats(song):
    bj = song["beats"]
    assert abs(bj["tempo_bpm"] - 120) / 120 < 0.02
    assert bj["feel"] == "straight" and bj["short_bars"] == []
    downbeats = [b["t"] for b in bj["beats"] if b["beat"] == 1]
    hits = sum(any(abs(d - S.bar_start(k)) < 0.07 for d in downbeats) for k in range(1, 17))
    assert hits >= 15


def test_key(song):
    assert estimate_key(song["cb"].mean(axis=1))[:2] == ("G", "major")


def test_template_chords_accuracy(song):
    labels, _ = template_track(song["cb"])
    pairs = [(labels[i], truth_chord_at(b["t"])) for i, b in enumerate(song["beats"]["beats"])]
    pairs = [(a, t) for a, t in pairs if t is not None]
    acc = np.mean([theory.majmin(a) == theory.majmin(t) for a, t in pairs])
    assert acc >= 0.85, acc


def test_reference_chords_with_web_chart(song):
    out = reference_chords(song["cb"], song["beats"], parse_webchart(S.truth_webchart()))
    beats = song["beats"]["beats"]
    per_beat = {}
    for s in out["segments"]:
        for k in range(s["dur_beats"]):
            per_beat[(s["bar"], s["beat"] + k)] = s
    pairs = [(per_beat[(b["bar"], b["beat"])], truth_chord_at(b["t"])) for b in beats]
    pairs = [(s, t) for s, t in pairs if t is not None]
    assert np.mean([s["confidence"] == "high" for s, _ in pairs]) >= 0.8
    assert np.mean([theory.majmin(s["label"]) == theory.majmin(t) for s, t in pairs]) >= 0.95


def test_melody_f1(song):
    mel = extract_melody(song["stems"] / "vocals.wav", song["beats"], scale_pcs("G", "major"))
    truth = [(float(time_to_beat(song["beats"], S.OFFSET + s * S.SPB)), p) for s, _, p in S.melody_truth()]
    est = [(n["start_beats"], n["midi"]) for n in mel["notes"]]
    used, hit = set(), 0
    for ts, tp in truth:
        for j, (es, ep) in enumerate(est):
            if j not in used and ep == tp and abs(es - ts) <= 1 / 3:
                used.add(j)
                hit += 1
                break
    p, r = hit / max(1, len(est)), hit / len(truth)
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    assert f1 >= 0.8, (p, r)


def test_separate_produces_all_stems(song, tmp_path):
    out = separate(song["mix"], tmp_path / "stems")
    assert set(out) == {*STEMS, "harmonic"} and all(p.exists() for p in out.values())
