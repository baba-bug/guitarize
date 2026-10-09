import numpy as np
import soundfile as sf

from guitarize.analyze import features

SR = features.SR


def tone(midis, dur, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    return sum(np.sin(2 * np.pi * 440 * 2 ** ((m - 69) / 12) * t) for m in midis) / len(midis)


def test_load_mix_pads_and_sums(tmp_path):
    sf.write(tmp_path / "a.wav", np.ones(100) * 0.1, SR)
    sf.write(tmp_path / "b.wav", np.ones(150) * 0.2, SR)
    y = features.load_mix([tmp_path / "a.wav", tmp_path / "b.wav"])
    assert len(y) == 150
    assert np.allclose(y[:100], 0.3, atol=1e-3) and np.allclose(y[100:], 0.2, atol=1e-3)


def test_beat_sync_chroma_follows_chords():
    y = np.concatenate([tone([60, 64, 67], 1.0), tone([57, 60, 64], 1.0)])
    cb = features.beat_sync_chroma(y, [0.0, 0.5, 1.0, 1.5])
    assert cb.shape == (12, 4)
    assert np.allclose(np.linalg.norm(cb, axis=0), 1.0, atol=1e-6)
    assert set(np.argsort(cb[:, 0])[-3:]) == {0, 4, 7}
    assert set(np.argsort(cb[:, 3])[-3:]) == {9, 0, 4}
    nov = features.chroma_novelty(cb)
    assert nov[0] == 1.0 and nov[2] > nov[1] and nov[2] > nov[3]


def clicks(times, amps, dur, sr=SR):
    y = np.zeros(int(dur * sr))
    burst = np.random.default_rng(0).standard_normal(int(0.01 * sr)) * np.exp(-np.linspace(0, 8, int(0.01 * sr)))
    for t, a in zip(times, amps):
        i = int(t * sr)
        y[i:i + len(burst)] += a * burst[: len(y) - i]
    return y


def test_detect_feel():
    beats = np.arange(0.2, 8.0, 0.5)
    on = list(beats)
    shuffle = clicks(on + list(beats + 0.5 * 2 / 3), [1.0] * len(beats) + [0.6] * len(beats), 8.5)
    straight = clicks(on + list(beats + 0.25), [1.0] * len(beats) + [0.6] * len(beats), 8.5)
    assert features.detect_feel(shuffle, beats) == "shuffle"
    assert features.detect_feel(straight, beats) == "straight"


def test_window_chroma_rows_unit():
    y = tone([62, 66, 69], 2.0)
    w = features.window_chroma(y, [(0.0, 1.0), (1.0, 2.0), (5.0, 6.0)])
    assert w.shape == (3, 12)
    assert set(np.argsort(w[0])[-3:]) == {2, 6, 9}
    assert np.allclose(w[2], 0.0)
