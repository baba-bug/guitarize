import json

import numpy as np
import soundfile as sf

from guitarize.config import load_song_config
from guitarize.render import alphatab
from guitarize.render.pipeline import render_all
from guitarize.render.synth import midi_to_wav


def test_midi_to_wav(demo_song, tmp_path):
    built = alphatab.build(demo_song.atex("finger"), tmp_path / "b", "f")
    wav = midi_to_wav(built["mid"], tmp_path / "f.wav")
    y, sr = sf.read(wav)
    assert sr == 44100
    assert 19.0 < len(y) / sr < 24.0  # 8 小节 × 2.4 秒 + 余音
    assert np.sqrt(np.mean(np.square(y))) > 1e-3


def test_render_all(demo_song):
    cfg = load_song_config(demo_song)
    out = render_all(demo_song, cfg, "finger")
    assert set(out) == {"gp", "txt", "pdf", "html"}
    assert out["pdf"].read_bytes()[:4] == b"%PDF" and out["pdf"].stat().st_size > 10_000
    assert "e|" in out["txt"].read_text(encoding="utf-8")
    page = out["html"].read_text(encoding="utf-8")
    assert "{{" not in page
    assert "alphaTab.js" in page and "\\voice" in page
    bar_times = json.loads(page.split("window.BAR_TIMES = ")[1].split(";")[0])
    assert len(bar_times) == 8 and bar_times[1] == [2.4, 4.8]
