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


def test_viewer_offers_original_and_vocal_playback_from_the_current_bar(demo_song):
    # 用户要求：谱面网页上能直接从当前小节听提取出来的人声，和合成音对照
    from guitarize.render.viewer import build_viewer
    vocals = demo_song.stem("vocals")
    page = build_viewer(demo_song, "finger", "T").read_text(encoding="utf-8")
    refs = json.loads(page.split("window.REF_AUDIO = ")[1].split(";\n")[0])
    if not vocals.exists():
        assert [r["label"] for r in refs] == ["原曲"]
        vocals.parent.mkdir(parents=True, exist_ok=True)
        vocals.write_bytes(demo_song.original_wav.read_bytes() if demo_song.original_wav.exists() else b"RIFF")
        page = build_viewer(demo_song, "finger", "T").read_text(encoding="utf-8")
        refs = json.loads(page.split("window.REF_AUDIO = ")[1].split(";\n")[0])
    assert [r["label"] for r in refs] == ["原曲", "人声"]
    out_dir = demo_song.out("finger", "html").parent
    targets = [(out_dir / r["url"]).resolve() for r in refs]
    assert targets == [demo_song.original_wav.resolve(), vocals.resolve()]     # 相对链接指向原曲与人声分轨
    assert "{{" not in page


def test_viewer_lists_every_available_stem(demo_song):
    # 参考音频做成列表：原曲 + 有的分轨（人声、吉他、贝斯、鼓、其他、钢琴、和声混合）都能选来从当前小节播放
    from guitarize.render.viewer import build_viewer
    for name in ("vocals", "guitar", "drums"):
        p = demo_song.stem(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"RIFF")
    page = build_viewer(demo_song, "finger", "T").read_text(encoding="utf-8")
    refs = json.loads(page.split("window.REF_AUDIO = ")[1].split(";\n")[0])
    assert [r["label"] for r in refs] == ["原曲", "人声", "吉他", "鼓"]
    assert 'id="ref"' in page and 'id="refplay"' in page


def test_viewer_lists_the_lead_and_backing_vocals_from_the_karaoke_split(demo_song):
    # 伴唱分离模型分出的主唱 / 伴唱都能选来对照；只用 L/R 掩码时没有 backing.wav，就不列伴唱
    from guitarize.render.viewer import build_viewer
    for name in ("vocals", "lead", "backing"):
        p = demo_song.stem(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"RIFF")

    def labels():
        page = build_viewer(demo_song, "finger", "T").read_text(encoding="utf-8")
        return [r["label"] for r in json.loads(page.split("window.REF_AUDIO = ")[1].split(";\n")[0])]

    assert labels() == ["原曲", "人声", "主唱（模型去掉伴唱）", "伴唱"]
    demo_song.stem("backing").unlink()
    assert labels() == ["原曲", "人声", "主唱（模型去掉伴唱）"]
