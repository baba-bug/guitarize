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
    assert set(out) == {"gp", "txt", "pdf", "html", "index"}   # index：这首歌的总页面（全部编配切换）
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


def test_melody_and_kazoo_pages_default_to_the_lead_vocal(demo_song):
    # 卡祖笛就是跟着主唱哼：旋律 / 卡祖笛页面的参考音轨默认选主唱，其余编配默认原曲
    from guitarize.render.viewer import build_viewer
    for name in ("vocals", "lead"):
        p = demo_song.stem(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"RIFF")
    page = build_viewer(demo_song, "finger", "T").read_text(encoding="utf-8")
    assert "window.REF_DEFAULT = 0;" in page
    for arr in ("melody", "kazoo"):
        demo_song.atex(arr).write_text(demo_song.atex("finger").read_text(encoding="utf-8"), encoding="utf-8")
        page = build_viewer(demo_song, arr, "T").read_text(encoding="utf-8")
        refs = json.loads(page.split("window.REF_AUDIO = ")[1].split(";\n")[0])
        default = int(page.split("window.REF_DEFAULT = ")[1].split(";")[0])
        assert refs[default]["label"] == "主唱（模型去掉伴唱）"


def test_hub_page_switches_between_every_rendered_arrangement(demo_song):
    # 用户要求：所有谱子放在一个网页里切换
    from guitarize.config import load_song_config
    from guitarize.render.hub import build_hub
    cfg = load_song_config(demo_song)
    for arr in ("finger", "melody"):
        demo_song.out(arr, "html").parent.mkdir(parents=True, exist_ok=True)
        demo_song.out(arr, "html").write_text("<html></html>", encoding="utf-8")
    demo_song.out("melody", "pdf").write_bytes(b"%PDF")
    hub = build_hub(demo_song, cfg)
    page = hub.read_text(encoding="utf-8")
    pages = json.loads(page.split("const PAGES = ")[1].split(";\n")[0])
    keys = [p["key"] for p in pages]
    assert "finger" in keys and "melody" in keys and hub.name == "index.html"
    mel = next(p for p in pages if p["key"] == "melody")
    assert mel["html"] == "melody.html" and {"label": "PDF", "url": "melody.pdf"} in mel["files"]


def test_variant_names_and_labels():
    import pytest

    from guitarize.config import arr_label, split_arr
    assert split_arr("strum_old") == ("strum", "old") and split_arr("power") == ("power", None)
    assert arr_label("strum_old") == "弹唱（旧版）" and arr_label("finger_v2") == "指弹（v2）" and arr_label("strum") == "弹唱"
    for bad in ("nope_old", "strum_", "strum_Old", "_old", "strum_old_2"):
        with pytest.raises(ValueError, match="arrangement"):
            split_arr(bad)


def test_variant_renders_like_its_arrangement_and_follows_it_in_the_hub(demo_song):
    # 用户要求：重新生成前的旧版弹唱留着对比——arrangements/strum_old.atex 按弹唱的设置渲染成 out/strum_old.*，
    # 总页面里排在弹唱后面，叫「弹唱（旧版）」
    import shutil

    import pytest

    from guitarize.cli import main
    songs = str(demo_song.root.parent)
    shutil.copyfile(demo_song.atex("strum"), demo_song.atex("strum_old"))
    assert main(["--songs-dir", songs, "render", "demo", "--arr", "strum_old"]) == 0
    assert all(demo_song.out("strum_old", ext).exists() for ext in ("gp", "txt", "pdf", "html"))
    assert "<title>Demo Twinkle · 弹唱（旧版）</title>" in demo_song.out("strum_old", "html").read_text(encoding="utf-8")
    for arr in ("strum", "finger", "power_x"):
        demo_song.out(arr, "html").write_text("<html></html>", encoding="utf-8")
    demo_song.out("strum_Bad", "html").write_text("<html></html>", encoding="utf-8")   # 不是变体名：不列
    from guitarize.render.hub import build_hub
    page = build_hub(demo_song, load_song_config(demo_song)).read_text(encoding="utf-8")
    pages = json.loads(page.split("const PAGES = ")[1].split(";\n")[0])
    assert [(p["key"], p["label"]) for p in pages] == [("strum", "弹唱"), ("strum_old", "弹唱（旧版）"), ("finger", "指弹"),
                                                      ("power_x", "强力和弦（x）")]
    old = pages[1]
    assert old["html"] == "strum_old.html" and {"label": "PDF", "url": "strum_old.pdf"} in old["files"]
    for bad in ("nope_old", "strum_"):                                              # render 之外不认变体
        with pytest.raises(SystemExit):
            main(["--songs-dir", songs, "render", "demo", "--arr", bad])
    with pytest.raises(SystemExit):
        main(["--songs-dir", songs, "verify", "demo", "--arr", "strum_old"])
