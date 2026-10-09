import json
import subprocess

import soundfile as sf

from guitarize.acquire import acquire, normalize
from guitarize.paths import SongPaths


def sine(path, seconds=3):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
                    "-ar", "48000", str(path)], check=True)
    return path


def test_normalize_with_trim(tmp_path):
    out = normalize(sine(tmp_path / "a.m4a"), tmp_path / "o.wav", trim=(0.5, 2.0))
    info = sf.info(str(out))
    assert info.samplerate == 44100 and info.channels == 2
    assert abs(info.duration - 1.5) < 0.05


def test_acquire_from_local_raw(tmp_path):
    paths = SongPaths(tmp_path / "song")
    wav = acquire(paths, raw=sine(tmp_path / "x.m4a"))
    assert wav == paths.original_wav and wav.exists()
    meta = json.loads((paths.source_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["kind"] == "local" and len(meta["sha256"]) == 64 and meta["trim"] is None


def test_download_parses_filepath(tmp_path, monkeypatch):
    import subprocess as sp
    from pathlib import Path

    from guitarize import acquire as mod
    seen = {}

    def fake(cmd, **kw):
        seen["cmd"] = cmd
        return sp.CompletedProcess(cmd, 0, stdout="x\n/path/to/BVxyz.m4a\n", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake)
    assert mod.download("https://www.bilibili.com/video/BVxyz", tmp_path) == Path("/path/to/BVxyz.m4a")
    assert seen["cmd"][-2:] == ["--", "https://www.bilibili.com/video/BVxyz"]


def test_download_failure_raises(tmp_path, monkeypatch):
    import subprocess as sp

    import pytest

    from guitarize import acquire as mod
    monkeypatch.setattr(mod.subprocess, "run", lambda cmd, **kw: sp.CompletedProcess(cmd, 1, stdout="", stderr="boom"))
    with pytest.raises(RuntimeError, match="boom"):
        mod.download("https://example.com/v", tmp_path)


def test_kind_detection():
    from guitarize.acquire import _kind
    assert _kind("https://www.bilibili.com/video/BV1") == "bilibili"
    assert _kind("https://b23.tv/abc") == "bilibili"
    assert _kind("https://evil.com/?u=bilibili.com") == "url"
    assert _kind("https://notbilibili.com/x") == "url"
    assert _kind(None) == "local"


def test_trim_validation(tmp_path):
    import pytest
    src = sine(tmp_path / "t.m4a")
    for bad in ((2.0, 1.0), (-1.0, 1.0), (1.0, 1.0)):
        with pytest.raises(ValueError):
            normalize(src, tmp_path / "o.wav", trim=bad)
        with pytest.raises(ValueError):
            acquire(SongPaths(tmp_path / "s"), raw=src, trim=bad)


SONG_YAML = """slug: demo
title: テスト曲
artist: 誰か
version: tv
level: intermediate
arrangements: [strum, finger]
outputs: [html, pdf, txt]
source:
  url: null
  kind: null
  trim: {trim}
reference_charts:
  - https://example.com/chart
overrides:
  key: B
sections:
  1: イントロ
  22: Aメロ
"""


def fake_tools(seen):
    """假的 yt-dlp / ffmpeg：记录命令行，并生成它们本该写出的文件。"""
    import subprocess as sp
    from pathlib import Path

    def run(cmd, **kw):
        if "yt_dlp" in cmd:
            out = Path(cmd[cmd.index("-o") + 1]).parent / "BVxyz.m4a"
            out.write_bytes(b"audio")
            seen["dl"] = cmd
            return sp.CompletedProcess(cmd, 0, stdout=f"{out}\n", stderr="")
        seen["ffmpeg"] = cmd
        Path(cmd[-1]).write_bytes(b"wav")
        return sp.CompletedProcess(cmd, 0)
    return run


def song_dir(tmp_path, trim="null"):
    paths = SongPaths(tmp_path / "songs" / "demo")
    paths.root.mkdir(parents=True)
    paths.config.write_text(SONG_YAML.format(trim=trim), encoding="utf-8")
    return paths


def test_acquire_writes_source_back_to_song_yaml(tmp_path, monkeypatch):
    import yaml

    from guitarize import acquire as mod
    from guitarize.config import load_song_config
    seen = {}
    monkeypatch.setattr(mod.subprocess, "run", fake_tools(seen))
    paths = song_dir(tmp_path)
    before = yaml.safe_load(paths.config.read_text(encoding="utf-8"))
    mod.acquire(paths, url="https://www.bilibili.com/video/BVxyz", trim=(1.0, 2.5))
    text = paths.config.read_text(encoding="utf-8")
    after = yaml.safe_load(text)
    assert after["source"] == {"url": "https://www.bilibili.com/video/BVxyz", "kind": "bilibili", "trim": [1.0, 2.5]}
    assert list(after) == list(before)
    assert {k: v for k, v in after.items() if k != "source"} == {k: v for k, v in before.items() if k != "source"}
    assert list(after["sections"]) == [1, 22] and "テスト曲" in text and "イントロ" in text
    assert load_song_config(paths).sections == {1: "イントロ", 22: "Aメロ"}
    assert seen["ffmpeg"][seen["ffmpeg"].index("-ss") + 1] == "1.0"


def test_acquire_uses_trim_from_song_yaml(tmp_path, monkeypatch):
    import yaml

    from guitarize import acquire as mod
    seen = {}
    monkeypatch.setattr(mod.subprocess, "run", fake_tools(seen))
    paths = song_dir(tmp_path, trim="[0.5, 2.0]")
    raw = tmp_path / "mine.m4a"
    raw.write_bytes(b"audio")
    mod.acquire(paths, raw=raw)
    cmd = seen["ffmpeg"]
    assert cmd[cmd.index("-ss") + 1:cmd.index("-to") + 2] == ["0.5", "-to", "2.0"]
    source = yaml.safe_load(paths.config.read_text(encoding="utf-8"))["source"]
    assert source == {"url": None, "kind": "local", "trim": [0.5, 2.0]}
    meta = json.loads((paths.source_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["trim"] == [0.5, 2.0]


def test_explicit_trim_wins_over_song_yaml(tmp_path, monkeypatch):
    import yaml

    from guitarize import acquire as mod
    seen = {}
    monkeypatch.setattr(mod.subprocess, "run", fake_tools(seen))
    paths = song_dir(tmp_path, trim="[0.5, 2.0]")
    raw = tmp_path / "mine.m4a"
    raw.write_bytes(b"audio")
    mod.acquire(paths, raw=raw, trim=(3.0, 4.0))
    cmd = seen["ffmpeg"]
    assert cmd[cmd.index("-ss") + 1:cmd.index("-to") + 2] == ["3.0", "-to", "4.0"]
    assert yaml.safe_load(paths.config.read_text(encoding="utf-8"))["source"]["trim"] == [3.0, 4.0]


def test_bad_trim_in_song_yaml_rejected_before_download(tmp_path, monkeypatch):
    import pytest

    from guitarize import acquire as mod
    seen = {}
    monkeypatch.setattr(mod.subprocess, "run", fake_tools(seen))
    for bad in ("[2.0, 1.0]", "[1.0]", "oops"):
        paths = song_dir(tmp_path / bad.replace("[", "").replace("]", "").replace(", ", "_"), trim=bad)
        with pytest.raises(ValueError):
            mod.acquire(paths, url="https://www.bilibili.com/video/BVxyz")
    assert seen == {}
