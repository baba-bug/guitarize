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
