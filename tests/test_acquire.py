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
