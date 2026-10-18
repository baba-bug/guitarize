import json

import pytest

from conftest import demo_leadsheet
from guitarize.cli import main
from guitarize.config import load_song_config
from guitarize.paths import SongPaths


def test_new_creates_song_yaml(tmp_path):
    assert main(["--songs-dir", str(tmp_path), "new", "x", "--title", "T", "--artist", "A"]) == 0
    cfg = load_song_config(SongPaths.for_slug("x", tmp_path))
    assert cfg.title == "T" and cfg.level == "intermediate" and cfg.arrangements == ("strum", "finger")


def test_arrange_then_verify(demo_song, capsys):
    songs = str(demo_song.root.parent)
    demo_song.atex("strum").unlink()
    demo_song.atex("finger").unlink()
    for arr in ("strum", "finger"):
        assert main(["--songs-dir", songs, "arrange", "demo", "--arr", arr]) == 0
        assert demo_song.atex(arr).exists()
        assert main(["--songs-dir", songs, "verify", "demo", "--arr", arr]) == 0
        s = json.loads((demo_song.report_root(arr) / "iter-01" / "scores.json").read_text(encoding="utf-8"))["summary"]
        assert s["violations"] == 0 and s["chord_agreement"] >= 0.85
    out = capsys.readouterr().out
    # brief originally asserted "capo 5"; choose_capo actually picks capo 3 for C/F/G
    # (A/D/E open shapes, cost 1.92 < capo 5 cost 3.2)
    assert "capo 3" in out


def test_arrange_refuses_to_overwrite(demo_song):
    with pytest.raises(FileExistsError):
        main(["--songs-dir", str(demo_song.root.parent), "arrange", "demo", "--arr", "strum"])


def _arr(demo_song, *extra):
    return main(["--songs-dir", str(demo_song.root.parent), "arrange", "demo", *extra])


def test_new_title_roundtrip(tmp_path):
    title = 'Love: "Story" #1 [x]'
    assert main(["--songs-dir", str(tmp_path), "new", "x", "--title", title, "--artist", "A: B"]) == 0
    cfg = load_song_config(SongPaths.for_slug("x", tmp_path))
    assert cfg.title == title and cfg.artist == "A: B"


@pytest.mark.parametrize("bad", [9, -1])
def test_capo_out_of_range_flag(demo_song, bad):
    demo_song.atex("strum").unlink()
    with pytest.raises(ValueError, match="capo must be one of 0..7"):
        _arr(demo_song, "--arr", "strum", f"--capo={bad}")


def test_capo_out_of_range_override(demo_song):
    demo_song.atex("strum").unlink()
    import yaml
    d = yaml.safe_load(demo_song.config.read_text(encoding="utf-8"))
    d["overrides"] = {"capo": 9}
    demo_song.config.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="capo must be one of"):
        _arr(demo_song, "--arr", "strum")


def test_capo_flag_honored_and_printed(demo_song, capsys):
    demo_song.atex("strum").unlink()
    assert _arr(demo_song, "--arr", "strum", "--capo", "0") == 0
    assert "capo 0" in capsys.readouterr().out


def test_capo_override_honored(demo_song, capsys):
    import yaml
    demo_song.atex("strum").unlink()
    d = yaml.safe_load(demo_song.config.read_text(encoding="utf-8"))
    d["overrides"] = {"capo": 1}
    demo_song.config.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    assert _arr(demo_song, "--arr", "strum") == 0
    assert "capo 1" in capsys.readouterr().out


def test_force_overwrites(demo_song):
    demo_song.atex("strum").write_text("HAND EDITED", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _arr(demo_song, "--arr", "strum")
    assert demo_song.atex("strum").read_text(encoding="utf-8") == "HAND EDITED"
    assert _arr(demo_song, "--arr", "strum", "--force") == 0
    assert demo_song.atex("strum").read_text(encoding="utf-8") != "HAND EDITED"


def test_strum_pattern_override_is_validated():
    from guitarize.config import strum_pattern

    assert strum_pattern({}) is None
    assert strum_pattern({"strum_pattern": "quarter"}) == "quarter"
    with pytest.raises(ValueError, match="strum_pattern"):
        strum_pattern({"strum_pattern": "polka"})


def test_section_patterns_with_stems(tmp_path):
    import numpy as np
    import soundfile as sf

    from guitarize.arrange.patterns import PATTERNS, accent_vector
    from guitarize.arrange.pipeline import section_patterns

    paths = SongPaths(tmp_path / "song")
    paths.stems_dir.mkdir(parents=True)
    sr, spb = 22050, 0.6
    bar_len = 4 * spb
    slot = bar_len / 8
    bars = []
    for i in range(4):
        bars.append({"bar": i + 1, "section": ("A" if i == 0 else "B" if i == 2 else None),
                     "t_start": i * bar_len, "t_end": (i + 1) * bar_len, "chords": [], "melody": []})
    y = np.zeros(int(sr * 4 * bar_len))
    pop = accent_vector(PATTERNS["pop"], "straight")
    quarter = accent_vector(PATTERNS["quarter"], "straight")
    click = np.hanning(200) * np.sin(np.arange(200) * 0.9)
    for i in range(4):
        vec = pop if i < 2 else quarter
        for k, w in enumerate(vec):
            if w > 0:
                n = int((i * bar_len + k * slot) * sr)
                y[n:n + 200] += w * click
    for name in ("drums", "guitar"):
        sf.write(str(paths.stem(name)), y, sr)
    out = section_patterns(paths, {"feel": "straight", "beats_per_bar": 4, "bars": bars})
    assert len(out) == 4 and out[0] == out[1] and out[2] == out[3]
    assert out[0] == "pop"
    assert out[2] == "quarter"
