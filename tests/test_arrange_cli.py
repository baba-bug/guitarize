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
