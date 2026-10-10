from pathlib import Path

from guitarize.config import ARR_NAMES, load_profile, load_song_config
from guitarize.paths import SongPaths


def test_song_paths(tmp_path: Path):
    p = SongPaths.for_slug("demo", tmp_path)
    assert p.root == tmp_path / "demo"
    assert p.slug == "demo"
    assert p.atex("finger") == tmp_path / "demo" / "arrangements" / "finger.atex"
    assert p.stem("vocals") == tmp_path / "demo" / "analysis" / "stems" / "vocals.wav"
    assert p.build("strum", "mid") == tmp_path / "demo" / "build" / "strum.mid"
    assert p.report_root("finger") == tmp_path / "demo" / "reports" / "finger"
    assert p.out("finger", "pdf") == tmp_path / "demo" / "out" / "finger.pdf"


def test_load_profile_intermediate():
    p = load_profile("intermediate")
    assert (p.max_span, p.max_fingers, p.max_fret, p.max_plucks, p.max_shift) == (3, 4, 12, 4, 4)


def test_load_song_config(tmp_path: Path):
    paths = SongPaths(tmp_path / "demo")
    paths.root.mkdir()
    paths.config.write_text(
        "slug: demo\ntitle: T\nartist: A\nsections: {1: Intro, 5: サビ}\n"
        "thresholds: {melody_f1: 0.7}\nsource: {url: null}\n",
        encoding="utf-8",
    )
    cfg = load_song_config(paths)
    assert cfg.slug == "demo" and cfg.sections == {1: "Intro", 5: "サビ"}
    assert cfg.thresholds.melody_f1 == 0.7 and cfg.thresholds.chord_strum == 0.90
    assert cfg.arrangements == ("strum", "finger")
    assert ARR_NAMES["finger"] == "指弹"
    assert cfg.language is None                     # 没写：GAME 不按语言分音


def test_song_config_language(tmp_path: Path):
    # 歌词语言给 GAME 分音用（melody 步骤）：en / ja / yue / zh
    paths = SongPaths(tmp_path / "demo")
    paths.root.mkdir()
    paths.config.write_text("slug: demo\ntitle: T\nartist: A\nlanguage: ja\n", encoding="utf-8")
    assert load_song_config(paths).language == "ja"
    assert paths.game_notes == tmp_path / "demo" / "analysis" / "game_notes.csv"
