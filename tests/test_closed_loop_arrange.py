"""闭环（续）：合成歌曲 → 分析 → 编配（弹唱 / 指弹 / 旋律谱）→ 校验。spec §8：已知答案的歌走完整条真实路径。

歌按真实布局放进临时 songs 目录（source/original.wav、analysis/stems/*.wav、analysis/webchart.txt、song.yaml），
分析只跑 beats / chords / melody / leadsheet——合成分轨是现成的，不经 Demucs。
"""
import csv
import json
import shutil

import pytest
import yaml

import synth_song as S
from guitarize.analyze.pipeline import run_analysis
from guitarize.analyze.stems import STEMS, mix_harmonic
from guitarize.arrange.pipeline import arrange
from guitarize.config import SongConfig, load_song_config
from guitarize.paths import SongPaths
from guitarize.verify.report import run_verify

pytestmark = pytest.mark.slow

SLUG = "synth-closed-loop"
BARS = len(S.PROGRESSION)
SECTIONS = {1: "Intro", 9: "Chorus"}


def _scores(paths: SongPaths, arr: str) -> dict:
    return json.loads((paths.report_root(arr) / "iter-01" / "scores.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def song(tmp_path_factory):
    work = tmp_path_factory.mktemp("closed_loop")
    rendered = S.render_song(work / "render")
    paths = SongPaths.for_slug(SLUG, work / "songs")
    paths.source_dir.mkdir(parents=True)
    paths.stems_dir.mkdir(parents=True)
    shutil.copy(rendered["mix"], paths.original_wav)
    for name in STEMS:
        shutil.copy(rendered["stems"] / f"{name}.wav", paths.stem(name))
    mix_harmonic(paths.stems_dir)
    paths.webchart_txt.write_text(S.truth_webchart(), encoding="utf-8")
    paths.config.write_text(yaml.safe_dump({
        "slug": SLUG, "title": "Synthetic Closed Loop", "artist": "guitarize tests", "version": "tv",
        "level": "intermediate", "arrangements": ["strum", "finger", "melody"], "outputs": ["html", "pdf", "txt"],
        "reference_charts": [], "overrides": {}, "sections": SECTIONS,
    }, sort_keys=False), encoding="utf-8")
    cfg = load_song_config(paths)
    assert isinstance(cfg, SongConfig) and cfg.level == "intermediate" and cfg.sections == SECTIONS

    run_analysis(paths, cfg, steps=("beats", "chords", "melody", "leadsheet"))
    out = {"paths": paths, "cfg": cfg}
    for arr in ("strum", "finger", "melody"):
        _, _, violations = arrange(paths, cfg, arr)  # capo 自动选
        out[arr] = {"violations": violations}
        run_verify(paths, cfg, arr)
        out[arr]["scores"] = _scores(paths, arr)
    return out


def test_leadsheet_matches_the_known_song(song):
    """arrange / verify 都以 leadsheet 为参考：先确认 leadsheet 本身对得上真值，下面的一致率才有意义。"""
    ls = json.loads(song["paths"].leadsheet_json.read_text(encoding="utf-8"))
    bars = ls["bars"]
    # 歌结束后的尾音可能让节拍网格多出一个不满的小节；前 16 小节必须与真值一一对上
    assert BARS <= len(bars) <= BARS + 1 and ls["beats_per_bar"] == 4
    assert (ls["key"], ls["mode"]) == ("G", "major")
    assert {b["bar"]: b["section"] for b in bars if b["section"]} == SECTIONS
    truth = S.melody_truth()
    for k, bar in enumerate(bars[:BARS]):
        assert [c["label"] for c in bar["chords"]] == [S.PROGRESSION[k]] and bar["chords"][0]["dur_beats"] == 4, bar["bar"]
        want = [p for s, _, p in truth if s // 4 == k]
        got = [m["midi"] for m in bar["melody"]]
        assert got == want if k < BARS - 1 else got[:1] == want, (bar["bar"], got, want)  # 末小节全音符尾部可能被切出一个短音


@pytest.mark.parametrize("arr", ["strum", "finger", "melody"])
def test_draft_has_no_playability_violations(song, arr):
    assert song[arr]["violations"] == []  # 草稿 0 违规是硬约束


@pytest.mark.parametrize("arr", ["strum", "finger"])
def test_verify_reports_match_the_reference(song, arr):
    s = song[arr]["scores"]["summary"]
    n_bars = len(json.loads(song["paths"].leadsheet_json.read_text(encoding="utf-8"))["bars"])
    assert s["violations"] == 0 and song[arr]["scores"]["violations"] == []
    assert s["arrangement_bars"] == s["reference_bars"] == n_bars
    assert s["checks"]["bar_count"] and s["checks"]["playability"] and s["checks"]["reference"]
    # 真歌的 64 拍全部计分；唯一不计分的是尾音多出来的那一小节里的低置信拍（网上谱只有 16 小节，对不上）
    assert s["scored_beats"] == 4 * BARS and s["low_conf_beats"] <= 1, s
    assert s["harmonic_similarity"] is not None and s["harmonic_similarity"] >= 0.8, s
    assert song[arr]["scores"]["problems"] == [] and s["pass"] is True, s
    assert s["chord_agreement"] >= (0.90 if arr == "strum" else 0.85), s


def test_finger_melody_f1(song):
    s = song["finger"]["scores"]["summary"]
    assert s["melody_ref_notes"] > 0 and s["melody_f1"] >= 0.80, s


def test_melody_arrangement_plays_the_known_melody(song):
    """旋律谱：每个旋律音都在谱上（召回按全部参考音），单声部 0 违规，和人声分轨的色度对得上；不评和弦。"""
    s = song["melody"]["scores"]["summary"]
    n_bars = len(json.loads(song["paths"].leadsheet_json.read_text(encoding="utf-8"))["bars"])
    assert s["violations"] == 0 and s["arrangement_bars"] == s["reference_bars"] == n_bars
    assert s["melody_f1"] >= 0.9 and s["melody_recall"] >= 0.9 and s["melody_precision"] >= 0.9, s
    assert s["melody_ref_notes"] >= len(S.melody_truth()) - 1, s
    assert "chord" not in s["checks"] and s["chord_agreement"] is None
    assert s["vocal_similarity"] is not None and s["vocal_similarity"] >= 0.7, s
    assert s["checks"]["melody"] and s["checks"]["playability"] and s["checks"]["reference"], s
    assert song["paths"].vocal_f0.exists()                  # melody 步骤写的人声基频缓存（音高图用）
    assert "\\voice" not in song["paths"].atex("melody").read_text(encoding="utf-8")


def test_history_records_each_verify(song):
    for arr in ("strum", "finger"):
        rows = list(csv.DictReader((song["paths"].report_root(arr) / "history.csv").open(encoding="utf-8")))
        assert [r["iter"] for r in rows] == ["iter-01"]
        assert float(rows[0]["chord_agreement"]) == pytest.approx(song[arr]["scores"]["summary"]["chord_agreement"])
        assert rows[0]["violations"] == "0" and rows[0]["harmonic_similarity"] != ""
