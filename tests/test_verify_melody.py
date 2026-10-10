"""旋律谱的校验：召回按全部参考音、F1 达标线 0.9、不评和弦；人声分轨相似度、音高图、人声试听片段、需人耳确认。"""
import csv
import json

import numpy as np
import pytest
import soundfile as sf

from conftest import TEX_HEAD, demo_leadsheet
from guitarize.analyze import melody as amel
from guitarize.arrange.melody import generate_melody
from guitarize.cli import main
from guitarize.config import Thresholds, load_profile
from guitarize.render import alphatab
from guitarize.render.synth import midi_to_wav
from guitarize.verify import metrics
from guitarize.verify.model import load_score

MID = load_profile("intermediate")


def _one_bar(notes, feel="straight"):
    return {"key": "C", "mode": "major", "tempo_bpm": 100.0, "feel": feel, "beats_per_bar": 4,
            "bars": [{"bar": 1, "section": None, "t_start": 0.0, "t_end": 2.4,
                      "chords": [{"beat": 1, "dur_beats": 4, "label": "C", "confidence": "high"}],
                      "melody": [{"start_beats": s, "dur_beats": d, "midi": m, "conf": 0.9, "out_of_key": False}
                                 for s, d, m in notes]}]}


# C4（拍点）· D4（2&，半拍、紧跟前一个音：不是骨架音）· E4（第 3 拍）；谱上漏了 D4
REF = [(0, 1, 60), (1.5, 0.5, 62), (2, 2, 64)]
TAB_WITHOUT_D4 = "1.2.4 r.4 0.1.2 |"


def test_melody_recall_counts_every_reference_note(build_tex):
    score = build_tex(TAB_WITHOUT_D4, head=TEX_HEAD)
    skeleton = metrics.score_bars(score, _one_bar(REF), "finger", finger_style="skeleton")
    assert metrics.summarize(skeleton, [], Thresholds(), "finger")["melody_recall"] == 1.0   # 骨架指弹：D4 不算漏
    finger = metrics.summarize(metrics.score_bars(score, _one_bar(REF), "finger"), [], Thresholds(), "finger")
    assert finger["melody_recall"] == pytest.approx(2 / 3)     # 完整旋律的指弹（默认）和旋律谱一样考每一个音
    bars = metrics.score_bars(score, _one_bar(REF), "melody")
    s = metrics.summarize(bars, [], Thresholds(), "melody")
    assert s["melody_precision"] == 1.0 and s["melody_recall"] == pytest.approx(2 / 3)
    assert s["melody_f1"] == pytest.approx(0.8)
    assert bars[0].mel_issues == ["漏掉旋律音：拍 2& D4（指法调 D4）"]


def test_melody_summary_has_no_chord_check_and_a_stricter_line(build_tex):
    score = build_tex(TAB_WITHOUT_D4, head=TEX_HEAD)
    bars = metrics.score_bars(score, _one_bar(REF), "melody")
    s = metrics.summarize(bars, [], Thresholds(), "melody", arrangement_bars=1, reference_bars=1)
    assert "chord" not in s["checks"] and "chord_marks" not in s["checks"]
    assert s["chord_agreement"] is None and s["scored_beats"] == 0 and bars[0].chord_total == 0
    assert bars[0].mark_issues == []                            # 单旋律线不核对谱上的和弦名
    assert s["melody_target"] == 0.9 and s["checks"]["melody"] is False and s["pass"] is False   # F1 0.8 < 0.9
    assert s["checks"]["reference"] and s["checks"]["playability"] and s["checks"]["bar_count"]
    assert metrics.problem_bars(bars, [], Thresholds(), "melody") == [1]   # 少一个音的小节就标出来
    full = build_tex("1.2.4 r.8 3.2.8 0.1.2 |", head=TEX_HEAD)
    ok = metrics.summarize(metrics.score_bars(full, _one_bar(REF), "melody"), [], Thresholds(), "melody")
    assert ok["melody_f1"] == 1.0 and ok["pass"] is True


def test_tab_notes_where_the_reference_melody_is_empty_count_against_precision(build_tex):
    # 旋律谱：参考这一小节没有旋律（人声没唱），谱上却有音——每个音都算进精确率（指弹照旧跳过这样的小节），
    # 这一小节标成问题小节
    ls = _one_bar(REF)
    ls["bars"].append({**ls["bars"][0], "bar": 2, "t_start": 2.4, "t_end": 4.8, "melody": []})
    score = build_tex("1.2.4 r.8 3.2.8 0.1.2 |\n1.2.4 3.2.4 0.1.2 |", head=TEX_HEAD)
    bars = metrics.score_bars(score, ls, "melody")
    assert (bars[0].mel_hit, bars[0].mel_est, bars[0].mel_found, bars[0].mel_ref) == (3, 3, 3, 3)
    assert (bars[1].mel_hit, bars[1].mel_est, bars[1].mel_ref) == (0, 3, 0) and bars[1].mel_f1 is None
    assert bars[1].mel_issues == ["多余/错音：拍 1 C4", "多余/错音：拍 2 D4", "多余/错音：拍 3 E4"]
    s = metrics.summarize(bars, [], Thresholds(), "melody")
    assert s["melody_precision"] == pytest.approx(0.5) and s["melody_recall"] == 1.0
    assert s["checks"]["melody"] is False and s["pass"] is False
    assert metrics.problem_bars(bars, [], Thresholds(), "melody") == [2]
    from guitarize.verify.report import render_report
    report = render_report("melody", "iter-01", s, None, [], bars, [2], {}, {}, score)
    assert "- 参考这一小节没有旋律，谱上却有 3 个音\n  - 多余/错音：拍 1 C4\n" in report
    finger = metrics.score_bars(score, ls, "finger", finger_style="skeleton")
    assert finger[1].mel_est == 0 and metrics.summarize(finger, [], Thresholds(), "finger")["melody_precision"] == 1.0
    full = metrics.score_bars(score, ls, "finger")
    assert full[1].mel_est == 3 and metrics.summarize(full, [], Thresholds(), "finger")["melody_precision"] == 0.5


def test_melody_onset_tolerance_is_half_a_triplet(build_tex):
    # 旋律谱照原唱落到三连音 / 十六分格（演奏时间）：差一个三连音格（1/3 拍）的音不算命中；指弹的容差不变
    ref = [(1 / 3, 2 / 3, 60), (2, 2, 64)]                     # C4 唱在 1+⅓
    score = build_tex("1.2.2 0.1.2 |", head=TEX_HEAD)           # 谱上 C4 在拍点：早了 1/3 拍
    finger = metrics.score_bars(score, _one_bar(ref), "finger", finger_style="skeleton")
    assert (finger[0].mel_hit, finger[0].mel_est) == (2, 2)
    assert metrics.score_bars(score, _one_bar(ref), "finger")[0].mel_hit == 1   # 完整旋律的指弹与旋律谱同一容差
    bars = metrics.score_bars(score, _one_bar(ref), "melody")
    assert (bars[0].mel_hit, bars[0].mel_est, bars[0].mel_found, bars[0].mel_ref) == (1, 2, 1, 2)
    assert bars[0].mel_issues == ["多余/错音：拍 1 C4", "漏掉旋律音：拍 1+⅓ C4（指法调 C4）"]
    near = metrics.score_bars(score, _one_bar([(1 / 6, 5 / 6, 60), (2, 2, 64)]), "melody")   # 半格以内照样命中
    assert near[0].mel_hit == 2


def test_ear_check_lists_out_of_key_and_low_confidence_notes(build_tex):
    ls = _one_bar(REF)
    ls["bars"][0]["melody"][1].update(out_of_key=True)
    ls["bars"][0]["melody"][2].update(conf=0.42)
    score = build_tex("1.2.4 r.8 3.2.8 0.1.2 |", head=TEX_HEAD)
    bars = metrics.score_bars(score, ls, "melody")
    assert bars[0].ear_notes == ["拍 2& D4（调外）", "拍 3 E4（置信 0.42）"]
    s = metrics.summarize(bars, [], Thresholds(), "melody")
    assert s["ear_check_bars"] == 1 and s["ear_check_notes"] == 2
    ls["bars"][0]["melody"][1]["manual"] = True                 # overrides.melody 写的音已经人耳确认过
    assert metrics.score_bars(score, ls, "melody")[0].ear_notes == ["拍 3 E4（置信 0.42）"]


def _sine_vocal(path, leadsheet, wrong_bar=None, sr=22050, spb=0.6):
    """leadsheet 旋律的正弦波"人声"；wrong_bar 那一小节唱成高三全音的错音。"""
    n = int(len(leadsheet["bars"]) * 4 * spb * sr) + sr
    y = np.zeros(n)
    for b in leadsheet["bars"]:
        for m in b["melody"]:
            midi = m["midi"] + (6 if b["bar"] == wrong_bar else 0)
            a, z = int(m["start_beats"] * spb * sr), int((m["start_beats"] + m["dur_beats"] * 0.9) * spb * sr)
            t = np.arange(z - a) / sr
            y[a:z] += 0.3 * np.sin(2 * np.pi * 440 * 2 ** ((midi - 69) / 12) * t)
    sf.write(str(path), y, sr)
    return path


def test_vocal_similarity_against_a_sine_vocal(tmp_path):
    ls = demo_leadsheet()
    src = tmp_path / "m.atex"
    src.write_text(generate_melody(ls, 0, MID, "T"), encoding="utf-8")
    built = alphatab.build(src, tmp_path / "b", "m")
    score = load_score(built["json"])
    synth = midi_to_wav(built["mid"], tmp_path / "m.wav")
    vocal = _sine_vocal(tmp_path / "vocals.wav", ls, wrong_bar=2)
    bars = metrics.score_bars(score, ls, "melody", synth_wav=synth, harmonic_wav=tmp_path / "missing.wav",
                              vocal_wav=vocal)
    sims = [b.vocal_sim for b in bars]
    assert all(b.harmonic_sim is None for b in bars)            # 旋律谱不和和声分轨比
    right = [x for i, x in enumerate(sims) if i != 1]
    assert min(right) > 0.75 and sims[1] < min(right) - 0.2, sims
    s = metrics.summarize(bars, [], Thresholds(), "melody")
    assert s["vocal_similarity"] == pytest.approx(np.mean(sims)) and s["harmonic_similarity"] is None
    assert 2 in metrics.problem_bars(bars, [], Thresholds(bar_vocal_flag=sims[1] + 0.05), "melody")


def test_vocal_f0_cache_is_reused_until_the_stem_changes(tmp_path, monkeypatch):
    wav = _sine_vocal(tmp_path / "vocals.wav", demo_leadsheet())
    cache = tmp_path / "analysis" / "vocal_f0.npz"
    calls = []

    def fake(path):
        calls.append(path)
        return np.arange(3) * 0.01, np.array([60.0, np.nan, 62.0]), np.ones(3)

    monkeypatch.setattr(amel, "vocal_pitch", fake)
    t, m, p = amel.cached_vocal_pitch(wav, cache)
    assert len(calls) == 1 and cache.exists() and np.isnan(m[1]) and m[2] == 62.0
    amel.cached_vocal_pitch(wav, cache)
    assert len(calls) == 1                                      # 分轨没变：直接读缓存
    _sine_vocal(wav, demo_leadsheet(), wrong_bar=1)             # 分轨重新生成
    amel.cached_vocal_pitch(wav, cache)
    assert len(calls) == 2


def _fake_pitch(calls):
    def fake(path):
        calls.append(path)
        return np.arange(3) * 0.01, np.array([60.0, np.nan, 62.0]), np.ones(3)
    return fake


def test_vocal_f0_cache_that_cannot_be_read_is_recomputed(tmp_path, monkeypatch):
    # 截断 / 损坏 / 缺字段的缓存当作没有缓存：重算并重写，不让 melody 步骤或校验报错
    wav = _sine_vocal(tmp_path / "vocals.wav", demo_leadsheet())
    cache = tmp_path / "analysis" / "vocal_f0.npz"
    calls = []
    monkeypatch.setattr(amel, "vocal_pitch", _fake_pitch(calls))
    amel.cached_vocal_pitch(wav, cache)
    good = cache.read_bytes()
    with open(cache, "wb") as f:                                # 签名对得上，但缺 times / midi
        np.savez(f, **amel._stem_signature(wav))
    junk = [cache.read_bytes(), b"", b"not an npz at all", good[: len(good) // 2]]
    for i, data in enumerate(junk):
        cache.write_bytes(data)
        t, m, _ = amel.cached_vocal_pitch(wav, cache)
        assert len(calls) == i + 2 and m[2] == 62.0
        assert amel._read_vocal_f0(cache, amel._stem_signature(wav)) is not None   # 重写成完整的缓存
    amel.cached_vocal_pitch(wav, cache)
    assert len(calls) == len(junk) + 1


def test_vocal_f0_cache_is_written_atomically(tmp_path, monkeypatch):
    # 写到一半出错（磁盘满、被打断）不留下半个缓存：旧缓存原样，目录里也没有临时文件
    wav = _sine_vocal(tmp_path / "vocals.wav", demo_leadsheet())
    cache = tmp_path / "analysis" / "vocal_f0.npz"
    amel.write_vocal_f0(cache, wav, np.arange(3) * 0.01, np.array([60.0, 61.0, 62.0]), np.ones(3))
    before = cache.read_bytes()

    def boom(f, **arrays):
        f.write(b"PK partial")
        raise OSError("disk full")

    monkeypatch.setattr(amel.np, "savez", boom)
    with pytest.raises(OSError, match="disk full"):
        amel.write_vocal_f0(cache, wav, np.arange(3) * 0.01, np.array([50.0, 51.0, 52.0]), np.ones(3))
    assert cache.read_bytes() == before
    assert [p.name for p in cache.parent.iterdir()] == ["vocal_f0.npz"]


def test_vocal_f0_cache_records_which_stem_it_came_from(tmp_path, monkeypatch):
    # lead.wav 与 vocals.wav 共用一个缓存文件：大小、修改时间都一样也不能拿错（签名里有分轨文件名）
    import os
    import shutil

    lead = _sine_vocal(tmp_path / "lead.wav", demo_leadsheet())
    vocals = tmp_path / "vocals.wav"
    shutil.copyfile(lead, vocals)
    st = lead.stat()
    os.utime(vocals, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert vocals.stat().st_size == st.st_size and vocals.stat().st_mtime_ns == st.st_mtime_ns
    cache = tmp_path / "analysis" / "vocal_f0.npz"
    calls = []
    monkeypatch.setattr(amel, "vocal_pitch", _fake_pitch(calls))
    amel.cached_vocal_pitch(lead, cache)
    amel.cached_vocal_pitch(vocals, cache)
    assert calls == [lead, vocals]
    amel.cached_vocal_pitch(vocals, cache)
    assert calls == [lead, vocals]


def test_verify_melody_writes_vocal_snippets_pitch_plots_and_report(demo_song, monkeypatch, capsys):
    songs = str(demo_song.root.parent)
    assert main(["--songs-dir", songs, "arrange", "demo", "--arr", "melody"]) == 0
    ls = json.loads(demo_song.leadsheet_json.read_text(encoding="utf-8"))
    demo_song.stems_dir.mkdir(parents=True, exist_ok=True)
    vocal = _sine_vocal(demo_song.stem("vocals"), ls)
    times = np.arange(0, 20, 0.01)
    contour = np.full(len(times), np.nan)
    for b in ls["bars"]:
        for m in b["melody"]:
            contour[(times >= m["start_beats"] * 0.6) & (times < (m["start_beats"] + m["dur_beats"]) * 0.6)] = m["midi"]
    amel.write_vocal_f0(demo_song.vocal_f0, vocal, times, contour, np.ones(len(times)))
    monkeypatch.setattr(amel, "vocal_pitch", lambda *a, **k: pytest.fail("cache should be used"))
    ls["bars"][1]["melody"][0]["midi"] = 71                     # 参考第 2 小节第 1 个音改成 B4：这一小节被标出
    ls["bars"][2]["melody"][0]["out_of_key"] = True             # 第 3 小节有调外音：需人耳确认
    demo_song.leadsheet_json.write_text(json.dumps(ls), encoding="utf-8")
    from guitarize.verify import report as vreport
    monkeypatch.setattr(vreport, "MAX_SNIPPETS", 1)             # 旋律谱列出的小节全部出试听和音高图，不受这个上限截断
    assert main(["--songs-dir", songs, "verify", "demo", "--arr", "melody"]) == 0
    it = demo_song.report_root("melody") / "iter-01"
    scores = json.loads((it / "scores.json").read_text(encoding="utf-8"))
    s = scores["summary"]
    assert 2 in scores["problems"] and s["violations"] == 0 and s["vocal_similarity"] is not None
    assert s["ear_check_bars"] == 1 and "chord" not in s["checks"]
    for name in ("plots/bar-002.png", "plots/bar-003.png", "snippets/bar-002_vocal.wav", "snippets/bar-002_synth.wav",
                 "snippets/bar-003_vocal.wav", "snippets/bar-003_synth.wav"):
        assert (it / name).exists(), name
    assert not (it / "snippets" / "bar-002_orig.wav").exists()
    report = (it / "report.md").read_text(encoding="utf-8")
    assert "旋律校验报告" in report and "旋律召回率（全部参考音）" in report and "人声相似度" in report
    assert "和弦一致率" not in report and "对照的人声分轨：analysis/stems/vocals.wav" in report
    assert "试听：snippets/bar-002_vocal.wav ↔ snippets/bar-002_synth.wav" in report
    assert "音高图：plots/bar-002.png" in report
    assert "第 3 小节：拍 1 F4（调外）" in report
    rows = list(csv.DictReader((demo_song.report_root("melody") / "history.csv").open(encoding="utf-8")))
    assert rows[0]["vocal_similarity"] != "" and rows[0]["chord_agreement"] == ""
    assert main(["--songs-dir", songs, "render", "demo", "--arr", "melody"]) == 0
    assert demo_song.out("melody", "html").exists() and "旋律" in demo_song.out("melody", "html").read_text(encoding="utf-8")
    assert "e|" in demo_song.out("melody", "txt").read_text(encoding="utf-8")


def test_melody_is_checked_against_the_stem_it_was_transcribed_from(tmp_path):
    # melody 步骤默认从居中的主唱 stems/lead.wav 识别旋律（overrides.melody_source: vocals 时用整条人声）：
    # 旋律谱的人声相似度、试听片段、音高图都对照同一条分轨；没有 lead.wav 时退回 vocals.wav
    from guitarize.paths import SongPaths
    from guitarize.verify.report import melody_reference

    paths = SongPaths(tmp_path / "song")
    paths.stems_dir.mkdir(parents=True)
    assert melody_reference(paths, {}) == paths.stem("vocals")
    paths.stem("lead").write_bytes(b"RIFF")
    assert melody_reference(paths, {}) == paths.stem("lead")
    assert melody_reference(paths, {"melody_source": "lead"}) == paths.stem("lead")
    assert melody_reference(paths, {"melody_source": "vocals"}) == paths.stem("vocals")
    with pytest.raises(ValueError, match=r'overrides\.melody_source must be "lead" or "vocals"'):
        melody_reference(paths, {"melody_source": "mix"})
    # melody.json 记着旋律是从哪条分轨识别的（melody 步骤写的 "source"）：以它为准，不看 song.yaml 现在怎么写
    paths.melody_json.write_text(json.dumps({"notes": [], "source": "vocals"}), encoding="utf-8")
    assert melody_reference(paths, {}) == paths.stem("vocals")
    assert melody_reference(paths, {"melody_source": "lead"}) == paths.stem("vocals")
    paths.melody_json.write_text(json.dumps({"notes": [], "source": "lead"}), encoding="utf-8")
    assert melody_reference(paths, {"melody_source": "vocals"}) == paths.stem("lead")
    paths.stem("lead").unlink()                                 # lead.wav 不在了：退回整条人声
    assert melody_reference(paths, {}) == paths.stem("vocals")
    paths.melody_json.write_text(json.dumps({"notes": [], "source": "mix"}), encoding="utf-8")
    with pytest.raises(ValueError, match=r'melody\.json source must be "lead" or "vocals", got \'mix\''):
        melody_reference(paths, {})
    paths.melody_json.write_text(json.dumps({"notes": []}), encoding="utf-8")   # 旧的 melody.json：按 song.yaml
    paths.stem("lead").write_bytes(b"RIFF")
    assert melody_reference(paths, {"melody_source": "vocals"}) == paths.stem("vocals")
    assert melody_reference(paths, {}) == paths.stem("lead")
