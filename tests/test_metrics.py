import copy

import numpy as np
import pytest

from conftest import demo_leadsheet
from guitarize.config import Thresholds
from guitarize.render import alphatab
from guitarize.render.synth import midi_to_wav
from guitarize.verify import metrics
from guitarize.verify.model import load_score


@pytest.fixture
def demo_scores(demo_song, tmp_path):
    out = {}
    for arr in ("finger", "strum"):
        built = alphatab.build(demo_song.atex(arr), tmp_path / "b", arr)
        out[arr] = (load_score(built["json"]), built["mid"])
    return out


def test_f1():
    assert metrics.f1(0, 0, 0, 0) is None
    assert metrics.f1(4, 4, 4, 4) == 1.0
    assert metrics.f1(2, 4, 1, 2) == pytest.approx(0.5)


def test_demo_finger_perfect(demo_scores):
    score, _ = demo_scores["finger"]
    bars = metrics.score_bars(score, demo_leadsheet(), "finger")
    s = metrics.summarize(bars, [], Thresholds(), "finger")
    assert s["chord_agreement"] == 1.0
    assert s["melody_precision"] == 1.0 and s["melody_recall"] == 1.0 and s["melody_f1"] == 1.0
    assert s["melody_octave_exact"] == 1.0
    assert s["pass"] is True
    assert metrics.problem_bars(bars, [], Thresholds(), "finger") == []


def test_wrong_reference_chord_flagged(demo_scores):
    score, _ = demo_scores["strum"]
    ls = demo_leadsheet()
    ls["bars"][0]["chords"] = [{"beat": 1, "dur_beats": 4, "label": "Am", "confidence": "high"}]
    bars = metrics.score_bars(score, ls, "strum")
    assert bars[0].chord_agree == 0 and bars[0].chord_total == 4
    assert "参考 Am" in bars[0].chord_issues[0]
    assert metrics.problem_bars(bars, [], Thresholds(), "strum") == [1]


def test_low_confidence_beats_excluded(demo_scores):
    score, _ = demo_scores["strum"]
    ls = demo_leadsheet()
    ls["bars"][0]["chords"] = [{"beat": 1, "dur_beats": 4, "label": "Am", "confidence": "low"}]
    bars = metrics.score_bars(score, ls, "strum")
    assert bars[0].chord_total == 0 and bars[0].low_conf_beats == [1, 2, 3, 4]


def test_melody_wrong_note(demo_scores):
    score, _ = demo_scores["finger"]
    ls = demo_leadsheet()
    ls["bars"][0]["melody"][2]["midi"] = 69  # 参考第 1 小节第 3 拍改成 A4
    bars = metrics.score_bars(score, ls, "finger")
    assert (bars[0].mel_hit, bars[0].mel_est) == (3, 4)
    assert (bars[0].mel_found, bars[0].mel_ref) == (3, 4)
    assert any("漏掉骨架音" in x for x in bars[0].mel_issues)
    assert any("多余/错音" in x for x in bars[0].mel_issues)


def test_melody_octave_tolerated(demo_scores):
    score, _ = demo_scores["finger"]
    ls = copy.deepcopy(demo_leadsheet())
    for b in ls["bars"]:
        for n in b["melody"]:
            n["midi"] -= 12
    s = metrics.summarize(metrics.score_bars(score, ls, "finger"), [], Thresholds(), "finger")
    assert s["melody_f1"] == 1.0 and s["melody_octave_exact"] == 0.0


def test_bars_without_reference_melody_not_scored(demo_scores):
    score, _ = demo_scores["finger"]
    ls = demo_leadsheet()
    ls["bars"][0]["melody"] = []
    bars = metrics.score_bars(score, ls, "finger")
    assert bars[0].mel_est == 0 and bars[0].mel_f1 is None


def test_harmonic_similarity_self(demo_scores, tmp_path):
    score, mid = demo_scores["finger"]
    wav = midi_to_wav(mid, tmp_path / "f.wav")
    bars = metrics.score_bars(score, demo_leadsheet(), "finger", synth_wav=wav, harmonic_wav=wav)
    sims = [b.harmonic_sim for b in bars]
    assert all(s is not None and s > 0.99 for s in sims)


def test_chord_mark_mismatch(build_tex):
    from conftest import TEX_HEAD

    score = build_tex('\\chord "G" 3 0 0 0 2 3\n:1 (0.1 1.2 0.3 2.4 3.5) {ch "G" sd} |', head=TEX_HEAD)
    bars = metrics.score_bars(score, None, "strum")
    assert bars[0].mark_issues and "标记 G" in bars[0].mark_issues[0]
    s = metrics.summarize(bars, [], Thresholds(), "strum")
    assert s["chord_mark_issues"] == 1 and s["checks"]["chord_marks"] is False
    assert metrics.problem_bars(bars, [], Thresholds(), "strum") == [1]


def test_demo_strum_marks_consistent(demo_scores):
    score, _ = demo_scores["strum"]
    bars = metrics.score_bars(score, demo_leadsheet(), "strum")
    assert all(not b.mark_issues for b in bars)
    assert metrics.summarize(bars, [], Thresholds(), "strum")["pass"] is True


def test_strum_vector_and_pearson(demo_scores):
    score, _ = demo_scores["strum"]
    v = metrics.strum_vector(score.attacks(), score.bars[0])
    # 每半小节：↓ 在 0，↓ 在 1 拍（slot 3），↑ 在 1.67 拍（slot 5）
    assert list(np.flatnonzero(v)) == [0, 3, 5, 6, 9, 11]
    assert v[5] == pytest.approx(0.6) and v[0] == 1.0
    assert metrics.pearson(v, v) == pytest.approx(1.0)
    assert metrics.pearson(np.ones(12), v) is None


# ---------- 评审修复轮 1 ----------

def _bare_leadsheet(melody_by_bar: dict[int, list[dict]], nbars: int = 8) -> dict:
    """只保留指定小节的旋律、清空所有和弦的参考谱，用来单独考核旋律逻辑。"""
    ls = demo_leadsheet()
    ls["bars"] = ls["bars"][:nbars]
    for i, b in enumerate(ls["bars"]):
        b["chords"] = []
        b["melody"] = melody_by_bar.get(i, [])
    return ls


def _note(start: float, dur: float, midi: int = 64) -> dict:
    return {"start_beats": start, "dur_beats": dur, "midi": midi, "conf": 1.0}


def test_bar_count_mismatch_fails_summary(demo_scores):
    score, _ = demo_scores["finger"]
    ls = demo_leadsheet()
    ls["bars"] = ls["bars"][:4]
    bars = metrics.score_bars(score, ls, "finger")
    assert len(bars) == 4  # 逐小节评分仍按较短的一方截断
    s = metrics.summarize(bars, [], Thresholds(), "finger", arrangement_bars=len(score.bars), reference_bars=4)
    assert (s["arrangement_bars"], s["reference_bars"]) == (8, 4)
    assert s["checks"]["bar_count"] is False and s["pass"] is False


def test_bar_count_match_passes_and_is_optional(demo_scores):
    score, _ = demo_scores["finger"]
    bars = metrics.score_bars(score, demo_leadsheet(), "finger")
    s = metrics.summarize(bars, [], Thresholds(), "finger", arrangement_bars=8, reference_bars=8)
    assert s["checks"]["bar_count"] is True and s["pass"] is True
    s = metrics.summarize(bars, [], Thresholds(), "finger")
    assert "bar_count" not in s["checks"]
    assert s["arrangement_bars"] is None and s["reference_bars"] is None


def test_no_reference_never_passes(demo_scores):
    score, _ = demo_scores["finger"]
    bars = metrics.score_bars(score, None, "finger")
    s = metrics.summarize(bars, [], Thresholds(), "finger", has_reference=False)
    assert s["checks"]["reference"] is False and s["pass"] is False


def test_all_low_confidence_reference_never_passes(demo_scores):
    score, _ = demo_scores["strum"]
    ls = demo_leadsheet()
    for b in ls["bars"]:
        for c in b["chords"]:
            c["confidence"] = "low"
    bars = metrics.score_bars(score, ls, "strum")
    s = metrics.summarize(bars, [], Thresholds(), "strum")
    assert s["scored_beats"] == 0 and s["chord_agreement"] is None
    assert s["checks"]["reference"] is False and s["pass"] is False


def test_reference_check_counts_scored_material(demo_scores):
    fscore, _ = demo_scores["finger"]
    s = metrics.summarize(metrics.score_bars(fscore, demo_leadsheet(), "finger"), [], Thresholds(), "finger")
    assert s["scored_beats"] == 32 and s["melody_ref_notes"] > 0 and s["checks"]["reference"] is True
    ls = demo_leadsheet()
    for b in ls["bars"]:
        b["melody"] = []  # 指弹但参考里没有任何旋律：没有东西可核对
    s = metrics.summarize(metrics.score_bars(fscore, ls, "finger"), [], Thresholds(), "finger")
    assert s["melody_ref_notes"] == 0 and s["checks"]["reference"] is False and s["pass"] is False


def test_f1_none_when_no_reference_notes():
    assert metrics.f1(1, 1, 0, 0) is None  # 参考没有骨架音：召回率无定义
    assert metrics.f1(0, 3, 0, 0) is None
    assert metrics.f1(0, 0, 0, 2) == 0.0   # 有参考音但编配一个都没弹：召回 0，仍是 0
    assert metrics.f1(0, 0, 2, 2) == 1.0   # 编配这一小节没有旋律音，但参考骨架音都被邻小节的音覆盖：取召回率
    assert metrics.f1(0, 0, 1, 4) == pytest.approx(0.25)


def test_bar_with_only_non_skeleton_reference_not_penalised(build_tex):
    from conftest import TEX_HEAD

    # 参考第 1 个音必为骨架音，所以非骨架音放在第 2 小节：4.5 拍、半拍长、前一个音刚结束 → 非骨架
    ls = _bare_leadsheet({0: [_note(3.0, 1.0)], 1: [_note(4.5, 0.5)]}, nbars=2)
    score = build_tex(':4 r r r 0.1 |\n:8 r 0.1 :4 r :2 r |', head=TEX_HEAD)
    bars = metrics.score_bars(score, ls, "finger")
    assert (bars[1].mel_ref, bars[1].mel_est, bars[1].mel_hit) == (0, 1, 1)
    assert bars[1].mel_f1 is None
    assert metrics.problem_bars(bars, [], Thresholds(), "finger") == []


def test_bar_with_only_non_skeleton_reference_flags_wrong_notes(build_tex):
    from conftest import TEX_HEAD

    ls = _bare_leadsheet({0: [_note(3.0, 1.0)], 1: [_note(4.5, 0.5)]}, nbars=2)
    score = build_tex(':4 r r r 0.1 |\n:8 r 1.1 :4 r :2 r |', head=TEX_HEAD)  # 第 2 小节弹成 F 而不是 E
    bars = metrics.score_bars(score, ls, "finger")
    assert (bars[1].mel_ref, bars[1].mel_est, bars[1].mel_hit) == (0, 1, 0)
    assert bars[1].mel_f1 is None
    assert metrics.problem_bars(bars, [], Thresholds(), "finger") == [2]


def test_skeleton_note_matched_by_arrangement_note_in_unreferenced_next_bar(build_tex):
    from conftest import TEX_HEAD

    # 参考第 1 小节只有 3+2/3 拍处一个骨架音（全曲第一个音，必为骨架音）；第 2 小节没有参考旋律。
    # 编配在第 2 小节 4.0 拍弹同名音，距离 1/3 拍，在容差内，应当算作覆盖。
    ls = _bare_leadsheet({0: [_note(3 + 2 / 3, 1 / 3)]}, nbars=2)
    score = build_tex(':1 r |\n:1 0.1 |', head=TEX_HEAD)
    bars = metrics.score_bars(score, ls, "finger")
    assert bars[0].mel_ref == 1 and bars[0].mel_found == bars[0].mel_ref
    assert not any("漏掉骨架音" in x for x in bars[0].mel_issues)
    # 无参考旋律的第 2 小节不计入精确率
    assert (bars[1].mel_est, bars[1].mel_hit, bars[1].mel_ref) == (0, 0, 0)
    # 本小节没有编配旋律音（精确率无定义），骨架音都被覆盖 → F1 取召回率 1.0，不该被标成问题小节
    assert bars[0].mel_est == 0 and bars[0].mel_f1 == 1.0
    assert metrics.problem_bars(bars, [], Thresholds(), "finger") == []


def test_reference_skeleton_but_no_arrangement_melody_is_a_problem(build_tex):
    from conftest import TEX_HEAD

    ls = _bare_leadsheet({0: [_note(0.0, 1.0), _note(2.0, 1.0)]}, nbars=1)
    score = build_tex(':1 r |', head=TEX_HEAD)  # 编配这一小节整个休止
    bars = metrics.score_bars(score, ls, "finger")
    assert (bars[0].mel_est, bars[0].mel_ref, bars[0].mel_found) == (0, 2, 0)
    assert bars[0].mel_f1 == 0.0
    assert metrics.problem_bars(bars, [], Thresholds(), "finger") == [1]


def test_onset_slots_finds_clicks_in_expected_slots(tmp_path):
    import soundfile as sf

    sr = 22050
    windows = [(0.0, 2.0), (2.0, 4.0)]
    step = 2.0 / metrics.SLOTS
    clicks = {0: [1, 4, 7], 1: [2, 8]}
    y = np.zeros(4 * sr)
    t = np.arange(int(0.05 * sr)) / sr
    burst = np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 60)
    for w, slots in clicks.items():
        for k in slots:
            i = int((windows[w][0] + k * step) * sr)
            y[i:i + len(burst)] += burst
    wav = tmp_path / "clicks.wav"
    sf.write(str(wav), y, sr)
    out = metrics.onset_slots([wav], windows)
    assert out.shape == (2, metrics.SLOTS)
    for w, slots in clicks.items():
        assert sorted(np.argsort(out[w])[-len(slots):].tolist()) == slots


def test_beat_label():
    from guitarize.verify.positions import beat_label

    assert beat_label(0) == "1" and beat_label(2.0) == "3" and beat_label(2.999999) == "4"
    assert beat_label(2.5) == "3&" and beat_label(2 + 2 / 3) == "3&"
    assert beat_label(2 + 1 / 3) == "3+⅓"
    assert beat_label(0.25) == "1e" and beat_label(3.75) == "4a"
    assert beat_label(0.4) == "1.40"
