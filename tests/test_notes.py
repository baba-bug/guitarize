"""旋律 v7（analyze/notes.py）：GAME 的分音 + 主唱信号检查。全部用合成的逐帧信号和假的 GAME 音，不跑 GAME / crepe。"""
import numpy as np
import pytest
import soundfile as sf

from guitarize import theory
from guitarize.analyze import notes as v7
from guitarize.analyze.key import scale_pcs

B_MAJOR = scale_pcs("B", "major")
FPS = 100


def _beats(n_bars=4, spb=0.5, feel="shuffle"):
    """拍在 k·spb 秒（spb = 0.5 → 120 BPM），4/4，没有弱起：拍轴 0 = 第 1 小节第 1 拍。"""
    beats = [{"t": k * spb, "bar": k // 4 + 1, "beat": k % 4 + 1, "g": k} for k in range(n_bars * 4)]
    bars = [{"bar": b + 1, "t_start": b * 4 * spb, "t_end": (b + 1) * 4 * spb, "beats": 4} for b in range(n_bars)]
    return {"tempo_bpm": 60 / spb, "feel": feel, "time_signature": [4, 4], "pickup_beats": 0, "pad_beats": 0,
            "short_bars": [], "beats": beats, "bars": bars}


def _labels(label="E", n_bars=4):
    return {b: [(1, label)] for b in range(1, n_bars + 1)}


def _frames(seconds=4.0, sung=(), dips=(), backing=(), lr=(), gaps=()):
    """合成主唱的逐帧特征（10 ms 一帧）。sung：[(起秒, 止秒, MIDI)] 主唱在唱（周期性 0.9、音量 0 dB），其余安静（−80 dB）；
    dips：[(秒, dB)] 平底的音量下陷（换字）；backing：[(起秒, 止秒)] 伴唱轨比主唱响 10 dB；lr：[(起秒, 止秒)] 左右不相关；
    gaps：[(起秒, 止秒, dB)] 无声的辅音（周期性掉到 0，音量降 dB）。"""
    n = int(seconds * FPS)
    midi, per, lead_db, back_db, lrc = np.full(n, np.nan), np.zeros(n), np.full(n, -80.0), np.full(n, -80.0), np.ones(n)
    fr = lambda t: int(round(t * FPS))
    for a, e, m in sung:
        midi[fr(a):fr(e)], per[fr(a):fr(e)], lead_db[fr(a):fr(e)] = m, 0.9, 0.0
    for t, depth in dips:
        c = fr(t)
        lead_db[c - 4:c + 4] -= depth * np.array([1 / 3, 2 / 3, 1, 1, 1, 1, 2 / 3, 1 / 3])
    for a, e, depth in gaps:
        midi[fr(a):fr(e)], per[fr(a):fr(e)], lead_db[fr(a):fr(e)] = np.nan, 0.0, -depth
    for a, e in backing:
        back_db[fr(a):fr(e)] = 10.0
    for a, e in lr:
        lrc[fr(a):fr(e)] = 0.1
    return v7.signal_frames(midi, per, 10 ** (lead_db / 20), 10 ** (back_db / 20), lrc)


def _melody(rows, frames, label="E", feel="shuffle"):
    got = v7.game_melody(rows, frames, _beats(feel=feel), B_MAJOR, _labels(label))
    return [(n["start_beats"], n["dur_beats"], n["midi"], bool(n.get("legato"))) for n in got], got


# ---------- 逐帧特征：伴唱漏音、换字点 ----------

def test_lr_correlation_is_one_for_the_centre_and_low_for_unrelated_channels():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000)
    assert np.allclose(v7.lr_correlation(x, x)[5:-5], 1.0)
    assert np.abs(v7.lr_correlation(x, rng.standard_normal(16000))[5:-5]).max() < 0.2
    assert len(v7.lr_correlation(x, x)) == 1 + 16000 // 160            # 与 librosa 的 RMS（center）同帧数


def test_backing_leak_is_where_the_backing_is_louder_or_left_and_right_disagree():
    fr = _frames(sung=[(0.5, 3.0, 64)], backing=[(1.0, 1.5)], lr=[(2.0, 2.5)])
    assert fr.leak[110:140].all() and fr.leak[210:240].all()
    assert not fr.leak[60:90].any() and not fr.leak[160:190].any() and not fr.leak[260:290].any()
    assert not fr.voiced[110:140].any() and not fr.voiced[210:240].any()   # 漏音不算主唱在唱
    assert fr.voiced[60:90].all() and fr.voiced[160:190].all()


def test_a_6db_dip_is_a_syllable_onset_and_a_2db_vibrato_wobble_is_not():
    n = 400
    t = np.arange(n) / FPS
    lead_db = np.full(n, -80.0)
    lead_db[50:350] = 0.0
    lead_db[146:154] -= 6.5 * np.array([1 / 3, 2 / 3, 1, 1, 1, 1, 2 / 3, 1 / 3])     # 换字：音量下陷 6 dB 多
    lead_db[200:340] += np.sin(2 * np.pi * 5.5 * t[200:340])                        # 颤音：±1 dB（2 dB）的起伏
    midi, per = np.where(lead_db > -80, 64.0, np.nan), np.where(lead_db > -80, 0.9, 0.0)
    fr = v7.signal_frames(midi, per, 10 ** (lead_db / 20), np.full(n, 1e-6), np.ones(n))
    onsets = np.flatnonzero(fr.onset)
    assert onsets[0] == 50                                    # 出声
    assert len(onsets) == 2 and 150 <= onsets[1] <= 156       # 下陷之后音量回升到一半的那一帧
    strong = v7.strong_onsets(fr.db, fr.voiced)
    assert list(strong) == [onsets[1]]                        # ≥ 6 dB 也是强换字点；颤音区里什么都没有


def test_a_consonant_gap_is_a_strong_onset_only_with_an_energy_dip():
    loud = _frames(sung=[(0.5, 2.0, 64)], gaps=[(1.0, 1.05, 0.0)])     # 周期性掉了、音量没掉：沙哑 / 气声
    assert list(v7.strong_onsets(loud.db, loud.voiced)) == []
    dip = _frames(sung=[(0.5, 2.0, 64)], gaps=[(1.0, 1.05, 3.0)])      # 无声 5 帧、音量也掉 3 dB：辅音
    assert list(v7.strong_onsets(dip.db, dip.voiced)) == [105]


# ---------- 音高：GAME 与 crepe 的平均 + 按当时和弦吸附 ----------

def test_snap_chord_prefers_the_chord_tone_over_the_key():
    # B 大调：67.4 夹在 G（调外）和 G#（调内）之间。D#/G 上是 G（和弦音，不算调外），E 上是 G#
    assert v7.snap_chord(67.4, B_MAJOR, theory.chord_tones("D#/G")) == (67, False)
    assert v7.snap_chord(67.4, B_MAJOR, theory.chord_tones("E")) == (68, False)
    assert v7.snap_chord(67.4, B_MAJOR, set()) == (68, False)
    # 既不在调里、也不在当时和弦里的才标调外
    assert v7.snap_chord(62.0, {0, 4, 7}, set()) == (62, True)
    assert v7.snap_chord(62.0, {0, 4, 7}, theory.chord_tones("D")) == (62, False)


def test_chord_aware_snap_in_the_melody_does_not_flag_the_chord_tone():
    fr = _frames(sung=[(1.0, 1.5, 67.4)])
    got, notes = _melody([(1.0, 1.5, 67.4)], fr, label="D#/G")
    assert got == [(2.0, 1.0, 67, False)] and notes[0]["out_of_key"] is False
    got, notes = _melody([(1.0, 1.5, 67.4)], fr, label="E")
    assert got == [(2.0, 1.0, 68, False)]


def test_bar_chord_labels_follow_chords_json_and_overrides(tmp_path):
    from guitarize.analyze.beats import assemble_beats

    beats = assemble_beats([0.5 + 0.5 * k for k in range(11)], [1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], "straight")
    chords = {"segments": [{"bar": 1, "beat": 2, "dur_beats": 3, "label": "C"},
                           {"bar": 2, "beat": 1, "dur_beats": 4, "label": "G"},
                           {"bar": 3, "beat": 1, "dur_beats": 2, "label": "Am"},
                           {"bar": 3, "beat": 3, "dur_beats": 2, "label": "N"}]}
    labels = v7.bar_chord_labels(beats, chords, {2: "E F#", 1: "D"})    # 与 build_leadsheet 同一套解析
    assert labels == {1: [(2, "D")], 2: [(1, "E"), (3, "F#")], 3: [(1, "Am"), (3, "N")]}   # 弱起小节从第 2 拍起
    assert v7.chord_tones_at(labels, 4, 4.5) == theory.chord_tones("E")
    assert v7.chord_tones_at(labels, 4, 6.0) == theory.chord_tones("F#")
    assert v7.chord_tones_at(labels, 4, 10.0) == set()                  # N
    assert v7.chord_tones_at(labels, 4, 0.5) == set()                   # 弱起小节第 2 拍之前还没有和弦
    assert v7.chord_tones_at(labels, 4, 1.0) == theory.chord_tones("D")
    assert v7.chord_tones_at(labels, 4, 40.0) == theory.chord_tones("Am")   # 最后一小节之后按最后一小节


# ---------- 分音：漏音、长音、换字、legato ----------

def test_notes_that_are_mostly_backing_leak_are_dropped():
    fr = _frames(sung=[(0.5, 2.0, 64)], backing=[(0.5, 1.0)], lr=[(1.0, 1.5)])
    got, _ = _melody([(0.5, 1.0, 64.0), (1.0, 1.5, 66.0), (1.5, 2.0, 68.0)], fr)
    assert got == [(3.0, 1.0, 68, False)]


def test_a_held_note_split_by_game_is_plucked_once():
    # 他唱了很久的一个音，只用弹一下等它延音：GAME 拆成两个同音、中间没有换字 → 一个音
    fr = _frames(sung=[(1.0, 2.0, 64.0)])
    got, notes = _melody([(1.0, 1.5, 64.1), (1.5, 2.0, 63.9)], fr)
    assert got == [(2.0, 2.0, 64, False)]
    assert "legato" not in notes[0]                         # 不是 legato 的音不带这个键（旧的读法不受影响）


def test_the_same_pitch_after_a_real_dip_is_plucked_again():
    fr = _frames(sung=[(1.0, 2.0, 64.0)], dips=[(1.5, 8.0)])
    got, _ = _melody([(1.0, 1.5, 64.1), (1.5, 2.0, 63.9)], fr)
    assert got == [(2.0, 1.0, 64, False), (3.0, 1.0, 64, False)]
    got, _ = _melody([(1.0, 2.0, 64.0)], fr)                # GAME 并成一个：在强换字点切开，后半段重新拨
    assert got == [(2.0, 1.0, 64, False), (3.0, 1.0, 64, False)]


def test_a_pitch_change_without_an_onset_is_legato():
    fr = _frames(sung=[(1.0, 1.5, 64.0), (1.5, 2.0, 66.0)])
    got, notes = _melody([(1.0, 1.5, 64.0), (1.5, 2.0, 66.0)], fr)
    assert got == [(2.0, 1.0, 64, False), (3.0, 1.0, 66, True)]
    assert notes[1]["legato"] is True and "legato" not in notes[0]


def test_a_long_note_whose_sung_pitch_really_changes_is_split():
    # D#/G 上一个长音前半唱 G#4、后半落到 G4（各 ≥ 1 拍）：在那儿切开，后半段 legato
    fr = _frames(sung=[(1.0, 2.0, 68.0), (2.0, 3.0, 67.0)])
    got, _ = _melody([(1.0, 3.0, 67.6)], fr, label="D#/G")
    assert got == [(2.0, 2.0, 68, False), (4.0, 2.0, 67, True)]


def test_a_plucked_note_starts_at_the_syllable_onset_just_before_it():
    # GAME 把字头的上滑切掉了（起点晚 0.12 秒）：拉回到换字点
    fr = _frames(sung=[(1.0, 2.0, 64.0)])
    got, _ = _melody([(1.12, 2.0, 64.0)], fr)
    assert got == [(2.0, 2.0, 64, False)]


def test_a_short_syllable_initial_note_is_absorbed_into_the_legato_note_after_it():
    fr = _frames(sung=[(1.0, 1.1, 63.0), (1.1, 2.0, 64.0)])
    got, _ = _melody([(1.0, 1.1, 63.0), (1.1, 2.0, 64.0)], fr)
    assert got == [(2.0, 2.0, 64, False)]


# ---------- 量化 ----------

def test_shuffle_start_prefers_the_last_triplet_over_the_middle_one():
    assert v7.quantize_start(2.5, 3, shuffle=True) == pytest.approx(2 + 2 / 3)    # 正好夹在中间：取 +2/3
    assert v7.quantize_start(2.48, 3, shuffle=True) == pytest.approx(2 + 2 / 3)   # 略近 +1/3 也还是 +2/3
    assert v7.quantize_start(2.4, 3, shuffle=True) == pytest.approx(2 + 1 / 3)    # 证据够了才用 +1/3
    assert v7.quantize_start(2.48, 4, shuffle=False) == pytest.approx(2.5)        # 直拍：十六分格，不加罚


def test_straight_feel_uses_the_sixteenth_grid():
    fr = _frames(sung=[(1.0, 1.5, 64.0), (1.5, 2.0, 66.0)])
    got, _ = _melody([(1.0, 1.5, 64.0), (1.5, 2.0, 66.0)], fr, feel="straight")
    assert got == [(2.0, 1.0, 64, False), (3.0, 1.0, 66, True)]


# ---------- 整条：读音频、crepe 走 vocal_f0 缓存 ----------

def test_extract_melody_v7_reads_the_raw_lead_and_the_backing(tmp_path, monkeypatch):
    sr = 16000
    t = np.arange(3 * sr) / sr
    env = ((t >= 1.0) & (t < 2.0)).astype(float)
    tone = 0.3 * env * np.sin(2 * np.pi * 329.63 * t)
    sf.write(tmp_path / "lead_raw.wav", np.stack([tone, tone], axis=1), sr)    # 立体声、居中
    sf.write(tmp_path / "backing.wav", np.zeros(len(t)), sr)
    n = 1 + len(t) // 160
    midi, per = np.full(n, np.nan), np.zeros(n)
    midi[100:200], per[100:200] = 64.0, 0.9
    calls = []
    monkeypatch.setattr(v7, "cached_vocal_pitch",
                        lambda wav, cache: calls.append((wav, cache)) or (np.arange(n) / FPS, midi, per))
    got = v7.extract_melody_v7(tmp_path / "lead_raw.wav", tmp_path / "backing.wav", [(1.0, 2.0, 64.2)], _beats(),
                               B_MAJOR, _labels(), f0_cache=tmp_path / "vocal_f0.npz")
    assert calls == [(tmp_path / "lead_raw.wav", tmp_path / "vocal_f0.npz")]   # 基频缓存还是 analysis/vocal_f0.npz
    assert got["grid"] == "triplet8"
    assert got["notes"] == [{"start_beats": 2.0, "dur_beats": 2.0, "midi": 64, "out_of_key": False, "conf": 0.9}]
