import sys
import types

import numpy as np
import pytest
import soundfile as sf


def test_extract_lead_keeps_the_centre_and_drops_panned_backing_vocals(tmp_path):
    # 主唱居中、伴唱摆在左右：只保留左右一致的成分（なんもねえ 副歌 6–8、14–15 小节的伴唱就是这样混的）
    from guitarize.analyze.stems import extract_lead
    sr = 22050
    t = np.arange(sr * 2) / sr
    lead = 0.3 * np.sin(2 * np.pi * 220 * t)          # 居中
    backing = 0.3 * np.sin(2 * np.pi * 330 * t)       # 只在左声道
    src = tmp_path / "vocals.wav"
    sf.write(src, np.stack([lead + backing, lead], axis=1), sr)
    out = extract_lead(src, tmp_path / "lead.wav")
    y, _ = sf.read(out)
    mono = y.mean(axis=1)
    spec = np.abs(np.fft.rfft(mono))
    f = np.fft.rfftfreq(len(mono), 1 / sr)
    at = lambda hz: spec[np.argmin(np.abs(f - hz))]
    assert at(220) > 5 * at(330)                     # 居中的主唱留下，单边的伴唱基本去掉
    assert y.shape == (len(t), 2)


# ---------- separate_lead：伴唱分离模型（audio-separator），不可用时退回 L/R 掩码 ----------

SR = 22050


def _tone(hz, seconds=1.0):
    t = np.arange(int(SR * seconds)) / SR
    return np.stack([0.3 * np.sin(2 * np.pi * hz * t)] * 2, axis=1)


def _model_dir(tmp_path):
    from guitarize.analyze.stems import KARAOKE_FILES

    d = tmp_path / "models"
    d.mkdir()
    for name in KARAOKE_FILES:
        (d / name).write_bytes(b"x")
    return d


def _fake_separator(monkeypatch, calls, fail=None):
    """假的 audio_separator.separator.Separator：Vocals（主唱）写 220 Hz、Instrumental（伴唱）写 330 Hz，
    和真的一样只返回相对 output_dir 的文件名。"""
    class Separator:
        def __init__(self, **kw):
            calls.append(("init", kw))
            self.out = kw["output_dir"]

        def load_model(self, model_filename):
            calls.append(("load_model", model_filename))

        def separate(self, audio_file_path, custom_output_names=None):
            calls.append(("separate", audio_file_path, custom_output_names))
            sf.write(f"{self.out}/{custom_output_names['Vocals']}.wav", _tone(220), SR)
            if fail:
                raise RuntimeError(fail)
            sf.write(f"{self.out}/{custom_output_names['Instrumental']}.wav", _tone(330), SR)
            return [f"{custom_output_names['Instrumental']}.wav", f"{custom_output_names['Vocals']}.wav"]

    monkeypatch.setitem(sys.modules, "audio_separator.separator", types.SimpleNamespace(Separator=Separator))


def _peak_hz(path):
    y, sr = sf.read(path)
    spec = np.abs(np.fft.rfft(y.mean(axis=1)))
    return np.fft.rfftfreq(len(y), 1 / sr)[np.argmax(spec)]


def test_separate_lead_runs_the_karaoke_model_and_writes_lead_and_backing(tmp_path, monkeypatch, capsys):
    # 伴唱不论摆在哪儿都能和主唱分开（なんもねえ 副歌伴唱持续的 B4 混在中间，L/R 掩码去不掉）
    from guitarize.analyze.stems import KARAOKE_METHOD, KARAOKE_MODEL, separate_lead

    models = _model_dir(tmp_path)
    monkeypatch.setenv("GUITARIZE_KARAOKE_DIR", str(models))   # 不传 model_dir：取环境变量
    calls = []
    _fake_separator(monkeypatch, calls)
    stems = tmp_path / "stems"
    stems.mkdir()
    vocals = stems / "vocals.wav"
    sf.write(vocals, _tone(220) + _tone(330), SR)
    method = separate_lead(vocals, stems / "lead.wav", stems / "backing.wav")
    assert method == KARAOKE_METHOD == "karaoke:mel_band_roformer_karaoke_becruily"
    assert round(_peak_hz(stems / "lead.wav")) == 220 and round(_peak_hz(stems / "backing.wav")) == 330
    (_, init), load, sep = calls
    assert init["model_file_dir"] == str(models) and init["output_format"] == "WAV"
    assert load == ("load_model", KARAOKE_MODEL)
    assert sep == ("separate", str(vocals), {"Vocals": "lead", "Instrumental": "backing"})   # 传 str：Path 会被当成列表
    assert sorted(p.name for p in stems.iterdir()) == ["backing.wav", "lead.wav", "vocals.wav"]   # 临时目录清掉了
    assert capsys.readouterr().err == ""


def test_separate_lead_leaves_the_old_files_alone_when_the_model_fails(tmp_path, monkeypatch):
    # 先写进临时目录，两个文件都出来了才换上去：中途失败不留下半套
    from guitarize.analyze.stems import separate_lead

    _fake_separator(monkeypatch, [], fail="MPS out of memory")
    stems = tmp_path / "stems"
    stems.mkdir()
    sf.write(stems / "vocals.wav", _tone(220), SR)
    (stems / "lead.wav").write_bytes(b"old")
    with pytest.raises(RuntimeError, match="MPS out of memory"):
        separate_lead(stems / "vocals.wav", stems / "lead.wav", stems / "backing.wav", model_dir=_model_dir(tmp_path))
    assert (stems / "lead.wav").read_bytes() == b"old"
    assert sorted(p.name for p in stems.iterdir()) == ["lead.wav", "vocals.wav"]


def _panned_vocals(path):
    t = np.arange(SR * 2) / SR
    lead, backing = 0.3 * np.sin(2 * np.pi * 220 * t), 0.3 * np.sin(2 * np.pi * 330 * t)
    sf.write(path, np.stack([lead + backing, lead], axis=1), SR)


def test_separate_lead_falls_back_to_the_lr_mask_when_the_model_is_missing(tmp_path, capsys):
    from guitarize.analyze.stems import LR_MASK_METHOD, separate_lead

    _panned_vocals(tmp_path / "vocals.wav")
    method = separate_lead(tmp_path / "vocals.wav", tmp_path / "lead.wav", tmp_path / "backing.wav",
                           model_dir=tmp_path / "nowhere")
    assert method == LR_MASK_METHOD == "lr-mask"
    assert round(_peak_hz(tmp_path / "lead.wav")) == 220            # 居中的主唱，L/R 掩码抽出来的
    assert not (tmp_path / "backing.wav").exists()                   # 掩码分不出伴唱
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "lr-mask" in err and "mel_band_roformer_karaoke_becruily.ckpt" in err


def test_separate_lead_falls_back_when_audio_separator_cannot_be_imported(tmp_path, monkeypatch, capsys):
    from guitarize.analyze.stems import separate_lead

    monkeypatch.setitem(sys.modules, "audio_separator.separator", None)   # import 失败
    _panned_vocals(tmp_path / "vocals.wav")
    method = separate_lead(tmp_path / "vocals.wav", tmp_path / "lead.wav", tmp_path / "backing.wav",
                           model_dir=_model_dir(tmp_path))
    assert method == "lr-mask"
    assert (tmp_path / "lead.wav").exists() and not (tmp_path / "backing.wav").exists()
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "audio-separator" in err and "lr-mask" in err


def test_karaoke_model_dir_defaults_to_tools_karaoke_models(tmp_path, monkeypatch):
    from guitarize.analyze.stems import karaoke_model_dir
    from guitarize.paths import REPO_ROOT

    monkeypatch.delenv("GUITARIZE_KARAOKE_DIR")
    assert karaoke_model_dir() == REPO_ROOT / "tools" / "karaoke" / "models"
    monkeypatch.setenv("GUITARIZE_KARAOKE_DIR", str(tmp_path))
    assert karaoke_model_dir() == tmp_path
