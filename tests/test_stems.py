

def test_extract_lead_keeps_the_centre_and_drops_panned_backing_vocals(tmp_path):
    # 主唱居中、伴唱摆在左右：只保留左右一致的成分（なんもねえ 副歌 6–8、14–15 小节的伴唱就是这样混的）
    import numpy as np
    import soundfile as sf
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
