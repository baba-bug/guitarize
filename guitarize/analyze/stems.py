"""音源分离：Demucs htdemucs_6s → analysis/stems/*.wav，并合成和声分轨 harmonic.wav（去人声、去鼓）、
从人声分轨里抽出居中的主唱 lead.wav（去掉摆在左右的伴唱，旋律只从主唱识别）。"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")
HARMONIC_PARTS = ("bass", "guitar", "piano", "other")


def _device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def mix_harmonic(stems_dir: Path) -> Path:
    total, sr = None, None
    for name in HARMONIC_PARTS:
        y, sr = sf.read(str(Path(stems_dir) / f"{name}.wav"), always_2d=True)
        if total is None:
            total = y
        else:
            n = max(len(total), len(y))
            total = np.pad(total, ((0, n - len(total)), (0, 0))) + np.pad(y, ((0, n - len(y)), (0, 0)))
    out = Path(stems_dir) / "harmonic.wav"
    sf.write(str(out), total, sr, subtype="FLOAT")
    return out


def separate(mix: Path, stems_dir: Path, model: str = "htdemucs_6s") -> dict[str, Path]:
    stems_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="guitarize-demucs-") as tmp:
        proc = subprocess.run([sys.executable, "-m", "demucs", "-n", model, "-d", _device(), "-o", tmp, str(mix)],
                              capture_output=True)
        if proc.returncode != 0:
            err = (proc.stderr or b"").decode("utf-8", "replace")[-2000:]
            raise RuntimeError(f"demucs failed (exit {proc.returncode}):\n{err}")
        produced = Path(tmp) / model / Path(mix).stem
        missing = [n for n in STEMS if not (produced / f"{n}.wav").exists()]
        if missing:
            raise RuntimeError(f"demucs did not produce stems {missing} in {produced}")
        for name in STEMS:
            shutil.move(str(produced / f"{name}.wav"), str(stems_dir / f"{name}.wav"))
    mix_harmonic(stems_dir)
    return {name: stems_dir / f"{name}.wav" for name in (*STEMS, "harmonic")}


LEAD_MASK_POWER = 4.0   # 掩码锐度：越大越只留严格居中的成分


def extract_lead(vocals_wav: Path, out_wav: Path, power: float = LEAD_MASK_POWER) -> Path:
    """主唱 = 人声分轨里左右声道一致（居中、同相）的成分。逐频点软掩码 (声像平衡 × 相位一致)^power 乘在 mid 上。
    伴唱 / 和声多半摆在左右（なんもねえ 副歌第 6–8、14–15 小节）；直接用整条人声识别旋律会跳到伴唱的音上。
    单声道输入原样输出（无法区分）。"""
    import librosa
    import soundfile as sf

    y, sr = sf.read(str(vocals_wav), always_2d=True)
    if y.shape[1] < 2:
        sf.write(str(out_wav), np.repeat(y[:, :1], 2, axis=1), sr)
        return Path(out_wav)
    left, right = y[:, 0], y[:, 1]
    n_fft, hop = 4096, 1024
    sl = librosa.stft(left, n_fft=n_fft, hop_length=hop)
    sr_ = librosa.stft(right, n_fft=n_fft, hop_length=hop)
    al, ar = np.abs(sl), np.abs(sr_)
    balance = 2 * al * ar / (al ** 2 + ar ** 2 + 1e-12)
    phase = np.cos(np.angle(sl) - np.angle(sr_)).clip(0, 1)
    mask = (balance * phase) ** power
    lead = librosa.istft((sl + sr_) / 2 * mask, hop_length=hop, length=len(left))
    sf.write(str(out_wav), np.stack([lead, lead], axis=1), sr)
    return Path(out_wav)
