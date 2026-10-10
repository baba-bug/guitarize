"""音源分离：Demucs htdemucs_6s → analysis/stems/*.wav，并合成和声分轨 harmonic.wav（去人声、去鼓）；
再用伴唱分离模型把人声分轨拆成主唱 lead.wav + 伴唱 backing.wav（旋律只从主唱识别）。"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

from ..paths import REPO_ROOT

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


# 伴唱分离（karaoke）模型：becruily 的 Mel-Band Roformer，用 audio-separator 跑。模型的 Vocals = 主唱，Instrumental = 伴唱。
KARAOKE_MODEL = "mel_band_roformer_karaoke_becruily.ckpt"
# audio-separator 按模型名查模型表 download_checks.json、读同名配置：三个文件都在本地才不会去联网下载
KARAOKE_FILES = (KARAOKE_MODEL, "config_mel_band_roformer_karaoke_becruily.yaml", "download_checks.json")
KARAOKE_OUTPUTS = {"Vocals": "lead", "Instrumental": "backing"}
KARAOKE_METHOD = f"karaoke:{Path(KARAOKE_MODEL).stem}"
LR_MASK_METHOD = "lr-mask"


def karaoke_model_dir() -> Path:
    """伴唱分离模型所在目录：环境变量 GUITARIZE_KARAOKE_DIR，否则仓库内 tools/karaoke/models（不入 git）。"""
    env = os.environ.get("GUITARIZE_KARAOKE_DIR")
    return Path(env) if env else REPO_ROOT / "tools" / "karaoke" / "models"


def separate_lead(vocals_wav: Path, lead_wav: Path, backing_wav: Path, model_dir: Path | None = None) -> str:
    """人声分轨 → 主唱 lead_wav + 伴唱 backing_wav，返回用的方法（KARAOKE_METHOD / LR_MASK_METHOD）。
    伴唱分离模型不管伴唱摆在哪儿都分得开（なんもねえ 副歌的伴唱持续唱 B4，混在中间，L/R 掩码去不掉，会漏进旋律）。
    设备由 audio-separator 自己选（CUDA > Apple 的 MPS > CPU）。两个文件先写进同目录下的临时目录，都出来了才换上去，
    中途失败不留下半套。没装 audio-separator 或模型文件不全时退回 extract_lead（L/R 掩码），stderr 打一行警告，
    不写 backing_wav。"""
    model_dir = Path(model_dir) if model_dir is not None else karaoke_model_dir()
    missing = [name for name in KARAOKE_FILES if not (model_dir / name).exists()]
    reason = f"{model_dir} 里没有 {', '.join(missing)}" if missing else None
    if reason is None:
        try:
            from audio_separator.separator import Separator
        except ImportError as e:
            reason = f"audio-separator 导入失败：{e}"
    if reason is not None:
        print(f"warning: 伴唱分离模型不可用（{reason}），主唱改用 L/R 掩码（{LR_MASK_METHOD}），不出伴唱分轨",
              file=sys.stderr)
        extract_lead(vocals_wav, lead_wav)
        return LR_MASK_METHOD
    lead_wav, backing_wav = Path(lead_wav), Path(backing_wav)
    lead_wav.parent.mkdir(parents=True, exist_ok=True)
    # Separator 构造时会全局 warnings.filterwarnings("ignore")：关在 catch_warnings 里，不影响之后的警告
    with tempfile.TemporaryDirectory(prefix=".guitarize-karaoke-", dir=lead_wav.parent) as tmp, \
            warnings.catch_warnings():
        sep = Separator(model_file_dir=str(model_dir), output_dir=tmp, output_format="WAV", log_level=logging.WARNING)
        sep.load_model(KARAOKE_MODEL)
        # 输入要传 str（Path 会被当成文件列表）；返回的是相对 output_dir 的文件名
        produced = [Path(tmp, f) for f in sep.separate(str(vocals_wav), custom_output_names=KARAOKE_OUTPUTS)]
        got = {f.name: f for f in produced if f.exists()}
        missing = [f"{name}.wav" for name in KARAOKE_OUTPUTS.values() if f"{name}.wav" not in got]
        if missing:
            raise RuntimeError(f"karaoke model did not produce {missing} in {tmp} (got {sorted(got)})")
        os.replace(got["lead.wav"], lead_wav)
        os.replace(got["backing.wav"], backing_wav)
    return KARAOKE_METHOD
