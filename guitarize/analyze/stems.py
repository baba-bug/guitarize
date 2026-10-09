"""音源分离：Demucs htdemucs_6s → analysis/stems/*.wav，并合成和声分轨 harmonic.wav（去人声、去鼓）。"""
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
    sf.write(str(out), total, sr)
    return out


def separate(mix: Path, stems_dir: Path, model: str = "htdemucs_6s") -> dict[str, Path]:
    stems_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="guitarize-demucs-") as tmp:
        subprocess.run([sys.executable, "-m", "demucs", "-n", model, "-d", _device(), "-o", tmp, str(mix)],
                       check=True, capture_output=True)
        produced = Path(tmp) / model / Path(mix).stem
        for name in STEMS:
            shutil.move(str(produced / f"{name}.wav"), str(stems_dir / f"{name}.wav"))
    mix_harmonic(stems_dir)
    return {name: stems_dir / f"{name}.wav" for name in (*STEMS, "harmonic")}
