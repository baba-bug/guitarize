"""MIDI → wav：FluidSynth + alphaTab 自带的 sonivox 音色库。"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..paths import REPO_ROOT

ALPHATAB_DIST = REPO_ROOT / "tools" / "alphatab" / "node_modules" / "@coderline" / "alphatab" / "dist"
SOUNDFONT = ALPHATAB_DIST / "soundfont" / "sonivox.sf2"


def midi_to_wav(mid: Path, wav: Path, sr: int = 44100) -> Path:
    exe = shutil.which("fluidsynth")
    if exe is None:
        raise RuntimeError("fluidsynth not found; run scripts/setup_env.sh")
    wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [exe, "-ni", "-g", "0.8", "-r", str(sr), "-F", str(wav), str(SOUNDFONT), str(mid)],
        check=True,
        capture_output=True,
    )
    return wav
