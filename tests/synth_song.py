"""闭环测试用的合成歌曲：已知节拍 / 和弦 / 旋律的 MIDI → 各声部 wav（FluidSynth + sonivox）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

from guitarize import theory
from guitarize.render.synth import midi_to_wav

TEMPO = 120.0
SPB = 60.0 / TEMPO
OFFSET = 0.25  # 第 1 小节起点（秒）
PROGRESSION = ["G", "D/F#", "Em", "C", "G", "D", "C", "D", "G", "D/F#", "Em", "C", "Am", "D", "G", "G"]


def bar_start(bar: int) -> float:
    return OFFSET + (bar - 1) * 4 * SPB


def _triad(label: str, low: int) -> list[int]:
    root, quality, _ = theory.parse_chord(label)
    r = low + ((root - low) % 12)
    return [r, r + (3 if quality == "min" else 4), r + 7]


def _bass(label: str) -> int:
    root, _, bass = theory.parse_chord(label)
    pc = bass if bass is not None else root
    return 40 + ((pc - 40) % 12)


def melody_truth() -> list[tuple[float, float, int]]:
    """(start_beats, dur_beats, midi)：每小节根音、三音、五音、三音四个四分音符（根音在 C4–B4），末小节全音符根音。"""
    out = []
    for i, label in enumerate(PROGRESSION):
        r, third, fifth = _triad(label, 60)
        if i == len(PROGRESSION) - 1:
            out.append((i * 4.0, 4.0, r))
        else:
            out += [(i * 4.0 + k, 1.0, p) for k, p in enumerate([r, third, fifth, third])]
    return out


def truth_webchart() -> str:
    rows = [PROGRESSION[i:i + 4] for i in range(0, len(PROGRESSION), 4)]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n"


def _part(program: int, is_drum: bool = False) -> tuple[pretty_midi.PrettyMIDI, pretty_midi.Instrument]:
    pm = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
    inst = pretty_midi.Instrument(program=program, is_drum=is_drum)
    pm.instruments.append(inst)
    return pm, inst


def build_parts() -> dict[str, pretty_midi.PrettyMIDI]:
    parts = {}
    pm, drums = _part(0, is_drum=True)
    for bar in range(1, len(PROGRESSION) + 1):
        for beat in range(4):
            t = bar_start(bar) + beat * SPB
            drums.notes.append(pretty_midi.Note(110 if beat == 0 else 95, 36 if beat % 2 == 0 else 38, t, t + 0.1))
            for half in (0.0, 0.5):
                th = t + half * SPB
                drums.notes.append(pretty_midi.Note(70, 42, th, th + 0.05))
    parts["drums"] = pm
    pm, bass = _part(33)
    for i, label in enumerate(PROGRESSION):
        for beat in range(4):
            t = bar_start(i + 1) + beat * SPB
            bass.notes.append(pretty_midi.Note(60, _bass(label), t, t + SPB * 0.9))
    parts["bass"] = pm
    pm, guitar = _part(25)
    for i, label in enumerate(PROGRESSION):
        for beat in (0, 2):
            t = bar_start(i + 1) + beat * SPB
            for p in _triad(label, 55):
                guitar.notes.append(pretty_midi.Note(105, p, t, t + 2 * SPB * 0.95))
    parts["guitar"] = pm
    pm, lead = _part(73)
    for s, d, p in melody_truth():
        t = OFFSET + s * SPB
        lead.notes.append(pretty_midi.Note(100, p, t, t + d * SPB * 0.9))
    parts["vocals"] = pm
    return parts


def render_song(out_dir: Path) -> dict[str, Path]:
    """渲染到 out_dir/stems/{vocals,drums,bass,guitar,piano,other}.wav 与 out_dir/mix.wav。"""
    stems = out_dir / "stems"
    stems.mkdir(parents=True, exist_ok=True)
    audio, sr = {}, 44100
    for name, pm in build_parts().items():
        mid = out_dir / f"{name}.mid"
        pm.write(str(mid))
        y, sr = sf.read(str(midi_to_wav(mid, out_dir / f"{name}.raw.wav")), always_2d=True)
        audio[name] = y
    n = max(len(y) for y in audio.values())
    padded = {k: np.pad(y, ((0, n - len(y)), (0, 0))) for k, y in audio.items()}
    padded["piano"] = np.zeros((n, 2))
    padded["other"] = np.zeros((n, 2))
    for name, y in padded.items():
        sf.write(str(stems / f"{name}.wav"), y, sr)
    mix = sum(padded[k] for k in ("drums", "bass", "guitar", "vocals"))
    sf.write(str(out_dir / "mix.wav"), mix / max(1.0, float(np.abs(mix).max())), sr)
    return {"mix": out_dir / "mix.wav", "stems": stems}
