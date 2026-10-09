"""分析编排：stems → beats → chords → melody → leadsheet，每步结果写入 analysis/。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from ..config import SongConfig
from ..paths import SongPaths
from . import features
from .beats import track_beats
from .chords import reference_chords
from .key import estimate_key, parse_key, scale_pcs
from .leadsheet import build_leadsheet
from .melody import extract_melody
from .stems import separate
from .webchart import parse_webchart

STEPS = ("stems", "beats", "chords", "melody", "leadsheet")


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _need(path: Path, slug: str, step: str) -> dict:
    if not path.exists():
        raise ValueError(f"missing {path.name}; run `guitarize analyze {slug} --steps {step}` first")
    return _read(path)


def run_analysis(paths: SongPaths, cfg: SongConfig, steps: Sequence[str] = STEPS) -> None:
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown analysis steps {sorted(unknown)}; choose from {STEPS}")
    overrides = cfg.overrides
    slug = paths.slug
    if "stems" in steps:
        separate(paths.original_wav, paths.stems_dir)
    if "beats" in steps:
        _write(paths.beats_json, track_beats(paths.original_wav, paths.stems_dir, overrides))
    beats = chords = None
    if "chords" in steps:
        beats = _need(paths.beats_json, slug, "beats")
        times = [b["t"] for b in beats["beats"]]
        cb = features.beat_sync_chroma(features.load_mix([paths.stem("harmonic")]), times)
        if overrides.get("key"):
            tonic, mode = parse_key(str(overrides["key"]))
            r = None
        else:
            tonic, mode, r = estimate_key(cb.mean(axis=1))
        web = parse_webchart(paths.webchart_txt.read_text(encoding="utf-8")) if paths.webchart_txt.exists() else None
        chords = reference_chords(cb, beats, web, key_pcs=scale_pcs(tonic, mode))
        chords["key"] = {"tonic": tonic, "mode": mode, "r": r}
        _write(paths.chords_json, chords)
    if "melody" in steps or "leadsheet" in steps:
        beats = beats or _need(paths.beats_json, slug, "beats")
        chords = chords or _need(paths.chords_json, slug, "chords")
        key = (chords["key"]["tonic"], chords["key"]["mode"])
    if "melody" in steps:
        _write(paths.melody_json, extract_melody(paths.stem("vocals"), beats, scale_pcs(*key)))
    if "leadsheet" in steps:
        melody = _need(paths.melody_json, slug, "melody")
        _write(paths.leadsheet_json, build_leadsheet(beats, chords, melody, key, cfg.sections))
