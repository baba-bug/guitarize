"""song.yaml 与难度档的加载。"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from .paths import SongPaths

PROFILES_PATH = Path(__file__).with_name("profiles.yaml")
ARR_NAMES = {"strum": "弹唱", "finger": "指弹"}


@dataclass(frozen=True)
class Profile:
    name: str
    max_span: int
    max_fret: int
    max_fingers: int
    max_plucks: int
    max_shift: int
    allow_barre: bool
    melody_max_per_beat: int
    bass_on_beat3: bool


@dataclass(frozen=True)
class Thresholds:
    chord_strum: float = 0.90
    chord_finger: float = 0.85
    melody_f1: float = 0.80
    bar_chord_flag: float = 0.75
    bar_melody_flag: float = 0.60
    bar_harmonic_flag: float = 0.60


@dataclass(frozen=True)
class SongConfig:
    slug: str
    title: str
    artist: str
    version: str = "tv"
    level: str = "intermediate"
    arrangements: tuple[str, ...] = ("strum", "finger")
    outputs: tuple[str, ...] = ("html", "pdf", "txt")
    source: dict = field(default_factory=dict)
    reference_charts: tuple[str, ...] = ()
    overrides: dict = field(default_factory=dict)
    sections: dict = field(default_factory=dict)  # {小节号(1-based): 段落名}
    thresholds: Thresholds = Thresholds()


def load_profile(name: str) -> Profile:
    data = yaml.safe_load(PROFILES_PATH.read_text(encoding="utf-8"))
    if name not in data:
        raise KeyError(f"unknown level {name!r}; choose from {sorted(data)}")
    return Profile(name=name, **data[name])


def load_song_config(paths: SongPaths) -> SongConfig:
    raw = yaml.safe_load(paths.config.read_text(encoding="utf-8")) or {}
    thresholds = Thresholds(**(raw.pop("thresholds", None) or {}))
    sections = {int(k): str(v) for k, v in (raw.pop("sections", None) or {}).items()}
    for key in ("arrangements", "outputs", "reference_charts"):
        if raw.get(key) is not None:
            raw[key] = tuple(raw[key])
    raw = {k: v for k, v in raw.items() if v is not None}
    unknown = set(raw) - {f.name for f in fields(SongConfig)}
    if unknown:
        raise ValueError(f"unknown keys in {paths.config}: {sorted(unknown)}")
    return SongConfig(**raw, sections=sections, thresholds=thresholds)
