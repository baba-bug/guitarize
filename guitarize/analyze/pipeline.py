"""分析编排：stems → beats → chords → melody → leadsheet，每步结果写入 analysis/。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from ..config import SongConfig, melody_source
from ..paths import SongPaths
from . import features
from .beats import beats_signature, track_beats
from .chords import reference_chords
from .key import chosen_key, cross_check_key, estimate_key, key_report, parse_key, scale_pcs
from .leadsheet import build_leadsheet
from .melody import extract_melody
from .stems import extract_lead, separate
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


def _melody_source(paths: SongPaths, overrides: dict) -> tuple[str, Path]:
    """旋律识别用的音频 (分轨名, 路径)：默认居中的主唱 lead.wav（没有就从人声分轨抽出来）；
    overrides.melody_source: vocals 用整条人声。分轨名记进 melody.json 的 source（verify 据此选对照的分轨）。"""
    source = melody_source(overrides.get("melody_source"))
    if source == "lead" and not paths.stem("lead").exists() and paths.stem("vocals").exists():
        extract_lead(paths.stem("vocals"), paths.stem("lead"))
    return source, paths.stem(source)


def run_analysis(paths: SongPaths, cfg: SongConfig, steps: Sequence[str] = STEPS) -> None:
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown analysis steps {sorted(unknown)}; choose from {STEPS}")
    overrides = cfg.overrides
    slug = paths.slug
    if "stems" in steps:
        separate(paths.original_wav, paths.stems_dir)
        extract_lead(paths.stem("vocals"), paths.stem("lead"))
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
        # 色度调与和弦交叉确认：没有 overrides.key 时，和弦推断的调明显更合适就改用（旋律与 leadsheet）
        kc = cross_check_key((tonic, mode), chords["segments"])
        kc["used"] = kc["used"] and not overrides.get("key")
        chords["key_from_chords"] = kc
        chords["beats_signature"] = beats_signature(beats)
        print(f"[key] {key_report(chords, overridden=bool(overrides.get('key')))}")
        _write(paths.chords_json, chords)
    if "melody" in steps or "leadsheet" in steps:
        beats = beats or _need(paths.beats_json, slug, "beats")
        chords = chords or _need(paths.chords_json, slug, "chords")
        key = chosen_key(chords)
    if "melody" in steps:
        source, wav = _melody_source(paths, overrides)
        melody = extract_melody(wav, beats, scale_pcs(*key), f0_cache=paths.vocal_f0)
        melody["source"] = source   # 从哪条分轨识别：旋律谱校验对照同一条
        melody["beats_signature"] = beats_signature(beats)
        melody["key"] = {"tonic": key[0], "mode": key[1]}  # 量化时用的调；leadsheet 用另一个调时据此报错
        _write(paths.melody_json, melody)
    if "leadsheet" in steps:
        melody = _need(paths.melody_json, slug, "melody")
        _write(paths.leadsheet_json, build_leadsheet(beats, chords, melody, key, cfg.sections, overrides.get("chords"),
                                                     overrides.get("melody")))
