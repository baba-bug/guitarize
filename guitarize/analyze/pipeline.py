"""分析编排：stems → lead → beats → chords → melody → leadsheet，每步结果写入 analysis/。
melody 步骤默认用旋律 v7（GAME + 主唱信号，notes.py），条件不够时退回 crepe 逐格量化（melody.py）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

from ..config import SongConfig, melody_source
from ..paths import SongPaths
from . import features
from .beats import beats_signature, track_beats
from .chords import reference_chords
from .game import game_notes
from .key import chosen_key, cross_check_key, estimate_key, key_report, parse_key, scale_pcs
from .leadsheet import build_leadsheet
from .melody import CREPE_METHOD, extract_melody
from .notes import V7_METHOD, bar_chord_labels, extract_melody_v7
from .stems import KARAOKE_METHOD, KARAOKE_MODEL, separate, separate_lead
from .webchart import parse_webchart

# lead 紧跟 stems 单独一步：换模型 / 重做主唱时不必重跑 Demucs
STEPS = ("stems", "lead", "beats", "chords", "melody", "leadsheet")


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _need(path: Path, slug: str, step: str) -> dict:
    if not path.exists():
        raise ValueError(f"missing {path.name}; run `guitarize analyze {slug} --steps {step}` first")
    return _read(path)


def _separate_lead(paths: SongPaths) -> str:
    """人声分轨 → 主唱 lead.wav + 伴唱 backing.wav + 模型的主唱原样 lead_raw.wav（stems.separate_lead）；
    用的方法记进 stems/lead.json。"""
    method = separate_lead(paths.stem("vocals"), paths.stem("lead"), paths.stem("backing"), paths.stem("lead_raw"))
    _write(paths.lead_json, {"method": method, "model": KARAOKE_MODEL if method == KARAOKE_METHOD else None})
    return method


def _melody_source(paths: SongPaths, overrides: dict) -> tuple[str, Path]:
    """旋律识别用的音频 (分轨名, 路径)：默认主唱 lead.wav（没有就从人声分轨分出来）；
    overrides.melody_source: vocals 用整条人声。分轨名记进 melody.json 的 source（verify 据此选对照的分轨）。"""
    source = melody_source(overrides.get("melody_source"))
    if source == "lead" and not paths.stem("lead").exists() and paths.stem("vocals").exists():
        _separate_lead(paths)
    return source, paths.stem(source)


def _extract_melody(paths: SongPaths, cfg: SongConfig, source: str, wav: Path, beats: dict, chords: dict,
                    key: tuple[str, str]) -> dict:
    """主唱（source = lead）且模型主唱原样 lead_raw.wav、伴唱 backing.wav、GAME 都在 → 旋律 v7（notes.py：GAME 分音 +
    主唱信号检查，和弦与 leadsheet 一样取 chords.json + overrides.chords）；否则 crepe 逐格量化（melody.extract_melody），
    主唱时打一行警告。用的方法记进 melody.json 的 method。"""
    key_pcs = scale_pcs(*key)
    if source == "lead":
        raw, backing = paths.stem("lead_raw"), paths.stem("backing")
        missing = [p.name for p in (raw, backing) if not p.exists()]
        rows = None if missing else game_notes(raw, paths.game_notes, cfg.language)
        if rows is not None:
            labels = bar_chord_labels(beats, chords, cfg.overrides.get("chords"))
            return {"method": V7_METHOD,
                    **extract_melody_v7(raw, backing, rows, beats, key_pcs, labels, f0_cache=paths.vocal_f0)}
        why = f"没有 {', '.join(missing)}，要用伴唱分离模型重跑 --steps lead" if missing else "GAME 不可用"
        print(f"[melody] warning: 旋律 v7 用不了（{why}），改用 crepe 逐格量化（{CREPE_METHOD}）", file=sys.stderr)
    return {"method": CREPE_METHOD, **extract_melody(wav, beats, key_pcs, f0_cache=paths.vocal_f0)}


def run_analysis(paths: SongPaths, cfg: SongConfig, steps: Sequence[str] = STEPS) -> None:
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown analysis steps {sorted(unknown)}; choose from {STEPS}")
    overrides = cfg.overrides
    slug = paths.slug
    if "stems" in steps:
        separate(paths.original_wav, paths.stems_dir)
        for name in ("lead", "backing", "lead_raw"):   # 从旧人声分出来的：作废
            paths.stem(name).unlink(missing_ok=True)
        paths.lead_json.unlink(missing_ok=True)
    if "lead" in steps:
        if not paths.stem("vocals").exists():
            raise ValueError(f"missing vocals.wav; run `guitarize analyze {slug} --steps stems` first")
        _separate_lead(paths)
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
        melody = _extract_melody(paths, cfg, source, wav, beats, chords, key)
        melody["source"] = source   # 从哪条分轨识别：旋律谱校验对照同一条
        if source == "lead" and paths.lead_json.exists():
            melody["lead_method"] = _read(paths.lead_json)["method"]   # 主唱是怎么分出来的（模型 / 只用 L/R 掩码）
        melody["beats_signature"] = beats_signature(beats)
        melody["key"] = {"tonic": key[0], "mode": key[1]}  # 量化时用的调；leadsheet 用另一个调时据此报错
        _write(paths.melody_json, melody)
    if "leadsheet" in steps:
        melody = _need(paths.melody_json, slug, "melody")
        _write(paths.leadsheet_json, build_leadsheet(beats, chords, melody, key, cfg.sections, overrides.get("chords"),
                                                     overrides.get("melody")))
