"""旋律对照原唱（analyze/melody_audit.py）：合成的主唱逐帧特征 + 手写的旋律，指标逐个算得出来。"""
import json

import numpy as np
import pytest

from guitarize.analyze import melody_audit
from guitarize.analyze.notes import signal_frames

FPS = 100


def _beats(n_bars=6, spb=0.5):
    beats = [{"t": k * spb, "bar": k // 4 + 1, "beat": k % 4 + 1, "g": k} for k in range(n_bars * 4)]
    bars = [{"bar": b + 1, "t_start": b * 4 * spb, "t_end": (b + 1) * 4 * spb, "beats": 4} for b in range(n_bars)]
    return {"tempo_bpm": 60 / spb, "feel": "shuffle", "time_signature": [4, 4], "pickup_beats": 0, "pad_beats": 0,
            "short_bars": [], "beats": beats, "bars": bars}


def _frames():
    """1.03–2.0 秒唱 E4、2.0–2.5 秒滑到 F#4（没有换字）；2.5–3.0 秒是伴唱（伴唱轨响 10 dB）；3.0–3.5 秒又唱 E4。"""
    n = 400
    midi, per, lead_db, back_db = np.full(n, np.nan), np.zeros(n), np.full(n, -80.0), np.full(n, -80.0)
    midi[103:200], midi[200:250], midi[250:350] = 64.0, 66.0, 64.0
    per[103:350], lead_db[103:350], back_db[250:300] = 0.9, 0.0, 10.0
    return signal_frames(midi, per, 10 ** (lead_db / 20), 10 ** (back_db / 20), np.ones(n))


def _note(start, dur, midi, **kw):
    return {"start_beats": start, "dur_beats": dur, "midi": midi, "out_of_key": False, "conf": 0.9, **kw}


NOTES = [_note(2.0, 1.0, 64),     # 起点比换字点早 30 ms
         _note(3.0, 1.0, 64),     # 同音、中间没有换字，又拨一下：切碎的长音
         _note(4.0, 1.0, 62),     # 唱的是 F#4：错音
         _note(5.0, 1.0, 64)]     # 伴唱那里：伴唱音


def test_audit_notes_against_the_lead():
    fr = _frames()
    assert list(np.flatnonzero(fr.onset & fr.voiced)) == [103, 300]
    got = melody_audit.audit_notes(NOTES, fr, _beats())
    assert got == {"notes": 4, "legato": 0, "coverage": round(147 / 197, 3), "pitch_ok": 0.667, "wrong": 1,
                   "backing_or_empty": 1, "chopped_held": 1, "onset_recall": 0.5, "pluck_ms": 30}
    held = [*NOTES[:1], _note(3.0, 1.0, 64, legato=True), *NOTES[2:]]
    assert melody_audit.audit_notes(held, fr, _beats())["chopped_held"] == 0   # legato（不重新拨）不算切碎
    assert melody_audit.audit_notes(held, fr, _beats())["legato"] == 1


def test_repeat_consistency_compares_sections_with_the_same_name():
    # 两遍サビ：第 2 小节 vs 第 4–6 小节，按短的那段（1 小节）比；±1 格（三连音）、同音高才算对上
    sections = {1: "A", 2: "サビ", 3: "B", 4: "サビ"}
    notes = [_note(4.0, 1, 64), _note(5.0, 1, 66), _note(6.0, 1, 68),
             _note(12.0, 1, 64), _note(13.3333, 1, 66), _note(14.0, 1, 69), _note(16.0, 1, 70)]
    got = melody_audit.repeat_consistency(notes, sections, _beats())
    assert got == {"f1": 0.667, "pairs": [{"section": "サビ", "bars": [[2, 2], [4, 4]], "f1": 0.667}]}
    assert melody_audit.repeat_consistency(notes, {1: "A", 3: "B"}, _beats()) == {"f1": None, "pairs": []}


def _song(tmp_path):
    from guitarize.paths import SongPaths

    paths = SongPaths.for_slug("demo", tmp_path / "songs")
    paths.stems_dir.mkdir(parents=True)
    paths.config.write_text("slug: demo\ntitle: T\nartist: A\nsections: {1: A, 2: サビ, 3: B, 4: サビ}\n",
                            encoding="utf-8")
    paths.beats_json.write_text(json.dumps(_beats()), encoding="utf-8")
    paths.melody_json.write_text(json.dumps({"method": "v7:game+signal", "notes": NOTES}), encoding="utf-8")
    paths.stem("lead_raw").write_bytes(b"RIFF")
    (tmp_path / "backing_elsewhere.wav").write_bytes(b"RIFF")
    paths.stem("backing").symlink_to(tmp_path / "backing_elsewhere.wav")   # 链到别处的分轨也按歌目录写
    return paths


def test_audit_melody_cli_prints_a_row_per_file_and_writes_the_report(tmp_path, monkeypatch, capsys):
    from guitarize.cli import main

    paths = _song(tmp_path)
    other = tmp_path / "v6.json"
    other.write_text(json.dumps({"notes": NOTES[:2]}), encoding="utf-8")
    calls = []
    monkeypatch.setattr(melody_audit, "frame_features", lambda *a: calls.append(a) or _frames())
    assert main(["--songs-dir", str(tmp_path / "songs"), "audit-melody", "demo", "--melody", str(other)]) == 0
    assert calls == [(paths.stem("lead_raw"), paths.stem("backing"), paths.vocal_f0)]   # 参考：这首歌自己的主唱
    out = capsys.readouterr().out.splitlines()
    assert out[0].split()[:3] == ["melody", "notes", "legato"] and "repeat_f1" in out[0]
    assert out[1].split()[:3] == ["analysis/melody.json", "4", "0"] and out[2].split()[:3] == [str(other), "2", "0"]
    report = json.loads((paths.reports_dir / "melody_audit.json").read_text(encoding="utf-8"))
    first, second = report["rows"]
    assert first["melody"] == "analysis/melody.json" and first["method"] == "v7:game+signal"
    assert first["pitch_ok"] == 0.667 and first["repeats"] == [{"section": "サビ", "bars": [[2, 2], [4, 4]], "f1": 0.0}]
    assert second["melody"] == str(other) and second["notes"] == 2 and second["method"] is None
    assert report["reference"] == {"lead_raw": "analysis/stems/lead_raw.wav", "backing": "analysis/stems/backing.wav"}


def test_audit_melody_needs_the_raw_lead_and_the_backing(tmp_path):
    from guitarize.config import load_song_config

    paths = _song(tmp_path)
    paths.stem("lead_raw").unlink()
    with pytest.raises(ValueError, match=r"lead_raw\.wav.*--steps lead"):
        melody_audit.audit_melody(paths, load_song_config(paths))
