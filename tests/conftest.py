import json
import shutil
from pathlib import Path

import pytest

from guitarize.paths import SongPaths
from guitarize.render import alphatab
from guitarize.verify.model import load_score

FIXTURES = Path(__file__).parent / "fixtures"
TEX_HEAD = '\\title "T"\n\\tempo 120\n.\n\\track "G"\n\\staff {tabs}\n\\tuning E4 B3 G3 D3 A2 E2\n'

# 《小星星》C 调：小节号 → [(和弦, 起始拍(1-based), 拍数)]
DEMO_CHORDS = {
    1: [("C", 1, 4)],
    2: [("F", 1, 2), ("C", 3, 2)],
    3: [("F", 1, 2), ("C", 3, 2)],
    4: [("G", 1, 2), ("C", 3, 2)],
    5: [("C", 1, 2), ("F", 3, 2)],
    6: [("C", 1, 2), ("G", 3, 2)],
    7: [("C", 1, 2), ("F", 3, 2)],
    8: [("C", 1, 2), ("G", 3, 2)],
}
# 小节号 → [(小节内偏移拍, 时值拍, midi)]
_A = [(0, 1, 67), (1, 1, 67), (2, 1, 65), (3, 1, 65)]
_B = [(0, 1, 64), (1, 1, 64), (2, 2, 62)]
DEMO_MELODY = {
    1: [(0, 1, 60), (1, 1, 60), (2, 1, 67), (3, 1, 67)],
    2: [(0, 1, 69), (1, 1, 69), (2, 2, 67)],
    3: [(0, 1, 65), (1, 1, 65), (2, 1, 64), (3, 1, 64)],
    4: [(0, 1, 62), (1, 1, 62), (2, 2, 60)],
    5: _A, 6: _B, 7: _A, 8: _B,
}


def demo_leadsheet(sec_per_beat: float = 0.6) -> dict:
    bars = []
    for b in range(1, 9):
        bars.append({
            "bar": b,
            "section": None,
            "t_start": (b - 1) * 4 * sec_per_beat,
            "t_end": b * 4 * sec_per_beat,
            "chords": [{"beat": s, "dur_beats": d, "label": lbl, "confidence": "high"} for lbl, s, d in DEMO_CHORDS[b]],
            "melody": [{"start_beats": (b - 1) * 4 + o, "dur_beats": d, "midi": m, "conf": 1.0} for o, d, m in DEMO_MELODY[b]],
        })
    return {"key": "C", "mode": "major", "tempo_bpm": 100.0, "feel": "straight", "beats_per_bar": 4, "bars": bars}


@pytest.fixture
def demo_song(tmp_path: Path) -> SongPaths:
    songs = tmp_path / "songs"
    shutil.copytree(FIXTURES / "demo", songs / "demo")
    paths = SongPaths.for_slug("demo", songs)
    paths.analysis_dir.mkdir(parents=True, exist_ok=True)
    paths.leadsheet_json.write_text(json.dumps(demo_leadsheet()), encoding="utf-8")
    return paths


@pytest.fixture
def build_tex(tmp_path: Path):
    counter = iter(range(10_000))

    def _build(body: str, head: str = TEX_HEAD):
        name = f"t{next(counter)}"
        src = tmp_path / f"{name}.atex"
        src.write_text(head + body, encoding="utf-8")
        return load_score(alphatab.build(src, tmp_path / "build", name)["json"])

    return _build
