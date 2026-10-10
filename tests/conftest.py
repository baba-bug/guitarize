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


@pytest.fixture(autouse=True)
def _no_karaoke_model(monkeypatch):
    """测试里不跑真的伴唱分离模型（1.7 GB、几分钟）：模型目录指向不存在的地方，主唱走 L/R 掩码；
    要测模型那条路的用例自己设 GUITARIZE_KARAOKE_DIR 并注入假的 Separator。"""
    monkeypatch.setenv("GUITARIZE_KARAOKE_DIR", str(FIXTURES / "no-karaoke-model"))


@pytest.fixture(autouse=True)
def _no_game(monkeypatch):
    """测试里不跑真的 GAME（唱歌 → 音符模型）：GAME 目录指向不存在的地方，melody 步骤退回 crepe 逐格量化；
    要测 GAME 那条路的用例自己设 GUITARIZE_GAME_DIR / 传 root，并替换 subprocess.run 或 game_notes。"""
    monkeypatch.setenv("GUITARIZE_GAME_DIR", str(FIXTURES / "no-game"))


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


def capo4_bar12_leadsheet() -> dict:
    """真实歌曲 capo 4 第 12 小节的样子：实际 D#/G（capo 4 下指法调 B/D#：空弦 B 填充 + 4 弦 1 品低音），
    第 2 拍的旋律 G4 弹在 2 弦（填充的弦），反拍 G#4 是空弦 E 延续过第 3 拍，第 4 拍 C#5。"""
    mel = [(1.0, 2 / 3, 67), (5 / 3, 4 / 3, 68), (3.0, 4 / 3, 73)]
    return {"key": "B", "mode": "major", "tempo_bpm": 167.0, "feel": "shuffle", "beats_per_bar": 4,
            "bars": [{"bar": 1, "section": "サビ", "t_start": 0.0, "t_end": 1.44,
                      "chords": [{"beat": 1, "dur_beats": 4, "label": "D#/G", "confidence": "high"}],
                      "melody": [{"start_beats": s, "dur_beats": d, "midi": m, "conf": 1.0} for s, d, m in mel]}]}
