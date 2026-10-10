"""每首歌的文件布局。所有阶段只从这里取路径，避免各处硬编码。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SongPaths:
    root: Path

    @classmethod
    def for_slug(cls, slug: str, songs_dir: Path | None = None) -> "SongPaths":
        return cls((songs_dir or REPO_ROOT / "songs") / slug)

    @property
    def slug(self) -> str:
        return self.root.name

    @property
    def config(self) -> Path:
        return self.root / "song.yaml"

    @property
    def feedback(self) -> Path:
        return self.root / "feedback.md"

    @property
    def source_dir(self) -> Path:
        return self.root / "source"

    @property
    def original_wav(self) -> Path:
        return self.source_dir / "original.wav"

    @property
    def analysis_dir(self) -> Path:
        return self.root / "analysis"

    @property
    def stems_dir(self) -> Path:
        return self.analysis_dir / "stems"

    def stem(self, name: str) -> Path:
        return self.stems_dir / f"{name}.wav"

    @property
    def lead_json(self) -> Path:
        """主唱 lead.wav 是怎么分出来的（analyze 的 lead 步骤写；melody.json 的 lead_method 取自这里）。"""
        return self.stems_dir / "lead.json"

    @property
    def beats_json(self) -> Path:
        return self.analysis_dir / "beats.json"

    @property
    def chords_json(self) -> Path:
        return self.analysis_dir / "chords.json"

    @property
    def melody_json(self) -> Path:
        return self.analysis_dir / "melody.json"

    @property
    def vocal_f0(self) -> Path:
        """人声基频缓存（analyze/melody.py 的 cached_vocal_pitch）：melody 步骤写，旋律谱校验的音高图读。"""
        return self.analysis_dir / "vocal_f0.npz"

    @property
    def game_notes(self) -> Path:
        """GAME 分音的缓存（analyze/game.py 的 game_notes）：melody 步骤（旋律 v7）写和读。"""
        return self.analysis_dir / "game_notes.csv"

    @property
    def leadsheet_json(self) -> Path:
        return self.analysis_dir / "leadsheet.json"

    @property
    def webchart_txt(self) -> Path:
        return self.analysis_dir / "webchart.txt"

    @property
    def arrangements_dir(self) -> Path:
        return self.root / "arrangements"

    def atex(self, arr: str) -> Path:
        return self.arrangements_dir / f"{arr}.atex"

    @property
    def build_dir(self) -> Path:
        return self.root / "build"

    def build(self, arr: str, ext: str) -> Path:
        return self.build_dir / f"{arr}.{ext}"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    def report_root(self, arr: str) -> Path:
        return self.reports_dir / arr

    @property
    def out_dir(self) -> Path:
        return self.root / "out"

    def out(self, arr: str, ext: str) -> Path:
        return self.out_dir / f"{arr}.{ext}"
