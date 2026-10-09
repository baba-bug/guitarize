# Guitarize Phase 1：渲染 + 校验核心 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给定一份 alphaTex 编配和一份 leadsheet.json 参考，能生成交付物（gp / txt / pdf / 交互网页），并输出逐小节的可弹性 + 差异评分报告（reports/<arr>/iter-NN）。

**Architecture:** alphaTex 是编配源文件；Node 端（alphaTab 1.8.4）解析成谱面模型 JSON、MIDI、Guitar Pro、SVG。Python 端读取模型 JSON 做可弹性检查和评分，用 FluidSynth 合成音频，用无头 Chrome 把 SVG 打印成 PDF。各模块之间只通过文件和本计划定义的数据类通信。

**Tech Stack:** Python 3.11（conda 环境 `guitarize`）、numpy、librosa、soundfile、matplotlib、pyyaml、pytest；Node 22 + `@coderline/alphatab@1.8.4`；FluidSynth 2.6；Google Chrome（无头模式）。

**Spec:** `docs/specs/2026-10-09-guitarize-design.md`

## 全局约束

- Python 解释器：`PY=~/anaconda3/envs/guitarize/bin/python`。所有测试用 `$PY -m pytest`。
- alphaTab 版本固定为 `1.8.4`（`tools/alphatab/package.json`）。
- 时间单位：**拍**（四分音符 = 1 拍，从第 1 小节第 1 拍 = 0 起算）和**秒**。alphaTab tick：960 / 四分音符。
- 弦号沿用 alphaTex 约定：**1 = 最高音弦（e），6 = 最低音弦（E）**。`tuning[0]` 是 1 弦。
- `midi` 字段是**含变调夹的实际音高**；`fret` 是相对变调夹的品位（谱上写的数字）。
- 小节号对外（报告、leadsheet）一律 **1-based**；`Score.bars[i].index` 是 0-based。
- 难度档 intermediate：`max_span = 3`（最高品 − 最低品，即四品格范围）、`max_fingers = 4`、`max_fret = 12`、`max_plucks = 4`、`max_shift = 4`（不到 1 拍内跳把超过 4 品即违规）。
- 达标线：弹唱和弦一致率 ≥ 0.90；指弹和弦一致率 ≥ 0.85、旋律 F1 ≥ 0.80；可弹性 0 违规。
- 原曲音频 / 分轨 / build 产物不进 git（`.gitignore` 已配置）。谱面不含歌词。
- 无头 Chrome 必须带独立的 `--user-data-dir`（临时目录），绝不触碰用户的 Chrome 配置。
- 报告目录：`songs/<slug>/reports/<arr>/iter-NN/`（spec 中的 `reports/iter-NN` 按编配类型再分一层）。

## 阶段总览

| 阶段 | 内容 | 计划 |
|---|---|---|
| **Phase 1** | 渲染 + 校验核心（本文件） | 本文件 |
| Phase 2 | 分析：分轨、节拍、调性、和弦三方投票、人声旋律、leadsheet；闭环测试 | Phase 1 完成后编写 |
| Phase 3 | 编配：和弦指型库、变调夹选择、指法 Viterbi、弹唱/指弹草稿生成器；获取（acquire）与 CLI 补全 | Phase 2 完成后编写 |
| Phase 4 | Skill + 首曲「なんもねえ」实跑与迭代 + GitHub 推送 | Phase 3 完成后编写 |
| Phase 5 | 复用验证（用户指定第二首歌） | 之后 |

## 文件结构（Phase 1）

```
pyproject.toml                    包定义 + pytest 配置
guitarize/
  __init__.py  __main__.py
  paths.py                        SongPaths：每首歌的全部路径
  config.py                       SongConfig / Profile / Thresholds / ARR_NAMES
  profiles.yaml                   三档难度参数
  theory.py                       音名、和弦解析、移调、大小三和弦推断
  skeleton.py                     旋律骨架规则（生成器与评分共用）
  cli.py                          render / verify / serve
  render/
    __init__.py                   （空）
    alphatab.py                   调用 Node CLI
    ascii_tab.py                  文本六线谱
    synth.py                      MIDI → wav（FluidSynth）
    pdf.py                        SVG → PDF（无头 Chrome）
    viewer.py                     交互网页
    pipeline.py                   render_all 编排
  verify/
    __init__.py                   （空）
    model.py                      谱面模型 JSON → Score
    playability.py                可弹性规则
    metrics.py                    逐小节评分
    report.py                     报告目录、片段、色度图、history.csv
tools/alphatab/cli.mjs            alphaTex → JSON / MIDI / GP / SVG
web/viewer.html                   交互谱模板
tests/
  conftest.py                     demo 歌曲夹具、build_tex 夹具
  fixtures/demo/{song.yaml, arrangements/strum.atex, arrangements/finger.atex}
  test_config.py test_theory.py test_alphatab.py test_model.py test_playability.py
  test_ascii.py test_render.py test_metrics.py test_verify_cli.py
```

---

### Task 1: 包骨架、路径、配置、难度档

**Files:**
- Create: `pyproject.toml`, `guitarize/__init__.py`, `guitarize/__main__.py`, `guitarize/paths.py`, `guitarize/config.py`, `guitarize/profiles.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `SongPaths`（属性见代码）、`REPO_ROOT`、`load_profile(name) -> Profile`、`load_song_config(paths) -> SongConfig`、`Thresholds`、`ARR_NAMES = {"strum": "弹唱", "finger": "指弹"}`

- [ ] **Step 1: 写 pyproject 并以可编辑模式安装**

`pyproject.toml`：
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "guitarize"
version = "0.1.0"
requires-python = ">=3.11"

[project.scripts]
guitarize = "guitarize.cli:main"

[tool.setuptools.packages.find]
include = ["guitarize*"]

[tool.setuptools.package-data]
guitarize = ["*.yaml"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: 需要模型或较长运行时间"]
```

`guitarize/__init__.py`：
```python
"""原曲音频 → 吉他谱（弹唱 / 指弹）+ 自动校验迭代。"""
__version__ = "0.1.0"
```

`guitarize/__main__.py`：
```python
from .cli import main

raise SystemExit(main())
```

Run: `$PY -m pip install -e /Users/s4146265/Desktop/Guitar`
Expected: `Successfully installed guitarize-0.1.0`

- [ ] **Step 2: 写失败测试 `tests/test_config.py`**

```python
from pathlib import Path

from guitarize.config import ARR_NAMES, load_profile, load_song_config
from guitarize.paths import SongPaths


def test_song_paths(tmp_path: Path):
    p = SongPaths.for_slug("demo", tmp_path)
    assert p.root == tmp_path / "demo"
    assert p.slug == "demo"
    assert p.atex("finger") == tmp_path / "demo" / "arrangements" / "finger.atex"
    assert p.stem("vocals") == tmp_path / "demo" / "analysis" / "stems" / "vocals.wav"
    assert p.build("strum", "mid") == tmp_path / "demo" / "build" / "strum.mid"
    assert p.report_root("finger") == tmp_path / "demo" / "reports" / "finger"
    assert p.out("finger", "pdf") == tmp_path / "demo" / "out" / "finger.pdf"


def test_load_profile_intermediate():
    p = load_profile("intermediate")
    assert (p.max_span, p.max_fingers, p.max_fret, p.max_plucks, p.max_shift) == (3, 4, 12, 4, 4)


def test_load_song_config(tmp_path: Path):
    paths = SongPaths(tmp_path / "demo")
    paths.root.mkdir()
    paths.config.write_text(
        "slug: demo\ntitle: T\nartist: A\nsections: {1: Intro, 5: サビ}\n"
        "thresholds: {melody_f1: 0.7}\nsource: {url: null}\n",
        encoding="utf-8",
    )
    cfg = load_song_config(paths)
    assert cfg.slug == "demo" and cfg.sections == {1: "Intro", 5: "サビ"}
    assert cfg.thresholds.melody_f1 == 0.7 and cfg.thresholds.chord_strum == 0.90
    assert cfg.arrangements == ("strum", "finger")
    assert ARR_NAMES["finger"] == "指弹"
```

- [ ] **Step 3: 运行，确认失败**

Run: `$PY -m pytest tests/test_config.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'guitarize.config'`）

- [ ] **Step 4: 实现 `guitarize/paths.py`**

```python
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
    def beats_json(self) -> Path:
        return self.analysis_dir / "beats.json"

    @property
    def chords_json(self) -> Path:
        return self.analysis_dir / "chords.json"

    @property
    def melody_json(self) -> Path:
        return self.analysis_dir / "melody.json"

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
```

- [ ] **Step 5: 实现 `guitarize/profiles.yaml` 与 `guitarize/config.py`**

`guitarize/profiles.yaml`：
```yaml
# 难度档：生成器与可弹性校验共用。max_span = 最高品 − 最低品（不含空弦）。
beginner:
  max_span: 3
  max_fret: 7
  max_fingers: 3
  max_plucks: 3
  max_shift: 3
  allow_barre: false
  melody_max_per_beat: 1
  bass_on_beat3: false
intermediate:
  max_span: 3
  max_fret: 12
  max_fingers: 4
  max_plucks: 4
  max_shift: 4
  allow_barre: true
  melody_max_per_beat: 2
  bass_on_beat3: true
advanced:
  max_span: 5
  max_fret: 15
  max_fingers: 4
  max_plucks: 5
  max_shift: 7
  allow_barre: true
  melody_max_per_beat: 3
  bass_on_beat3: true
```

`guitarize/config.py`：
```python
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
```

- [ ] **Step 6: 运行，确认通过**

Run: `$PY -m pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml guitarize tests/test_config.py
git commit -m "feat: package skeleton, song paths, config and difficulty profiles"
```

---

### Task 2: 乐理工具与旋律骨架规则

**Files:**
- Create: `guitarize/theory.py`, `guitarize/skeleton.py`
- Test: `tests/test_theory.py`

**Interfaces:**
- Produces:
  - `PITCH_CLASSES: list[str]`（升号写法）、`pc(name) -> int`、`pc_name(pc) -> str`、`midi_name(midi) -> str`（如 `"C4"`）
  - `parse_chord(label) -> tuple[int, str, int | None] | None`（根音 pc、规范化性质、低音 pc；`"N"` → None）
  - `majmin(label) -> str`：`"B:maj"` / `"G#:min"` / `"N"`
  - `transpose_label(label, semis) -> str`（保留性质后缀和斜杠低音）
  - `chord_tones(label) -> set[int]`
  - `MAJMIN_LABELS: list[str]`（24 个）、`infer_majmin(weights, bass_pc=None) -> set[str]`（并列最优全部返回）
  - `skeleton_mask(starts, durs) -> list[bool]`、`is_on_beat(start) -> bool`

- [ ] **Step 1: 写失败测试 `tests/test_theory.py`**

```python
import numpy as np

from guitarize import theory
from guitarize.skeleton import skeleton_mask


def test_pitch_names():
    assert theory.pc("C#") == 1 and theory.pc("Db") == 1
    assert theory.pc("Bb") == 10 and theory.pc("B") == 11 and theory.pc("E#") == 5
    assert theory.midi_name(60) == "C4" and theory.midi_name(71) == "B4"


def test_majmin():
    assert theory.majmin("B") == "B:maj"
    assert theory.majmin("G#m7") == "G#:min"
    assert theory.majmin("D#7") == "D#:maj"
    assert theory.majmin("Ebmaj7") == "D#:maj"
    assert theory.majmin("F#m7-5") == "F#:min"
    assert theory.majmin("E/G#") == "E:maj"
    assert theory.majmin("Asus4") == "A:maj"
    assert theory.majmin("N") == "N" and theory.majmin(None) == "N"


def test_transpose_label():
    assert theory.transpose_label("B", -4) == "G"
    assert theory.transpose_label("D#7", -4) == "B7"
    assert theory.transpose_label("G#m", -4) == "Em"
    assert theory.transpose_label("E/G#", -4) == "C/E"
    assert theory.transpose_label("N", 3) == "N"


def test_chord_tones():
    assert theory.chord_tones("G") == {7, 11, 2}
    assert theory.chord_tones("Am7") == {9, 0, 4, 7}
    assert theory.chord_tones("C/E") == {0, 4, 7}


def test_infer_majmin():
    w = np.zeros(12)
    w[[7, 11, 2]] = 1
    assert theory.infer_majmin(w, bass_pc=7) == {"G:maj"}
    w = np.zeros(12)
    w[[11, 6]] = 1  # 强力和弦：大小三并列
    assert theory.infer_majmin(w, bass_pc=11) == {"B:maj", "B:min"}
    assert theory.infer_majmin(np.zeros(12)) == {"N"}


def test_skeleton_mask():
    starts = [0, 5 / 3, 2, 7 / 3, 8 / 3, 5]
    durs = [1, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1]
    assert skeleton_mask(starts, durs) == [True, False, True, False, False, True]
    # 乐句开头（前面休止 ≥ 1 拍）即使在反拍也保留
    assert skeleton_mask([0, 8 / 3], [1, 1 / 3]) == [True, True]
    # 反拍长音保留
    assert skeleton_mask([0, 2 / 3], [2 / 3, 4 / 3]) == [True, True]
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest tests/test_theory.py -v`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 实现 `guitarize/theory.py`**

```python
"""音名、和弦标签、移调、大小三和弦推断。"""
from __future__ import annotations

import re

import numpy as np

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_ALIASES = {"DB": 1, "EB": 3, "GB": 6, "AB": 8, "BB": 10, "CB": 11, "FB": 4, "E#": 5, "B#": 0}
_CHORD_RE = re.compile(r"^([A-Ga-g])([#b♯♭]?)([^/]*)(?:/([A-Ga-g][#b♯♭]?))?$")
_NO_CHORD = {"", "N", "N.C.", "NC", "-"}

# (后缀正则, 规范化性质)，按顺序匹配
_QUALITIES = [
    (r"^(maj7|M7|Δ)", "maj7"),
    (r"^(m7-5|m7b5|ø)", "min7b5"),
    (r"^(dim|o)", "dim"),
    (r"^(aug|\+)", "aug"),
    (r"^(m7|min7|-7)", "min7"),
    (r"^(m|min|-)", "min"),
    (r"^sus2", "sus2"),
    (r"^sus", "sus4"),
    (r"^(7|9|11|13)", "7"),
    (r"", "maj"),
]
_INTERVALS = {
    "maj": (0, 4, 7), "min": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10), "dim": (0, 3, 6), "min7b5": (0, 3, 6, 10), "aug": (0, 4, 8),
    "sus4": (0, 5, 7), "sus2": (0, 2, 7),
}
_MAJ_LIKE = {"maj", "maj7", "7", "aug", "sus4", "sus2"}


def pc(name: str) -> int:
    n = name.strip().replace("♯", "#").replace("♭", "b")
    key = n[0].upper() + n[1:]
    if key in PITCH_CLASSES:
        return PITCH_CLASSES.index(key)
    if key.upper() in _ALIASES:
        return _ALIASES[key.upper()]
    raise ValueError(f"bad note name {name!r}")


def pc_name(p: int) -> str:
    return PITCH_CLASSES[p % 12]


def midi_name(m: int) -> str:
    return f"{pc_name(m)}{m // 12 - 1}"


def parse_chord(label: str) -> tuple[int, str, int | None] | None:
    s = label.strip()
    if s in _NO_CHORD:
        return None
    m = _CHORD_RE.match(s)
    if not m:
        raise ValueError(f"bad chord label {label!r}")
    root = pc(m.group(1) + m.group(2))
    quality = next(q for pat, q in _QUALITIES if re.match(pat, m.group(3)))
    bass = pc(m.group(4)) if m.group(4) else None
    return root, quality, bass


def majmin(label: str | None) -> str:
    parsed = None if label is None else parse_chord(label)
    if parsed is None:
        return "N"
    root, quality, _ = parsed
    return f"{pc_name(root)}:{'maj' if quality in _MAJ_LIKE else 'min'}"


def transpose_label(label: str, semis: int) -> str:
    s = label.strip()
    m = _CHORD_RE.match(s)
    if s in _NO_CHORD or not m:
        return label
    out = pc_name(pc(m.group(1) + m.group(2)) + semis) + m.group(3)
    if m.group(4):
        out += "/" + pc_name(pc(m.group(4)) + semis)
    return out


def chord_tones(label: str) -> set[int]:
    parsed = parse_chord(label)
    if parsed is None:
        return set()
    root, quality, bass = parsed
    tones = {(root + i) % 12 for i in _INTERVALS[quality]}
    if bass is not None:
        tones.add(bass)
    return tones


MAJMIN_LABELS = [f"{n}:{q}" for q in ("maj", "min") for n in PITCH_CLASSES]


def _template(label: str) -> np.ndarray:
    root = pc(label.split(":")[0])
    v = np.zeros(12)
    v[[(root + i) % 12 for i in ((0, 4, 7) if label.endswith("maj") else (0, 3, 7))]] = 1
    return v


TEMPLATES = np.stack([_template(lbl) for lbl in MAJMIN_LABELS])  # (24, 12)


def infer_majmin(weights, bass_pc: int | None = None, tol: float = 1e-6) -> set[str]:
    """按音名权重推断大小三和弦。得分 = 和弦音权重占比 + 低音为根音时 0.25；并列最优全部返回。"""
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    if total <= 0:
        return {"N"}
    scores = TEMPLATES @ w / total
    if bass_pc is not None:
        for i in range(len(MAJMIN_LABELS)):
            if i % 12 == bass_pc:
                scores[i] += 0.25
    best = scores.max()
    return {MAJMIN_LABELS[i] for i in np.flatnonzero(scores >= best - tol)}
```

- [ ] **Step 4: 实现 `guitarize/skeleton.py`**

```python
"""旋律骨架规则：指弹生成器和旋律评分共用，保证"保留哪些音"和"考核哪些音"一致。"""
from __future__ import annotations

from typing import Sequence

EPS = 1e-3


def is_on_beat(start: float) -> bool:
    frac = start % 1.0
    return frac < EPS or frac > 1 - EPS


def skeleton_mask(starts: Sequence[float], durs: Sequence[float]) -> list[bool]:
    """按开始时间排好序的音符 → 是否为骨架音。

    骨架音 = 落在拍点上，或时值 ≥ 2/3 拍，或前面休止 ≥ 1 拍（乐句开头）。
    """
    out = []
    prev_end = None
    for s, d in zip(starts, durs):
        after_rest = prev_end is None or s - prev_end >= 1 - EPS
        out.append(is_on_beat(s) or d >= 2 / 3 - EPS or after_rest)
        prev_end = s + d
    return out
```

- [ ] **Step 5: 运行，确认通过**

Run: `$PY -m pytest tests/test_theory.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add guitarize/theory.py guitarize/skeleton.py tests/test_theory.py
git commit -m "feat: music theory helpers and shared melody skeleton rule"
```

---

### Task 3: Node 构建器 + 谱面模型加载

**Files:**
- Create: `tools/alphatab/cli.mjs`, `guitarize/render/__init__.py`（空）, `guitarize/render/alphatab.py`, `guitarize/verify/__init__.py`（空）, `guitarize/verify/model.py`
- Create fixtures: `tests/fixtures/demo/song.yaml`, `tests/fixtures/demo/arrangements/finger.atex`, `tests/fixtures/demo/arrangements/strum.atex`, `tests/conftest.py`
- Test: `tests/test_alphatab.py`, `tests/test_model.py`

**Interfaces:**
- Consumes: `REPO_ROOT`、`SongPaths`（Task 1）
- Produces:
  - `alphatab.build(atex: Path, out_dir: Path, name: str, svg_width: int = 700) -> dict[str, Path]`，键为 `json / mid / gp / svg`；解析失败抛 `alphatab.AlphaTexError`（消息含 alphaTab 诊断）
  - 模型 JSON 结构：`{title, tempo, capo, tuning[6], bars:[{index, start, duration, ts, section, triplet_feel, voices:[[{start, duration, tuplet, rest, dots, chord, pick, brush, notes:[{string, fret, midi, tie, dead}]}]]}]}`（tick 单位）
  - `model.TPQ = 960`、`swing(beats, feel) -> float`
  - 数据类 `Note(string, fret, midi, tie, dead)`、`Event(bar, voice, start, dur, written_start, written_dur, notes, chord, pick, brush, rest)`、`Bar(index, start, length, section, feel)`、`Attack(start, bar, notes, strum, direction)`、`Sounding(start, end, midi, voice)`
  - `Score(title, tempo, capo, tuning, bars, events)`，方法 `seconds(beats)`、`attacks()`、`sounding()`、`melody(voice=0)`、`voice_lengths() -> dict[(bar, voice), float]`
  - `load_score(path) -> Score`
  - 夹具：`demo_song`（`SongPaths`，已写 leadsheet.json）、`build_tex(body, head=TEX_HEAD) -> Score`、`demo_leadsheet()`、`TEX_HEAD`

- [ ] **Step 1: 写 Node 构建器 `tools/alphatab/cli.mjs`**

```js
// alphaTex → 谱面模型 JSON / MIDI / Guitar Pro / SVG 片段（供 Python 端校验、合成与打印）
// 用法：node cli.mjs build <in.atex> <outDir> <name> [--svg-width 700]
// 成功时 stdout 最后一行是 {"json":..., "mid":..., "gp":..., "svg":...}
import * as alphaTab from '@coderline/alphatab';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

alphaTab.Logger.logLevel = alphaTab.LogLevel.None;

const PICK = { 0: null, 1: 'up', 2: 'down' };
const BRUSH = { 0: null, 1: 'up', 2: 'down', 3: 'up', 4: 'down' };
const FEEL = { 0: 'none', 1: 'triplet16th', 2: 'triplet8th', 3: 'dotted16th', 4: 'dotted8th', 5: 'scottish16th', 6: 'scottish8th' };

function toModel(score) {
  const staff = score.tracks[0].staves[0];
  const nStrings = staff.tuning.length;
  return {
    title: score.title,
    tempo: score.tempo,
    capo: staff.capo,
    tuning: staff.tuning,
    bars: staff.bars.map((bar) => {
      const mb = bar.masterBar;
      return {
        index: bar.index,
        start: mb.start,
        duration: mb.calculateDuration(),
        ts: [mb.timeSignatureNumerator, mb.timeSignatureDenominator],
        section: mb.section ? mb.section.text : null,
        triplet_feel: FEEL[mb.tripletFeel] ?? 'none',
        voices: bar.voices.map((voice) =>
          voice.beats
            .filter((b) => !b.isEmpty)
            .map((b) => ({
              start: b.displayStart,
              duration: b.displayDuration,
              tuplet: b.tupletNumerator > 0 ? b.tupletNumerator : 1,
              rest: b.isRest,
              dots: b.dots,
              chord: b.chord ? b.chord.name : null,
              pick: PICK[b.pickStroke] ?? null,
              brush: BRUSH[b.brushType] ?? null,
              notes: b.notes.map((n) => ({
                string: nStrings - n.string + 1,
                fret: n.fret,
                midi: n.realValue,
                tie: n.isTieDestination,
                dead: n.isDead,
              })),
            })),
        ),
      };
    }),
  };
}

function toMidi(score, settings) {
  const midiFile = new alphaTab.midi.MidiFile();
  const handler = new alphaTab.midi.AlphaSynthMidiFileHandler(midiFile, true);
  new alphaTab.midi.MidiFileGenerator(score, settings, handler).generate();
  return midiFile.toBinary();
}

function toSvg(score, width) {
  return new Promise((resolveSvg, reject) => {
    const settings = new alphaTab.Settings();
    settings.core.engine = 'svg';
    settings.core.enableLazyLoading = false;
    settings.core.useWorkers = false;
    const renderer = new alphaTab.rendering.ScoreRenderer(settings);
    renderer.width = width;
    const parts = [];
    renderer.partialRenderFinished.on((r) => parts.push(r.renderResult));
    renderer.renderFinished.on(() => resolveSvg(parts.filter((svg) => !svg.includes('rendered by alphaTab')).join('\n')));
    renderer.error.on(reject);
    renderer.renderScore(score, [0]);
  });
}

async function main() {
  const [cmd, inPath, outDir, name, ...rest] = process.argv.slice(2);
  if (cmd !== 'build' || !inPath || !outDir || !name) {
    console.error('usage: node cli.mjs build <in.atex> <outDir> <name> [--svg-width 700]');
    process.exit(64);
  }
  const wIdx = rest.indexOf('--svg-width');
  const svgWidth = wIdx >= 0 ? Number(rest[wIdx + 1]) : 700;
  const settings = new alphaTab.Settings();
  settings.core.logLevel = alphaTab.LogLevel.None;
  let score;
  try {
    score = alphaTab.importer.ScoreLoader.loadAlphaTex(readFileSync(inPath, 'utf8'), settings);
  } catch (e) {
    const inner = e.inner ?? e;
    const lines = [`alphaTex 解析失败: ${inPath}`, String(inner)];
    if (typeof inner.iterateDiagnostics === 'function') {
      for (const d of inner.iterateDiagnostics()) lines.push(`- ${d.message ?? JSON.stringify(d)}`);
    }
    console.error(lines.join('\n'));
    process.exit(2);
  }
  mkdirSync(outDir, { recursive: true });
  const out = {
    json: resolve(join(outDir, `${name}.json`)),
    mid: resolve(join(outDir, `${name}.mid`)),
    gp: resolve(join(outDir, `${name}.gp`)),
    svg: resolve(join(outDir, `${name}.svg.html`)),
  };
  writeFileSync(out.json, JSON.stringify(toModel(score)));
  writeFileSync(out.mid, toMidi(score, settings));
  writeFileSync(out.gp, new alphaTab.exporter.Gp7Exporter().export(score, settings));
  writeFileSync(out.svg, await toSvg(score, svgWidth));
  process.stdout.write('\n' + JSON.stringify(out) + '\n');
}

main().catch((e) => {
  console.error(String(e?.stack ?? e));
  process.exit(1);
});
```

- [ ] **Step 2: 写 demo 夹具**

`tests/fixtures/demo/song.yaml`：
```yaml
slug: demo
title: Demo Twinkle
artist: Traditional
version: tv
level: intermediate
```

`tests/fixtures/demo/arrangements/finger.atex`（《小星星》C 调，voice 1 旋律，voice 2 低音）：
```
\title "Demo Twinkle"
\tempo 100
.
\track "Guitar"
\instrument 25
\staff {score tabs}
\tuning E4 B3 G3 D3 A2 E2
\ts 4 4
\voice
:4 1.2 1.2 3.1 3.1 | 5.1 5.1 :2 3.1 | :4 1.1 1.1 0.1 0.1 | 3.2 3.2 :2 1.2 | :4 3.1 3.1 1.1 1.1 | 0.1 0.1 :2 3.2 | :4 3.1 3.1 1.1 1.1 | 0.1 0.1 :2 3.2 |
\voice
:1 3.5 | :2 3.4 3.5 | :2 1.6 3.5 | :2 3.6 3.5 | :2 3.5 1.6 | :2 3.5 3.6 | :2 3.5 1.6 | :2 3.5 3.6 |
```

`tests/fixtures/demo/arrangements/strum.atex`（每半小节：↓四分 ↓八分 ↑八分）：
```
\title "Demo Twinkle"
\tempo 100
.
\track "Guitar"
\instrument 25
\staff {tabs}
\tuning E4 B3 G3 D3 A2 E2
\chord "C" 0 1 0 2 3 x
\chord "F" 1 1 2 3 3 1
\chord "G" 3 0 0 0 2 3
\ts 4 4
:4 (0.1 1.2 0.3 2.4 3.5) {ch "C" sd} :8 (0.1 1.2 0.3 2.4 3.5) {sd} (0.1 1.2 0.3) {su} :4 (0.1 1.2 0.3 2.4 3.5) {sd} :8 (0.1 1.2 0.3 2.4 3.5) {sd} (0.1 1.2 0.3) {su} |
:4 (1.1 1.2 2.3 3.4 3.5 1.6) {ch "F" sd} :8 (1.1 1.2 2.3 3.4 3.5 1.6) {sd} (1.1 1.2 2.3) {su} :4 (0.1 1.2 0.3 2.4 3.5) {ch "C" sd} :8 (0.1 1.2 0.3 2.4 3.5) {sd} (0.1 1.2 0.3) {su} |
:4 (1.1 1.2 2.3 3.4 3.5 1.6) {ch "F" sd} :8 (1.1 1.2 2.3 3.4 3.5 1.6) {sd} (1.1 1.2 2.3) {su} :4 (0.1 1.2 0.3 2.4 3.5) {ch "C" sd} :8 (0.1 1.2 0.3 2.4 3.5) {sd} (0.1 1.2 0.3) {su} |
:4 (3.1 0.2 0.3 0.4 2.5 3.6) {ch "G" sd} :8 (3.1 0.2 0.3 0.4 2.5 3.6) {sd} (3.1 0.2 0.3) {su} :4 (0.1 1.2 0.3 2.4 3.5) {ch "C" sd} :8 (0.1 1.2 0.3 2.4 3.5) {sd} (0.1 1.2 0.3) {su} |
:4 (0.1 1.2 0.3 2.4 3.5) {ch "C" sd} :8 (0.1 1.2 0.3 2.4 3.5) {sd} (0.1 1.2 0.3) {su} :4 (1.1 1.2 2.3 3.4 3.5 1.6) {ch "F" sd} :8 (1.1 1.2 2.3 3.4 3.5 1.6) {sd} (1.1 1.2 2.3) {su} |
:4 (0.1 1.2 0.3 2.4 3.5) {ch "C" sd} :8 (0.1 1.2 0.3 2.4 3.5) {sd} (0.1 1.2 0.3) {su} :4 (3.1 0.2 0.3 0.4 2.5 3.6) {ch "G" sd} :8 (3.1 0.2 0.3 0.4 2.5 3.6) {sd} (3.1 0.2 0.3) {su} |
:4 (0.1 1.2 0.3 2.4 3.5) {ch "C" sd} :8 (0.1 1.2 0.3 2.4 3.5) {sd} (0.1 1.2 0.3) {su} :4 (1.1 1.2 2.3 3.4 3.5 1.6) {ch "F" sd} :8 (1.1 1.2 2.3 3.4 3.5 1.6) {sd} (1.1 1.2 2.3) {su} |
:4 (0.1 1.2 0.3 2.4 3.5) {ch "C" sd} :8 (0.1 1.2 0.3 2.4 3.5) {sd} (0.1 1.2 0.3) {su} :4 (3.1 0.2 0.3 0.4 2.5 3.6) {ch "G" sd} :8 (3.1 0.2 0.3 0.4 2.5 3.6) {sd} (3.1 0.2 0.3) {su} |
```

`tests/conftest.py`：
```python
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
```

- [ ] **Step 3: 写失败测试 `tests/test_alphatab.py` 与 `tests/test_model.py`**

`tests/test_alphatab.py`：
```python
import json

import pytest

from conftest import TEX_HEAD
from guitarize.render import alphatab


def test_build_outputs(demo_song, tmp_path):
    out = alphatab.build(demo_song.atex("finger"), tmp_path / "b", "finger")
    assert set(out) == {"json", "mid", "gp", "svg"}
    assert all(p.exists() and p.stat().st_size > 0 for p in out.values())
    model = json.loads(out["json"].read_text())
    assert model["tempo"] == 100 and len(model["bars"]) == 8
    assert model["tuning"][0] == 64 and model["capo"] == 0
    assert "rendered by alphaTab" not in out["svg"].read_text()


def test_build_error_reports_diagnostics(tmp_path):
    src = tmp_path / "bad.atex"
    src.write_text(TEX_HEAD + ":4 0.9 |", encoding="utf-8")
    with pytest.raises(alphatab.AlphaTexError, match="out of range"):
        alphatab.build(src, tmp_path / "b", "bad")
```

`tests/test_model.py`：
```python
import pytest

from conftest import TEX_HEAD
from guitarize.verify.model import load_score, swing


def test_swing_mapping():
    assert swing(0.5, "triplet8th") == pytest.approx(2 / 3)
    assert swing(1.25, "triplet8th") == pytest.approx(1 + 1 / 3)
    assert swing(0.75, "triplet8th") == pytest.approx(2 / 3 + 1 / 6)
    assert swing(0.5, "none") == 0.5


def test_demo_finger_model(demo_song, tmp_path):
    from guitarize.render import alphatab

    score = load_score(alphatab.build(demo_song.atex("finger"), tmp_path / "b", "f")["json"])
    assert score.tempo == 100 and len(score.bars) == 8
    assert score.bars[1].start == 4 and score.bars[1].length == 4
    mel = score.melody(voice=0)
    assert len(mel) == 28
    assert [m.midi for m in mel[:4]] == [60, 60, 67, 67]
    assert mel[6].start == 6 and mel[6].end == 8  # 第 2 小节的二分音符 G4
    assert score.seconds(4) == pytest.approx(2.4)


def test_triplet_feel_swings_offbeats(build_tex):
    score = build_tex("\\tf triplet-8th\n:8 0.1 0.1 0.1 0.1 :2 0.1 |")
    starts = [e.start for e in score.events]
    assert starts == pytest.approx([0, 2 / 3, 1, 5 / 3, 2])
    assert [e.written_start for e in score.events] == pytest.approx([0, 0.5, 1, 1.5, 2])


def test_explicit_triplets_not_swung(build_tex):
    score = build_tex("\\tf triplet-8th\n:8 0.1 {tu 3} 0.1 {tu 3} 0.1 {tu 3} :4 0.1 :2 0.1 |")
    assert [e.start for e in score.events][:3] == pytest.approx([0, 1 / 3, 2 / 3])


def test_ties_extend_sounding(build_tex):
    score = build_tex(":2 3.1 :4 -.1 r |")
    snd = score.sounding()
    assert len(snd) == 1 and snd[0].start == 0 and snd[0].end == 3
    assert len(score.attacks()) == 1


def test_capo_in_midi_and_attack_flags(build_tex):
    head = TEX_HEAD + "\\capo 4\n"
    score = build_tex(":4 (3.6 2.5) {sd} (0.1 0.2) {su} 0.3 r |", head=head)
    assert score.capo == 4
    attacks = score.attacks()
    assert [n.midi for n in attacks[0].notes] == [47, 51]
    assert attacks[0].strum and attacks[0].direction == "down"
    assert attacks[1].direction == "up"
    assert not attacks[2].strum and attacks[2].direction is None


def test_voice_lengths_detect_overfull_bar(build_tex):
    score = build_tex(":4 0.1 0.1 0.1 0.1 0.1 |")
    assert score.voice_lengths()[(0, 0)] == pytest.approx(5)
```

- [ ] **Step 4: 运行，确认失败**

Run: `cd /Users/s4146265/Desktop/Guitar && $PY -m pytest tests/test_alphatab.py tests/test_model.py -v`
Expected: FAIL（`ImportError: cannot import name 'alphatab'`）

- [ ] **Step 5: 实现 `guitarize/render/alphatab.py`（并建空的 `guitarize/render/__init__.py`、`guitarize/verify/__init__.py`）**

```python
"""调用 tools/alphatab/cli.mjs：alphaTex → 模型 JSON / MIDI / GP / SVG。"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..paths import REPO_ROOT

NODE_CLI = REPO_ROOT / "tools" / "alphatab" / "cli.mjs"


class AlphaTexError(RuntimeError):
    pass


def build(atex: Path, out_dir: Path, name: str, svg_width: int = 700) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["node", str(NODE_CLI), "build", str(atex), str(out_dir), name, "--svg-width", str(svg_width)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AlphaTexError(f"alphaTex build failed for {atex}:\n{proc.stderr.strip()}")
    last = proc.stdout.strip().splitlines()[-1]
    return {k: Path(v) for k, v in json.loads(last).items()}
```

- [ ] **Step 6: 实现 `guitarize/verify/model.py`**

```python
"""谱面模型：把 Node 端导出的 JSON 变成带演奏时间（已应用 swing）的事件序列。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

TPQ = 960  # alphaTab 每四分音符的 tick 数


@dataclass(frozen=True)
class Note:
    string: int  # 1 = 最高音弦
    fret: int    # 相对变调夹
    midi: int    # 含变调夹的实际音高
    tie: bool
    dead: bool


@dataclass(frozen=True)
class Event:
    """某声部里的一拍（alphaTab 的 Beat）。"""
    bar: int              # 0-based
    voice: int
    start: float          # 演奏时间（拍，已应用 swing）
    dur: float
    written_start: float  # 记谱位置（拍）
    written_dur: float
    notes: tuple[Note, ...]
    chord: str | None
    pick: str | None
    brush: str | None
    rest: bool


@dataclass(frozen=True)
class Bar:
    index: int     # 0-based
    start: float   # 拍
    length: float  # 拍
    section: str | None
    feel: str


@dataclass(frozen=True)
class Attack:
    """同一时刻新拨响的所有音（跨声部合并，不含连音线延续）。"""
    start: float
    bar: int
    notes: tuple[Note, ...]
    strum: bool               # 带扫弦 / 刷弦标记
    direction: str | None     # "down" / "up" / None


@dataclass(frozen=True)
class Sounding:
    start: float
    end: float
    midi: int
    voice: int


def swing(beats: float, feel: str) -> float:
    """三连音感觉：每拍前半（直八分）拉长到 2/3，后半压缩到 1/3。"""
    if feel != "triplet8th":
        return beats
    whole = int(beats // 1)
    frac = beats - whole
    if frac <= 0.5:
        return whole + frac * 4 / 3
    return whole + 2 / 3 + (frac - 0.5) * 2 / 3


@dataclass
class Score:
    title: str
    tempo: float
    capo: int
    tuning: list[int]
    bars: list[Bar]
    events: list[Event]

    def seconds(self, beats: float) -> float:
        return beats * 60.0 / self.tempo

    def attacks(self) -> list[Attack]:
        groups: dict[float, list[Event]] = {}
        for e in self.events:
            if not e.rest and any(not n.tie for n in e.notes):
                groups.setdefault(round(e.start, 6), []).append(e)
        out = []
        for start in sorted(groups):
            evs = groups[start]
            direction = next((e.pick or e.brush for e in evs if e.pick or e.brush), None)
            out.append(Attack(
                start=start,
                bar=evs[0].bar,
                notes=tuple(n for e in evs for n in e.notes if not n.tie),
                strum=direction is not None,
                direction=direction,
            ))
        return out

    def sounding(self) -> list[Sounding]:
        """每个音的实际发声区间；连音线把时值并入前一个同弦同声部的音。"""
        out: list[Sounding] = []
        open_notes: dict[tuple[int, int], int] = {}
        for e in sorted(self.events, key=lambda ev: (ev.start, ev.voice)):
            for n in e.notes:
                if n.dead:
                    continue
                key = (e.voice, n.string)
                if n.tie and key in open_notes:
                    i = open_notes[key]
                    s = out[i]
                    out[i] = Sounding(s.start, e.start + e.dur, s.midi, s.voice)
                else:
                    open_notes[key] = len(out)
                    out.append(Sounding(e.start, e.start + e.dur, n.midi, e.voice))
        return out

    def melody(self, voice: int = 0) -> list[Sounding]:
        """旋律：指定声部中每个起音时刻的最高音。"""
        best: dict[float, Sounding] = {}
        for s in self.sounding():
            if s.voice != voice:
                continue
            k = round(s.start, 6)
            if k not in best or s.midi > best[k].midi:
                best[k] = s
        return [best[k] for k in sorted(best)]

    def voice_lengths(self) -> dict[tuple[int, int], float]:
        """(小节, 声部) → 记谱总时值（拍），用于发现小节拍数写错。"""
        out: dict[tuple[int, int], float] = {}
        for e in self.events:
            out[(e.bar, e.voice)] = out.get((e.bar, e.voice), 0.0) + e.written_dur
        return out


def load_score(path: Path) -> Score:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    bars: list[Bar] = []
    events: list[Event] = []
    for b in data["bars"]:
        bar_start = b["start"] / TPQ
        feel = b["triplet_feel"]
        bars.append(Bar(b["index"], bar_start, b["duration"] / TPQ, b["section"], feel))
        for vi, voice in enumerate(b["voices"]):
            for bt in voice:
                ws = bar_start + bt["start"] / TPQ
                wd = bt["duration"] / TPQ
                if bt["tuplet"] == 1:
                    ps, pe = swing(ws, feel), swing(ws + wd, feel)
                else:
                    ps, pe = ws, ws + wd
                notes = tuple(Note(n["string"], n["fret"], n["midi"], n["tie"], n["dead"]) for n in bt["notes"])
                events.append(Event(b["index"], vi, ps, pe - ps, ws, wd, notes,
                                    bt["chord"], bt["pick"], bt["brush"], bt["rest"]))
    return Score(data["title"], float(data["tempo"]), int(data["capo"]), list(data["tuning"]), bars, events)
```

- [ ] **Step 7: 运行，确认通过**

Run: `cd /Users/s4146265/Desktop/Guitar && $PY -m pytest tests/test_alphatab.py tests/test_model.py -v`
Expected: 9 passed。若 `test_explicit_triplets_not_swung` 失败，先用 `node -e` 打印该 alphaTex 的 `tupletNumerator` 确认三连音语法，再修正测试中的 alphaTex（不要改 swing 规则）。

- [ ] **Step 8: Commit**

```bash
git add tools/alphatab/cli.mjs guitarize/render guitarize/verify tests/conftest.py tests/fixtures tests/test_alphatab.py tests/test_model.py
git commit -m "feat: alphaTab node builder and score model with swing timing"
```

---

### Task 4: 可弹性检查

**Files:**
- Create: `guitarize/verify/playability.py`
- Test: `tests/test_playability.py`

**Interfaces:**
- Consumes: `Score`、`Attack`（Task 3）、`Profile`（Task 1）
- Produces:
  - `Violation(bar: int /*1-based*/, start: float, rule: str, detail: str)`，`rule ∈ {"same_string","span","fingers","max_fret","shift","plucks","bar_length"}`
  - `fingers_needed(frets: list[int]) -> int`
  - `check_playability(score, profile, kind: "strum"|"finger") -> list[Violation]`
  - `soft_stats(score) -> dict[int, dict]`（键 1-based 小节号：`attacks_per_beat`、`max_span`、`barre`、`chord_marks`）

- [ ] **Step 1: 写失败测试 `tests/test_playability.py`**

```python
from guitarize.config import load_profile
from guitarize.render import alphatab
from guitarize.verify.model import load_score
from guitarize.verify.playability import check_playability, fingers_needed, soft_stats

PROFILE = load_profile("intermediate")


def rules(score, kind="finger"):
    return [v.rule for v in check_playability(score, PROFILE, kind)]


def test_fingers_needed():
    assert fingers_needed([1, 1, 2, 3, 3, 1]) == 4  # F 大横按
    assert fingers_needed([0, 0, 0]) == 0
    assert fingers_needed([2, 3, 4, 4, 3]) == 5


def test_span_violation(build_tex):
    assert rules(build_tex(":4 (1.1 5.2) r r r |")) == ["span"]


def test_fingers_violation(build_tex):
    assert rules(build_tex(":4 (2.1 3.2 4.3 4.4 3.5) {bd} r r r |")) == ["fingers"]


def test_fast_shift_violation(build_tex):
    assert rules(build_tex(":8 1.1 8.1 :4 r :2 r |")) == ["shift"]


def test_slow_shift_ok(build_tex):
    assert rules(build_tex(":4 1.1 8.1 r r |")) == []


def test_plucks_only_for_finger_without_strum(build_tex):
    s = build_tex(":4 (0.1 0.2 0.3 2.4 3.5) r r r |")
    assert rules(s, "finger") == ["plucks"]
    assert rules(s, "strum") == []
    assert rules(build_tex(":4 (0.1 0.2 0.3 2.4 3.5) {bd} r r r |"), "finger") == []


def test_max_fret_violation(build_tex):
    assert rules(build_tex(":4 13.1 r r r |")) == ["max_fret"]


def test_bar_length_violation(build_tex):
    assert rules(build_tex(":4 0.1 0.1 0.1 0.1 0.1 |")) == ["bar_length"]


def test_demo_fixtures_clean(demo_song, tmp_path):
    for arr in ("finger", "strum"):
        score = load_score(alphatab.build(demo_song.atex(arr), tmp_path / "b", arr)["json"])
        assert check_playability(score, PROFILE, arr) == []
    stats = soft_stats(score)
    assert stats[2]["barre"] is True and stats[1]["barre"] is False
    assert stats[2]["chord_marks"] == ["F", "C"]
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest tests/test_playability.py -v`
Expected: FAIL（`ModuleNotFoundError: guitarize.verify.playability`）

- [ ] **Step 3: 实现 `guitarize/verify/playability.py`**

```python
"""可弹性检查：硬规则（违反即不合格）+ 软统计（只报告）。"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Profile
from .model import Attack, Score


@dataclass(frozen=True)
class Violation:
    bar: int      # 1-based
    start: float  # 拍
    rule: str
    detail: str


def fingers_needed(frets: list[int]) -> int:
    """按弦手指数：最低品上 ≥ 2 根弦视为横按，只算 1 指。"""
    fretted = [f for f in frets if f > 0]
    if not fretted:
        return 0
    low = min(fretted)
    return len(fretted) - (fretted.count(low) - 1)


def _fretted(a: Attack) -> list[int]:
    return [n.fret for n in a.notes if n.fret > 0 and not n.dead]


def check_playability(score: Score, profile: Profile, kind: str) -> list[Violation]:
    out: list[Violation] = []
    for (bar, voice), length in sorted(score.voice_lengths().items()):
        expected = score.bars[bar].length
        if abs(length - expected) > 1e-6:
            out.append(Violation(bar + 1, score.bars[bar].start, "bar_length",
                                 f"声部 {voice + 1} 共 {length:g} 拍，应为 {expected:g} 拍"))
    last: tuple[float, int] | None = None  # (时间, 把位)
    for a in score.attacks():
        bar = a.bar + 1
        strings = [n.string for n in a.notes]
        if len(strings) != len(set(strings)):
            out.append(Violation(bar, a.start, "same_string", f"同一弦同时两个音：{sorted(strings)}"))
        fr = _fretted(a)
        if fr:
            if max(fr) - min(fr) > profile.max_span:
                out.append(Violation(bar, a.start, "span", f"跨度 {min(fr)}–{max(fr)} 品，超过 {profile.max_span}"))
            if fingers_needed(fr) > profile.max_fingers:
                out.append(Violation(bar, a.start, "fingers", f"需要 {fingers_needed(fr)} 根手指"))
            if max(fr) > profile.max_fret:
                out.append(Violation(bar, a.start, "max_fret", f"最高 {max(fr)} 品，超过 {profile.max_fret}"))
            pos = min(fr)
            if last is not None and a.start - last[0] < 1.0 and abs(pos - last[1]) > profile.max_shift:
                out.append(Violation(bar, a.start, "shift",
                                     f"{a.start - last[0]:.2f} 拍内换把 {last[1]}→{pos} 品"))
            last = (a.start, pos)
        if kind == "finger" and not a.strum and len(a.notes) > profile.max_plucks:
            out.append(Violation(bar, a.start, "plucks", f"同时拨 {len(a.notes)} 个音"))
    return out


def _is_barre(a: Attack) -> bool:
    fr = _fretted(a)
    return bool(fr) and fr.count(min(fr)) >= 3


def soft_stats(score: Score) -> dict[int, dict]:
    attacks = score.attacks()
    stats: dict[int, dict] = {}
    for bar in score.bars:
        mine = [a for a in attacks if a.bar == bar.index]
        spans = [max(f) - min(f) for f in (_fretted(a) for a in mine) if f]
        stats[bar.index + 1] = {
            "attacks_per_beat": round(len(mine) / bar.length, 2),
            "max_span": max(spans, default=0),
            "barre": any(_is_barre(a) for a in mine),
            "chord_marks": [e.chord for e in score.events if e.bar == bar.index and e.chord],
        }
    return stats
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest tests/test_playability.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add guitarize/verify/playability.py tests/test_playability.py
git commit -m "feat: playability rules for strum and fingerstyle arrangements"
```

---

### Task 5: 文本六线谱

**Files:**
- Create: `guitarize/render/ascii_tab.py`
- Test: `tests/test_ascii.py`

**Interfaces:**
- Consumes: `Score`、`Bar`（Task 3）、`theory.pc_name`（Task 2）
- Produces: `render_ascii(score: Score, bars_per_line: int = 4) -> str`

- [ ] **Step 1: 写失败测试 `tests/test_ascii.py`**

```python
from guitarize.render import alphatab
from guitarize.render.ascii_tab import render_ascii
from guitarize.verify.model import load_score


def test_single_bar_layout(build_tex):
    text = render_ascii(build_tex(":4 0.1 1.1 2.1 3.1 |"))
    lines = text.splitlines()
    assert "e|0---1---2---3---|" in lines
    assert "B|----------------|" in lines
    assert "E|----------------|" in lines


def test_shuffle_and_capo_header(build_tex):
    from conftest import TEX_HEAD

    text = render_ascii(build_tex("\\tf triplet-8th\n:1 0.1 |", head=TEX_HEAD + "\\capo 4\n"))
    assert "Capo 4" in text and "Shuffle" in text


def test_demo_strum_has_chords_and_arrows(demo_song, tmp_path):
    score = load_score(alphatab.build(demo_song.atex("strum"), tmp_path / "b", "s")["json"])
    text = render_ascii(score)
    lines = text.splitlines()
    chord_lines = [l for l in lines if l.lstrip().startswith("C ")]
    assert chord_lines, text
    assert any("↓" in l and "↑" in l for l in lines)
    assert sum(1 for l in lines if l.startswith("e|")) == 2  # 8 小节 / 每行 4 小节
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest tests/test_ascii.py -v`
Expected: FAIL（`ModuleNotFoundError: guitarize.render.ascii_tab`）

- [ ] **Step 3: 实现 `guitarize/render/ascii_tab.py`**

```python
"""文本六线谱：每行若干小节，带小节号、段落、和弦名、扫弦方向。"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import theory
from ..verify.model import Bar, Score

CHARS_PER_BEAT = 4


@dataclass
class _Col:
    start: float
    frets: dict[int, str] = field(default_factory=dict)
    chord: str = ""
    strum: str = ""


def _columns(score: Score, bar: Bar) -> list[_Col]:
    cols: dict[float, _Col] = {}
    for e in score.events:
        if e.bar != bar.index:
            continue
        c = cols.setdefault(round(e.written_start, 6), _Col(e.written_start))
        for n in e.notes:
            if not n.tie:
                c.frets[n.string] = "x" if n.dead else str(n.fret)
        if e.chord:
            c.chord = e.chord
        direction = e.pick or e.brush
        if direction:
            c.strum = "↓" if direction == "down" else "↑"
    return [cols[k] for k in sorted(cols)]


def _bar_block(score: Score, bar: Bar) -> dict[str, str]:
    n = len(score.tuning)
    rows = {"chord": "", "strum": "", **{f"s{s}": "" for s in range(1, n + 1)}}
    cols = _columns(score, bar)
    if not cols:
        width = round(bar.length * CHARS_PER_BEAT)
        return {k: ("-" * width if k.startswith("s") else " " * width) for k in rows}
    end = bar.start + bar.length
    for i, c in enumerate(cols):
        nxt = cols[i + 1].start if i + 1 < len(cols) else end
        width = max(
            round((nxt - c.start) * CHARS_PER_BEAT),
            max((len(t) for t in c.frets.values()), default=1) + 1,
            len(c.chord) + 1,
            2,
        )
        rows["chord"] += c.chord.ljust(width)
        rows["strum"] += c.strum.ljust(width)
        for s in range(1, n + 1):
            rows[f"s{s}"] += c.frets.get(s, "").ljust(width, "-")
    return rows


def render_ascii(score: Score, bars_per_line: int = 4) -> str:
    n = len(score.tuning)
    names = [theory.pc_name(m) for m in score.tuning]  # tuning[0] = 1 弦
    names[0] = names[0].lower()
    label_w = max(len(x) for x in names)
    info = f"♩ = {score.tempo:g}"
    if score.capo:
        info += f" · Capo {score.capo}"
    info += " · 调弦 " + " ".join(reversed([theory.pc_name(m) for m in score.tuning]))
    out = [score.title, info]
    if any(b.feel == "triplet8th" for b in score.bars):
        out.append("Shuffle：八分音符按三连音感觉弹（♪♪ = ♩♪³）")
    out.append("")
    pad = " " * (label_w + 1)
    for i in range(0, len(score.bars), bars_per_line):
        group = score.bars[i:i + bars_per_line]
        top, chords, strum = pad, pad, pad
        strings = [names[s - 1].rjust(label_w) + "|" for s in range(1, n + 1)]
        for bar in group:
            blk = _bar_block(score, bar)
            width = len(blk["s1"])
            label = f"{bar.index + 1}" + (f" [{bar.section}]" if bar.section else "")
            top += label.ljust(width + 1)
            chords += blk["chord"] + " "
            strum += blk["strum"] + " "
            for s in range(1, n + 1):
                strings[s - 1] += blk[f"s{s}"] + "|"
        out.append(top.rstrip())
        if chords.strip():
            out.append(chords.rstrip())
        out += strings
        if strum.strip():
            out.append(strum.rstrip())
        out.append("")
    return "\n".join(out)
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest tests/test_ascii.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add guitarize/render/ascii_tab.py tests/test_ascii.py
git commit -m "feat: plain-text tab renderer"
```

---

### Task 6: 合成音频、PDF、交互网页、渲染编排

**Files:**
- Create: `guitarize/render/synth.py`, `guitarize/render/pdf.py`, `guitarize/render/viewer.py`, `guitarize/render/pipeline.py`, `web/viewer.html`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `alphatab.build`、`load_score`、`render_ascii`、`SongPaths`、`SongConfig`、`ARR_NAMES`
- Produces:
  - `synth.ALPHATAB_DIST: Path`、`synth.SOUNDFONT: Path`、`midi_to_wav(mid, wav, sr=44100) -> Path`
  - `pdf.svg_to_pdf(svg_fragment: Path, pdf: Path, title: str) -> Path`
  - `viewer.build_viewer(paths, arr, title) -> Path`（out/<arr>.html）
  - `pipeline.render_all(paths, cfg, arr) -> dict[str, Path]`（键：`gp` 以及 cfg.outputs 中的 `txt / pdf / html`）

- [ ] **Step 1: 写失败测试 `tests/test_render.py`**

```python
import json

import numpy as np
import soundfile as sf

from guitarize.config import load_song_config
from guitarize.render import alphatab
from guitarize.render.pipeline import render_all
from guitarize.render.synth import midi_to_wav


def test_midi_to_wav(demo_song, tmp_path):
    built = alphatab.build(demo_song.atex("finger"), tmp_path / "b", "f")
    wav = midi_to_wav(built["mid"], tmp_path / "f.wav")
    y, sr = sf.read(wav)
    assert sr == 44100
    assert 19.0 < len(y) / sr < 24.0  # 8 小节 × 2.4 秒 + 余音
    assert np.sqrt(np.mean(np.square(y))) > 1e-3


def test_render_all(demo_song):
    cfg = load_song_config(demo_song)
    out = render_all(demo_song, cfg, "finger")
    assert set(out) == {"gp", "txt", "pdf", "html"}
    assert out["pdf"].read_bytes()[:4] == b"%PDF" and out["pdf"].stat().st_size > 10_000
    assert "e|" in out["txt"].read_text(encoding="utf-8")
    page = out["html"].read_text(encoding="utf-8")
    assert "{{" not in page
    assert "alphaTab.js" in page and "\\voice" in page
    bar_times = json.loads(page.split("window.BAR_TIMES = ")[1].split(";")[0])
    assert len(bar_times) == 8 and bar_times[1] == [2.4, 4.8]
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest tests/test_render.py -v`
Expected: FAIL（`ModuleNotFoundError: guitarize.render.pipeline`）

- [ ] **Step 3: 实现 `guitarize/render/synth.py`**

```python
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
```

- [ ] **Step 4: 实现 `guitarize/render/pdf.py`**

```python
"""SVG 片段 → PDF：套 Bravura 字体后用无头 Chrome 打印（独立临时配置目录，不碰用户的 Chrome）。"""
from __future__ import annotations

import html
import subprocess
import tempfile
from pathlib import Path

from .synth import ALPHATAB_DIST

CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
FONT = ALPHATAB_DIST / "font" / "Bravura.woff2"
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>
@font-face{{font-family:"alphaTab";src:url("{font}") format("woff2")}}
.at{{font-family:"alphaTab";speak:none;font-style:normal;font-weight:normal;line-height:1;font-size:34px}}
@page{{size:A4;margin:12mm}}
body{{margin:0;background:#fff}}
svg{{display:block;break-inside:avoid;margin:0 auto}}
</style></head><body>{body}</body></html>"""


def svg_to_pdf(svg_fragment: Path, pdf: Path, title: str) -> Path:
    if not CHROME.exists():
        raise RuntimeError(f"Google Chrome not found at {CHROME}")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    page = pdf.with_suffix(".print.html")
    page.write_text(
        PAGE.format(title=html.escape(title), font=FONT.as_uri(), body=svg_fragment.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    try:
        with tempfile.TemporaryDirectory(prefix="guitarize-chrome-") as profile:
            subprocess.run(
                [str(CHROME), "--headless=new", "--disable-gpu", f"--user-data-dir={profile}",
                 "--no-pdf-header-footer", f"--print-to-pdf={pdf}", "--virtual-time-budget=5000", page.as_uri()],
                check=True,
                capture_output=True,
                timeout=120,
            )
    finally:
        page.unlink(missing_ok=True)
    return pdf
```

- [ ] **Step 5: 写交互谱模板 `web/viewer.html`**

```html
<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<style>
  :root { --bg: #fafaf7; --fg: #1d1d1b; --muted: #6b6b66; --accent: #c2410c; --panel: #ffffffee; --line: #ddd; }
  body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.4 -apple-system, "PingFang SC", sans-serif; }
  header { position: sticky; top: 0; z-index: 10; background: var(--panel); backdrop-filter: blur(6px);
           border-bottom: 1px solid var(--line); padding: 8px 16px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  header h1 { font-size: 16px; margin: 0 12px 0 0; }
  button, select { font: inherit; padding: 4px 10px; border: 1px solid #ccc; border-radius: 6px; background: #fff; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  #status { color: var(--muted); font-size: 13px; }
  #wrap { padding: 8px 16px 80px; }
  .at-cursor-bar { background: rgba(194, 65, 12, .08); }
  .at-cursor-beat { background: var(--accent); width: 3px; }
  .at-highlight * { fill: var(--accent); stroke: var(--accent); }
  .at-selection div { background: rgba(194, 65, 12, .15); }
</style>
</head>
<body>
<header>
  <h1>{{TITLE}}</h1>
  <button id="play" disabled>▶ 合成音</button>
  <button id="orig">▶ 原曲（当前小节起）</button>
  <label>速度 <select id="speed"><option>0.5</option><option>0.75</option><option selected>1</option></select></label>
  <label><input type="checkbox" id="metro"> 节拍器</label>
  <label><input type="checkbox" id="loop"> 循环选区</label>
  <span id="status">加载中…</span>
</header>
<div id="wrap"><div id="at"></div></div>
<audio id="audio" src="{{AUDIO_URL}}" preload="auto"></audio>
<script id="tex" type="text/plain">{{ATEX}}</script>
<script>window.BAR_TIMES = {{BAR_TIMES}};</script>
<script src="{{ALPHATAB_BASE}}alphaTab.js"></script>
<script>
  const base = "{{ALPHATAB_BASE}}";
  const api = new alphaTab.AlphaTabApi(document.getElementById("at"), {
    core: { fontDirectory: base + "font/" },
    player: {
      playerMode: alphaTab.PlayerMode.EnabledSynthesizer,
      soundFont: base + "soundfont/sonivox.sf2",
      scrollElement: document.scrollingElement,
    },
  });
  const status = document.getElementById("status");
  const audio = document.getElementById("audio");
  const playBtn = document.getElementById("play");
  const origBtn = document.getElementById("orig");
  let currentBar = 0;

  api.tex(document.getElementById("tex").textContent);
  api.playerReady.on(() => {
    playBtn.disabled = false;
    status.textContent = "就绪：点音符选定小节，拖选可设置循环区";
  });
  api.error.on((e) => { status.textContent = "出错：" + e.message; });
  api.beatMouseDown.on((beat) => { currentBar = beat.voice.bar.index; });
  api.playedBeatChanged.on((beat) => { currentBar = beat.voice.bar.index; });

  playBtn.onclick = () => { audio.pause(); origBtn.textContent = "▶ 原曲（当前小节起）"; api.playPause(); };
  document.getElementById("speed").onchange = (e) => {
    const v = parseFloat(e.target.value);
    api.playbackSpeed = v;
    audio.playbackRate = v;
  };
  document.getElementById("metro").onchange = (e) => { api.metronomeVolume = e.target.checked ? 1 : 0; };
  document.getElementById("loop").onchange = (e) => { api.isLooping = e.target.checked; };

  if (!window.BAR_TIMES.length) {
    origBtn.disabled = true;
    origBtn.title = "缺少原曲小节时间（leadsheet.json）";
  }
  origBtn.onclick = () => {
    if (!audio.paused) {
      audio.pause();
      origBtn.textContent = "▶ 原曲（当前小节起）";
      return;
    }
    api.pause();
    const t = window.BAR_TIMES[Math.min(currentBar, window.BAR_TIMES.length - 1)];
    audio.currentTime = t[0];
    audio.play();
    origBtn.textContent = "⏸ 原曲";
  };
</script>
</body>
</html>
```

- [ ] **Step 6: 实现 `guitarize/render/viewer.py` 与 `guitarize/render/pipeline.py`**

`guitarize/render/viewer.py`：
```python
"""生成本地交互谱页面：alphaTab 渲染 + 合成播放 + 按小节跳转原曲。需从仓库根目录起 HTTP 服务打开。"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

from ..config import ARR_NAMES
from ..paths import REPO_ROOT, SongPaths
from .synth import ALPHATAB_DIST

TEMPLATE = REPO_ROOT / "web" / "viewer.html"


def _rel(target: Path, start: Path) -> str:
    return Path(os.path.relpath(target, start)).as_posix()


def build_viewer(paths: SongPaths, arr: str, title: str) -> Path:
    out = paths.out(arr, "html")
    out.parent.mkdir(parents=True, exist_ok=True)
    bar_times: list[list[float]] = []
    if paths.leadsheet_json.exists():
        leadsheet = json.loads(paths.leadsheet_json.read_text(encoding="utf-8"))
        bar_times = [[b["t_start"], b["t_end"]] for b in leadsheet["bars"]]
    replacements = {
        "{{TITLE}}": html.escape(f"{title} · {ARR_NAMES[arr]}"),
        "{{ALPHATAB_BASE}}": _rel(ALPHATAB_DIST, out.parent) + "/",
        "{{AUDIO_URL}}": _rel(paths.original_wav, out.parent),
        "{{BAR_TIMES}}": json.dumps(bar_times),
        "{{ATEX}}": paths.atex(arr).read_text(encoding="utf-8").replace("</script", "<\\/script"),
    }
    page = TEMPLATE.read_text(encoding="utf-8")
    for key, value in replacements.items():
        page = page.replace(key, value)
    out.write_text(page, encoding="utf-8")
    return out
```

`guitarize/render/pipeline.py`：
```python
"""渲染编排：alphaTex → build/ 中间产物 → out/ 交付物（gp / txt / pdf / html）。"""
from __future__ import annotations

import shutil
from pathlib import Path

from ..config import SongConfig
from ..paths import SongPaths
from ..verify.model import load_score
from . import alphatab
from .ascii_tab import render_ascii
from .pdf import svg_to_pdf
from .viewer import build_viewer


def render_all(paths: SongPaths, cfg: SongConfig, arr: str) -> dict[str, Path]:
    built = alphatab.build(paths.atex(arr), paths.build_dir, arr)
    score = load_score(built["json"])
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    out = {"gp": paths.out(arr, "gp")}
    shutil.copyfile(built["gp"], out["gp"])
    if "txt" in cfg.outputs:
        out["txt"] = paths.out(arr, "txt")
        out["txt"].write_text(render_ascii(score), encoding="utf-8")
    if "pdf" in cfg.outputs:
        out["pdf"] = svg_to_pdf(built["svg"], paths.out(arr, "pdf"), score.title)
    if "html" in cfg.outputs:
        out["html"] = build_viewer(paths, arr, score.title)
    return out
```

- [ ] **Step 7: 运行，确认通过**

Run: `$PY -m pytest tests/test_render.py -v`
Expected: 2 passed

- [ ] **Step 8: Commit**

```bash
git add guitarize/render web tests/test_render.py
git commit -m "feat: audio synthesis, PDF printing and interactive tab viewer"
```

---

### Task 7: 逐小节评分

**Files:**
- Create: `guitarize/verify/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `Score`（Task 3）、`theory`、`skeleton_mask`（Task 2）、`Thresholds`（Task 1）、`Violation`（Task 4）
- Produces:
  - `BarScore`（字段见代码；属性 `chord_ratio`、`mel_f1`；`to_dict()`）
  - `f1(hit, est, found, ref) -> float | None`
  - `score_bars(score, leadsheet: dict | None, arr, synth_wav=None, harmonic_wav=None, onset_wavs=()) -> list[BarScore]`
  - `summarize(bars, violations, thresholds, arr) -> dict`（键：`violations, chord_agreement, melody_precision, melody_recall, melody_f1, melody_octave_exact, harmonic_similarity, rhythm_correlation, low_conf_beats, chord_mark_issues, chord_target, melody_target, checks, pass`；弹唱的 `checks` 含 `chord_marks`，指弹只报告不卡线，因为指弹一拍内音少、推断不稳）
  - `problem_bars(bars, violations, thresholds, arr) -> list[int]`（1-based）
  - `chord_mark_issues(score, bars) -> None`：谱上写的和弦名（按变调夹换算到实际音高）与该处一拍内实际发声的音不符时，写入 `BarScore.mark_issues`；`score_bars` 总会调用它（有无 leadsheet 都调用）
  - 底层：`bar_chroma(wav, windows) -> np.ndarray`、`cosine(a, b)`、`onset_slots(wavs, windows, slots=12)`、`strum_vector(attacks, bar, slots=12)`、`pearson(a, b)`

- [ ] **Step 1: 写失败测试 `tests/test_metrics.py`**

```python
import copy

import numpy as np
import pytest

from conftest import demo_leadsheet
from guitarize.config import Thresholds
from guitarize.render import alphatab
from guitarize.render.synth import midi_to_wav
from guitarize.verify import metrics
from guitarize.verify.model import load_score


@pytest.fixture
def demo_scores(demo_song, tmp_path):
    out = {}
    for arr in ("finger", "strum"):
        built = alphatab.build(demo_song.atex(arr), tmp_path / "b", arr)
        out[arr] = (load_score(built["json"]), built["mid"])
    return out


def test_f1():
    assert metrics.f1(0, 0, 0, 0) is None
    assert metrics.f1(4, 4, 4, 4) == 1.0
    assert metrics.f1(2, 4, 1, 2) == pytest.approx(0.5)


def test_demo_finger_perfect(demo_scores):
    score, _ = demo_scores["finger"]
    bars = metrics.score_bars(score, demo_leadsheet(), "finger")
    s = metrics.summarize(bars, [], Thresholds(), "finger")
    assert s["chord_agreement"] == 1.0
    assert s["melody_precision"] == 1.0 and s["melody_recall"] == 1.0 and s["melody_f1"] == 1.0
    assert s["melody_octave_exact"] == 1.0
    assert s["pass"] is True
    assert metrics.problem_bars(bars, [], Thresholds(), "finger") == []


def test_wrong_reference_chord_flagged(demo_scores):
    score, _ = demo_scores["strum"]
    ls = demo_leadsheet()
    ls["bars"][0]["chords"] = [{"beat": 1, "dur_beats": 4, "label": "Am", "confidence": "high"}]
    bars = metrics.score_bars(score, ls, "strum")
    assert bars[0].chord_agree == 0 and bars[0].chord_total == 4
    assert "参考 Am" in bars[0].chord_issues[0]
    assert metrics.problem_bars(bars, [], Thresholds(), "strum") == [1]


def test_low_confidence_beats_excluded(demo_scores):
    score, _ = demo_scores["strum"]
    ls = demo_leadsheet()
    ls["bars"][0]["chords"] = [{"beat": 1, "dur_beats": 4, "label": "Am", "confidence": "low"}]
    bars = metrics.score_bars(score, ls, "strum")
    assert bars[0].chord_total == 0 and bars[0].low_conf_beats == [1, 2, 3, 4]


def test_melody_wrong_note(demo_scores):
    score, _ = demo_scores["finger"]
    ls = demo_leadsheet()
    ls["bars"][0]["melody"][2]["midi"] = 69  # 参考第 1 小节第 3 拍改成 A4
    bars = metrics.score_bars(score, ls, "finger")
    assert (bars[0].mel_hit, bars[0].mel_est) == (3, 4)
    assert (bars[0].mel_found, bars[0].mel_ref) == (3, 4)
    assert any("漏掉骨架音" in x for x in bars[0].mel_issues)
    assert any("多余/错音" in x for x in bars[0].mel_issues)


def test_melody_octave_tolerated(demo_scores):
    score, _ = demo_scores["finger"]
    ls = copy.deepcopy(demo_leadsheet())
    for b in ls["bars"]:
        for n in b["melody"]:
            n["midi"] -= 12
    s = metrics.summarize(metrics.score_bars(score, ls, "finger"), [], Thresholds(), "finger")
    assert s["melody_f1"] == 1.0 and s["melody_octave_exact"] == 0.0


def test_bars_without_reference_melody_not_scored(demo_scores):
    score, _ = demo_scores["finger"]
    ls = demo_leadsheet()
    ls["bars"][0]["melody"] = []
    bars = metrics.score_bars(score, ls, "finger")
    assert bars[0].mel_est == 0 and bars[0].mel_f1 is None


def test_harmonic_similarity_self(demo_scores, tmp_path):
    score, mid = demo_scores["finger"]
    wav = midi_to_wav(mid, tmp_path / "f.wav")
    bars = metrics.score_bars(score, demo_leadsheet(), "finger", synth_wav=wav, harmonic_wav=wav)
    sims = [b.harmonic_sim for b in bars]
    assert all(s is not None and s > 0.99 for s in sims)


def test_chord_mark_mismatch(build_tex):
    from conftest import TEX_HEAD

    score = build_tex('\\chord "G" 3 0 0 0 2 3\n:1 (0.1 1.2 0.3 2.4 3.5) {ch "G" sd} |', head=TEX_HEAD)
    bars = metrics.score_bars(score, None, "strum")
    assert bars[0].mark_issues and "标记 G" in bars[0].mark_issues[0]
    s = metrics.summarize(bars, [], Thresholds(), "strum")
    assert s["chord_mark_issues"] == 1 and s["checks"]["chord_marks"] is False
    assert metrics.problem_bars(bars, [], Thresholds(), "strum") == [1]


def test_demo_strum_marks_consistent(demo_scores):
    score, _ = demo_scores["strum"]
    bars = metrics.score_bars(score, demo_leadsheet(), "strum")
    assert all(not b.mark_issues for b in bars)
    assert metrics.summarize(bars, [], Thresholds(), "strum")["pass"] is True


def test_strum_vector_and_pearson(demo_scores):
    score, _ = demo_scores["strum"]
    v = metrics.strum_vector(score.attacks(), score.bars[0])
    # 每半小节：↓ 在 0，↓ 在 1 拍（slot 3），↑ 在 1.67 拍（slot 5）
    assert list(np.flatnonzero(v)) == [0, 3, 5, 6, 9, 11]
    assert v[5] == pytest.approx(0.6) and v[0] == 1.0
    assert metrics.pearson(v, v) == pytest.approx(1.0)
    assert metrics.pearson(np.ones(12), v) is None
```

注：demo 是直八分（无 triplet feel），反拍八分在 1.5 拍 → slot = round(1.5 × 3) = round(4.5) = 4（Python 银行家舍入），会与测试期望的 slot 5 冲突。因此 `strum_vector` 必须用 `int(x + 0.5)` 做四舍五入（1.5 拍 → slot 5），见实现。

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest tests/test_metrics.py -v`
Expected: FAIL（`ImportError: cannot import name 'metrics'`）

- [ ] **Step 3: 实现 `guitarize/verify/metrics.py`**

```python
"""编配 vs 参考（leadsheet.json）的逐小节评分。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .. import theory
from ..config import Thresholds
from ..skeleton import skeleton_mask
from .model import Attack, Bar, Score

MELODY_TOL = 1 / 3 + 1e-6  # 起音容差（拍）
SLOTS = 12                 # 4/4 小节的三连音八分格数


@dataclass
class BarScore:
    bar: int  # 1-based
    section: str | None = None
    chord_agree: int = 0
    chord_total: int = 0
    chord_issues: list[str] = field(default_factory=list)
    low_conf_beats: list[int] = field(default_factory=list)
    mel_hit: int = 0           # 编配旋律音命中参考（精确率分子）
    mel_est: int = 0           # 编配旋律音数（精确率分母）
    mel_found: int = 0         # 参考骨架音被覆盖（召回率分子）
    mel_ref: int = 0           # 参考骨架音数（召回率分母）
    mel_octave_exact: int = 0
    mel_issues: list[str] = field(default_factory=list)
    mark_issues: list[str] = field(default_factory=list)  # 谱上和弦名与实际音不符
    harmonic_sim: float | None = None
    rhythm_corr: float | None = None

    @property
    def chord_ratio(self) -> float | None:
        return self.chord_agree / self.chord_total if self.chord_total else None

    @property
    def mel_f1(self) -> float | None:
        return f1(self.mel_hit, self.mel_est, self.mel_found, self.mel_ref)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["chord_ratio"] = self.chord_ratio
        d["mel_f1"] = self.mel_f1
        return d


def f1(hit: int, est: int, found: int, ref: int) -> float | None:
    if est == 0 and ref == 0:
        return None
    p = hit / est if est else 0.0
    r = found / ref if ref else 0.0
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


# ---------- 和弦 ----------

def _ref_beats(lbar: dict, bpb: int) -> list[tuple[str, str, str]]:
    """小节内逐拍参考：(majmin, 置信度, 原标签)。"""
    out = [("N", "high", "N")] * bpb
    for seg in lbar.get("chords", []):
        first = int(round(seg["beat"])) - 1
        n = max(1, int(round(seg["dur_beats"])))
        for k in range(max(0, first), min(bpb, first + n)):
            out[k] = (theory.majmin(seg["label"]), seg.get("confidence", "high"), seg["label"])
    return out


def _pc_weights(sounding, t0: float, t1: float) -> tuple[np.ndarray, int | None]:
    w = np.zeros(12)
    bass = None
    for s in sounding:
        overlap = min(s.end, t1) - max(s.start, t0)
        if overlap > 1e-6:
            w[s.midi % 12] += overlap
            if bass is None or s.midi < bass:
                bass = s.midi
    return w, (None if bass is None else bass % 12)


def score_chords(score: Score, leadsheet: dict, bars: list[BarScore]) -> None:
    bpb = int(leadsheet.get("beats_per_bar", 4))
    sounding = score.sounding()
    prev: set[str] | None = None
    for bs, lbar, sbar in zip(bars, leadsheet["bars"], score.bars):
        for k, (ref, conf, raw) in enumerate(_ref_beats(lbar, bpb)):
            t0 = sbar.start + k
            w, bass = _pc_weights(sounding, t0, t0 + 1)
            if w.sum() > 0:
                cands = theory.infer_majmin(w, bass)
                prev = cands
            else:
                cands = prev or {"N"}  # 空拍：沿用上一拍的余音
            if ref == "N":
                continue
            if conf == "low":
                bs.low_conf_beats.append(k + 1)
                continue
            bs.chord_total += 1
            if ref in cands:
                bs.chord_agree += 1
            else:
                shape = theory.transpose_label(raw, -score.capo) if score.capo else raw
                bs.chord_issues.append(f"拍{k + 1}：参考 {raw}（指法 {shape}），编配听起来是 {'/'.join(sorted(cands))}")


def chord_mark_issues(score: Score, bars: list[BarScore]) -> None:
    """谱上写的和弦名（换算到实际音高）与标记处一拍内实际发声的音不符 → 记入 mark_issues。"""
    sounding = score.sounding()
    for e in score.events:
        if not e.chord or e.bar >= len(bars):
            continue
        expected = theory.majmin(theory.transpose_label(e.chord, score.capo))
        w, bass = _pc_weights(sounding, e.start, e.start + 1)
        if expected == "N" or w.sum() == 0:
            continue
        cands = theory.infer_majmin(w, bass)
        if expected not in cands:
            pos = e.start - score.bars[e.bar].start + 1
            bars[e.bar].mark_issues.append(f"拍 {pos:.2f} 标记 {e.chord}，实际音听起来是 {'/'.join(sorted(cands))}")


# ---------- 旋律 ----------

def _bar_of(score: Score, t: float) -> int | None:
    for b in score.bars:
        if b.start - 1e-6 <= t < b.start + b.length - 1e-6:
            return b.index
    return None


def _match(a: Sequence[tuple[float, int]], b: Sequence[tuple[float, int]], tol: float = MELODY_TOL) -> dict[int, int]:
    """a 中每个音找 b 中音名相同、起音最近（≤ tol）且未被占用的音。返回 {a 下标: b 下标}。"""
    used: set[int] = set()
    out: dict[int, int] = {}
    for i, (ta, ma) in enumerate(a):
        best = None
        for j, (tb, mb) in enumerate(b):
            if j in used or (ma - mb) % 12:
                continue
            d = abs(ta - tb)
            if d <= tol and (best is None or d < best[0]):
                best = (d, j)
        if best is not None:
            used.add(best[1])
            out[i] = best[1]
    return out


def score_melody(score: Score, leadsheet: dict, bars: list[BarScore]) -> None:
    n = len(bars)
    ref_all = sorted((m for b in leadsheet["bars"][:n] for m in b.get("melody", [])), key=lambda m: m["start_beats"])
    mask = skeleton_mask([m["start_beats"] for m in ref_all], [m["dur_beats"] for m in ref_all])
    skel = [m for m, keep in zip(ref_all, mask) if keep]
    ref_bars = {_bar_of(score, m["start_beats"]) for m in ref_all}
    est = [e for e in score.melody(voice=0) if _bar_of(score, e.start) in ref_bars and _bar_of(score, e.start) is not None]

    hits = _match([(e.start, e.midi) for e in est], [(m["start_beats"], m["midi"]) for m in ref_all])
    founds = _match([(m["start_beats"], m["midi"]) for m in skel], [(e.start, e.midi) for e in est])

    for i, e in enumerate(est):
        bi = _bar_of(score, e.start)
        if bi is None or bi >= n:
            continue
        bs, pos = bars[bi], e.start - score.bars[bi].start + 1
        bs.mel_est += 1
        if i in hits:
            bs.mel_hit += 1
            if ref_all[hits[i]]["midi"] == e.midi:
                bs.mel_octave_exact += 1
        else:
            bs.mel_issues.append(f"多余/错音：拍 {pos:.2f} {theory.midi_name(e.midi)}")
    for j, m in enumerate(skel):
        bi = _bar_of(score, m["start_beats"])
        if bi is None or bi >= n:
            continue
        bs, pos = bars[bi], m["start_beats"] - score.bars[bi].start + 1
        bs.mel_ref += 1
        if j in founds:
            bs.mel_found += 1
        else:
            shape = theory.midi_name(m["midi"] - score.capo)
            bs.mel_issues.append(f"漏掉骨架音：拍 {pos:.2f} {theory.midi_name(m['midi'])}（指法调 {shape}）")


# ---------- 和声相似度 ----------

def bar_chroma(wav: Path, windows: Sequence[tuple[float, float]]) -> np.ndarray:
    import librosa

    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    hop = 512
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    out = np.full((len(windows), 12), np.nan)
    for i, (a, b) in enumerate(windows):
        m = (times >= a) & (times < b)
        if m.any():
            out[i] = chroma[:, m].mean(axis=1)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.isnan(a).any() or np.isnan(b).any():
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return None if na == 0 or nb == 0 else float(a @ b / (na * nb))


def _synth_windows(score: Score, n: int) -> list[tuple[float, float]]:
    return [(score.seconds(b.start), score.seconds(b.start + b.length)) for b in score.bars[:n]]


def _ref_windows(leadsheet: dict, n: int) -> list[tuple[float, float]]:
    return [(b["t_start"], b["t_end"]) for b in leadsheet["bars"][:n]]


def score_harmonic(score: Score, leadsheet: dict, bars: list[BarScore], synth_wav: Path, harmonic_wav: Path) -> None:
    n = len(bars)
    ref = bar_chroma(harmonic_wav, _ref_windows(leadsheet, n))
    syn = bar_chroma(synth_wav, _synth_windows(score, n))
    for i, bs in enumerate(bars):
        bs.harmonic_sim = cosine(ref[i], syn[i])


# ---------- 节奏（弹唱） ----------

def onset_slots(wavs: Iterable[Path], windows: Sequence[tuple[float, float]], slots: int = SLOTS) -> np.ndarray:
    import librosa

    env = None
    for w in wavs:
        y, sr = librosa.load(str(w), sr=22050, mono=True)
        e = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        env = e if env is None else env[: min(len(env), len(e))] + e[: min(len(env), len(e))]
    times = librosa.frames_to_time(np.arange(len(env)), sr=22050, hop_length=512)
    out = np.zeros((len(windows), slots))
    for i, (a, b) in enumerate(windows):
        step = (b - a) / slots
        for k in range(slots):
            c = a + k * step
            m = (times >= c - step / 2) & (times < c + step / 2)
            out[i, k] = env[m].max() if m.any() else 0.0
    return out


def strum_vector(attacks: Sequence[Attack], bar: Bar, slots: int = SLOTS) -> np.ndarray:
    v = np.zeros(slots)
    for a in attacks:
        if a.bar != bar.index:
            continue
        k = int((a.start - bar.start) / bar.length * slots + 0.5) % slots
        v[k] = max(v[k], 0.6 if a.direction == "up" else 1.0)
    return v


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def score_rhythm(score: Score, leadsheet: dict, bars: list[BarScore], onset_wavs: Sequence[Path]) -> None:
    n = len(bars)
    orig = onset_slots(onset_wavs, _ref_windows(leadsheet, n))
    attacks = score.attacks()
    for i, bs in enumerate(bars):
        bs.rhythm_corr = pearson(strum_vector(attacks, score.bars[i]), orig[i])


# ---------- 汇总 ----------

def score_bars(score: Score, leadsheet: dict | None, arr: str, synth_wav: Path | None = None,
               harmonic_wav: Path | None = None, onset_wavs: Sequence[Path] = ()) -> list[BarScore]:
    if leadsheet is None:
        bars = [BarScore(bar=b.index + 1, section=b.section) for b in score.bars]
        chord_mark_issues(score, bars)
        return bars
    n = min(len(score.bars), len(leadsheet["bars"]))
    bars = [BarScore(bar=i + 1, section=score.bars[i].section or leadsheet["bars"][i].get("section")) for i in range(n)]
    chord_mark_issues(score, bars)
    score_chords(score, leadsheet, bars)
    if arr == "finger":
        score_melody(score, leadsheet, bars)
    if synth_wav is not None and harmonic_wav is not None and Path(harmonic_wav).exists():
        score_harmonic(score, leadsheet, bars, Path(synth_wav), Path(harmonic_wav))
    if arr == "strum" and onset_wavs and all(Path(w).exists() for w in onset_wavs):
        score_rhythm(score, leadsheet, bars, [Path(w) for w in onset_wavs])
    return bars


def summarize(bars: list[BarScore], violations: list, th: Thresholds, arr: str) -> dict:
    ct = sum(b.chord_total for b in bars)
    ca = sum(b.chord_agree for b in bars)
    hit, est = sum(b.mel_hit for b in bars), sum(b.mel_est for b in bars)
    found, ref = sum(b.mel_found for b in bars), sum(b.mel_ref for b in bars)
    chord = ca / ct if ct else None
    mel = f1(hit, est, found, ref)
    hs = [b.harmonic_sim for b in bars if b.harmonic_sim is not None]
    rc = [b.rhythm_corr for b in bars if b.rhythm_corr is not None]
    marks = sum(len(b.mark_issues) for b in bars)
    target = th.chord_strum if arr == "strum" else th.chord_finger
    checks = {"playability": not violations, "chord": chord is None or chord >= target}
    if arr == "strum":
        checks["chord_marks"] = marks == 0
    if arr == "finger":
        checks["melody"] = mel is None or mel >= th.melody_f1
    return {
        "violations": len(violations),
        "chord_agreement": chord,
        "melody_precision": hit / est if est else None,
        "melody_recall": found / ref if ref else None,
        "melody_f1": mel,
        "melody_octave_exact": sum(b.mel_octave_exact for b in bars) / hit if hit else None,
        "harmonic_similarity": float(np.mean(hs)) if hs else None,
        "rhythm_correlation": float(np.mean(rc)) if rc else None,
        "low_conf_beats": sum(len(b.low_conf_beats) for b in bars),
        "chord_mark_issues": marks,
        "chord_target": target,
        "melody_target": th.melody_f1 if arr == "finger" else None,
        "checks": checks,
        "pass": all(checks.values()),
    }


def problem_bars(bars: list[BarScore], violations: list, th: Thresholds, arr: str) -> list[int]:
    bad = {v.bar for v in violations}
    for b in bars:
        if arr == "strum" and b.mark_issues:
            bad.add(b.bar)
        if b.chord_ratio is not None and b.chord_ratio < th.bar_chord_flag:
            bad.add(b.bar)
        if arr == "finger" and b.mel_f1 is not None and b.mel_f1 < th.bar_melody_flag:
            bad.add(b.bar)
        if b.harmonic_sim is not None and b.harmonic_sim < th.bar_harmonic_flag:
            bad.add(b.bar)
    return sorted(bad)
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest tests/test_metrics.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add guitarize/verify/metrics.py tests/test_metrics.py
git commit -m "feat: per-bar chord, melody, harmonic and rhythm scoring"
```

---

### Task 8: 校验报告 + 命令行

**Files:**
- Create: `guitarize/verify/report.py`, `guitarize/cli.py`
- Test: `tests/test_verify_cli.py`

**Interfaces:**
- Consumes: Task 1–7 的全部公开接口
- Produces:
  - `report.run_verify(paths, cfg, arr) -> Path`（返回 `reports/<arr>/iter-NN`，内含 `report.md`、`scores.json`、可选 `snippets/`、`plots/`；并追加 `reports/<arr>/history.csv`）
  - `report.next_iter_dir(root) -> Path`、`report.render_report(...) -> str`
  - CLI：`guitarize [--songs-dir DIR] {render,verify} <slug> --arr {strum,finger}`、`guitarize serve [--port 8765]`

- [ ] **Step 1: 写失败测试 `tests/test_verify_cli.py`**

```python
import csv
import json

from guitarize.cli import main


def test_verify_twice_creates_iterations(demo_song, capsys):
    songs = str(demo_song.root.parent)
    assert main(["--songs-dir", songs, "verify", "demo", "--arr", "finger"]) == 0
    assert main(["--songs-dir", songs, "verify", "demo", "--arr", "finger"]) == 0
    root = demo_song.report_root("finger")
    assert (root / "iter-01" / "report.md").exists() and (root / "iter-02" / "report.md").exists()
    s = json.loads((root / "iter-02" / "scores.json").read_text(encoding="utf-8"))["summary"]
    assert s["pass"] is True and s["violations"] == 0
    assert s["chord_agreement"] == 1.0 and s["melody_f1"] == 1.0
    rows = list(csv.DictReader((root / "history.csv").open(encoding="utf-8")))
    assert [r["iter"] for r in rows] == ["iter-01", "iter-02"]
    report = (root / "iter-02" / "report.md").read_text(encoding="utf-8")
    assert "指弹校验报告" in report and "✅ 达标" in report
    assert "指弹校验报告" in capsys.readouterr().out


def test_verify_flags_problem_bar_with_snippet(demo_song):
    ls = json.loads(demo_song.leadsheet_json.read_text(encoding="utf-8"))
    ls["bars"][1]["chords"] = [{"beat": 1, "dur_beats": 4, "label": "D", "confidence": "high"}]
    demo_song.leadsheet_json.write_text(json.dumps(ls), encoding="utf-8")
    main(["--songs-dir", str(demo_song.root.parent), "verify", "demo", "--arr", "strum"])
    it = demo_song.report_root("strum") / "iter-01"
    s = json.loads((it / "scores.json").read_text(encoding="utf-8"))
    assert s["problems"] == [2] and s["summary"]["pass"] is False
    assert (it / "snippets" / "bar-002_synth.wav").exists()
    report = (it / "report.md").read_text(encoding="utf-8")
    assert "### 第 2 小节" in report and "参考 D" in report


def test_render_cli(demo_song, capsys):
    assert main(["--songs-dir", str(demo_song.root.parent), "render", "demo", "--arr", "strum"]) == 0
    printed = capsys.readouterr().out
    assert "pdf:" in printed and demo_song.out("strum", "html").exists()
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest tests/test_verify_cli.py -v`
Expected: FAIL（`ModuleNotFoundError: guitarize.cli`）

- [ ] **Step 3: 实现 `guitarize/verify/report.py`**

```python
"""一次校验：构建 → 合成 → 可弹性 → 评分 → reports/<arr>/iter-NN（报告、片段、色度图）+ history.csv。"""
from __future__ import annotations

import csv
import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import theory
from ..config import ARR_NAMES, SongConfig, load_profile
from ..paths import SongPaths
from ..render import alphatab
from ..render.synth import midi_to_wav
from . import metrics
from .model import Score, load_score
from .playability import Violation, check_playability, soft_stats

HISTORY_FIELDS = ["iter", "time", "violations", "chord_agreement", "melody_f1",
                  "harmonic_similarity", "rhythm_correlation", "pass"]
MAX_SNIPPETS = 20


def next_iter_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name[5:]) for p in root.glob("iter-*") if p.name[5:].isdigit()]
    out = root / f"iter-{(max(nums) + 1 if nums else 1):02d}"
    out.mkdir()
    return out


def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def run_verify(paths: SongPaths, cfg: SongConfig, arr: str) -> Path:
    profile = load_profile(cfg.level)
    built = alphatab.build(paths.atex(arr), paths.build_dir, arr)
    score = load_score(built["json"])
    synth = midi_to_wav(built["mid"], paths.build(arr, "wav"))
    violations = check_playability(score, profile, arr)
    leadsheet = _load_json(paths.leadsheet_json)
    bars = metrics.score_bars(score, leadsheet, arr, synth_wav=synth, harmonic_wav=paths.stem("harmonic"),
                              onset_wavs=(paths.stem("drums"), paths.stem("guitar")))
    summary = metrics.summarize(bars, violations, cfg.thresholds, arr)
    problems = metrics.problem_bars(bars, violations, cfg.thresholds, arr)

    root = paths.report_root(arr)
    out = next_iter_dir(root)
    snippets = _write_snippets(out, problems, score, leadsheet, synth, paths.original_wav)
    plots = _write_plots(out, problems, score, leadsheet, synth, paths.stem("harmonic"))
    (out / "scores.json").write_text(json.dumps({
        "summary": summary,
        "problems": problems,
        "bars": [b.to_dict() for b in bars],
        "violations": [asdict(v) for v in violations],
        "soft": soft_stats(score),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    history = root / "history.csv"
    prev = _last_row(history)
    (out / "report.md").write_text(
        render_report(arr, out.name, summary, prev, violations, bars, problems, snippets, plots, score),
        encoding="utf-8",
    )
    _append_history(history, out.name, summary)
    return out


# ---------- 片段与图 ----------

def _cut(wav: Path, a: float, b: float, dst: Path) -> None:
    info = sf.info(str(wav))
    start = max(0, int(a * info.samplerate))
    stop = min(info.frames, int(b * info.samplerate))
    data, sr = sf.read(str(wav), start=start, stop=stop)
    sf.write(str(dst), data, sr)


def _write_snippets(out: Path, problems: list[int], score: Score, leadsheet: dict | None,
                    synth: Path, original: Path) -> dict[int, list[str]]:
    res: dict[int, list[str]] = {}
    folder = out / "snippets"
    for bar in problems[:MAX_SNIPPETS]:
        i = bar - 1
        files = []
        folder.mkdir(exist_ok=True)
        if leadsheet and original.exists() and i < len(leadsheet["bars"]):
            lb = leadsheet["bars"][i]
            p = folder / f"bar-{bar:03d}_orig.wav"
            _cut(original, lb["t_start"], lb["t_end"], p)
            files.append(p.relative_to(out).as_posix())
        if i < len(score.bars):
            sb = score.bars[i]
            p = folder / f"bar-{bar:03d}_synth.wav"
            _cut(synth, score.seconds(sb.start), score.seconds(sb.start + sb.length), p)
            files.append(p.relative_to(out).as_posix())
        res[bar] = files
    return res


def _write_plots(out: Path, problems: list[int], score: Score, leadsheet: dict | None,
                 synth: Path, harmonic: Path) -> dict[int, str]:
    if not leadsheet or not harmonic.exists() or not problems:
        return {}
    import librosa
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def load(wav: Path):
        y, sr = librosa.load(str(wav), sr=22050, mono=True)
        c = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
        return c, librosa.frames_to_time(np.arange(c.shape[1]), sr=sr, hop_length=512)

    ref_c, ref_t = load(harmonic)
    syn_c, syn_t = load(synth)
    folder = out / "plots"
    folder.mkdir(exist_ok=True)
    res: dict[int, str] = {}
    n = min(len(score.bars), len(leadsheet["bars"]))
    for bar in problems[:MAX_SNIPPETS]:
        i = bar - 1
        if i >= n:
            continue
        lb, sb = leadsheet["bars"][i], score.bars[i]
        panels = [
            (ref_c, ref_t, lb["t_start"], lb["t_end"], "original (harmonic stem)"),
            (syn_c, syn_t, score.seconds(sb.start), score.seconds(sb.start + sb.length), "arrangement (synth)"),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
        for ax, (c, t, a, b, name) in zip(axes, panels):
            m = (t >= a) & (t < b)
            ax.imshow(c[:, m], aspect="auto", origin="lower", cmap="magma")
            ax.set_title(name)
            ax.set_yticks(range(12))
            ax.set_yticklabels(theory.PITCH_CLASSES)
        fig.suptitle(f"bar {bar} chroma")
        p = folder / f"bar-{bar:03d}.png"
        fig.savefig(p, dpi=90, bbox_inches="tight")
        plt.close(fig)
        res[bar] = p.relative_to(out).as_posix()
    return res


# ---------- history ----------

def _last_row(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return rows[-1] if rows else None


def _append_history(path: Path, iter_name: str, s: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if new:
            w.writeheader()
        row = {"iter": iter_name, "time": dt.datetime.now().isoformat(timespec="seconds")}
        row.update({k: ("" if s.get(k) is None else s[k]) for k in HISTORY_FIELDS[2:]})
        w.writerow(row)


def _num(v):
    if v in (None, ""):
        return None
    if v in ("True", "False"):
        return v == "True"
    return float(v)


# ---------- 报告 ----------

def _fmt(v, pct: bool = False) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✅" if v else "❌"
    if isinstance(v, float):
        return f"{v:.1%}" if pct else f"{v:.3f}"
    return str(v)


def render_report(arr: str, iter_name: str, s: dict, prev: dict | None, violations: list[Violation],
                  bars: list[metrics.BarScore], problems: list[int], snippets: dict[int, list[str]],
                  plots: dict[int, str], score: Score) -> str:
    failed = [k for k, ok in s["checks"].items() if not ok]
    status = "✅ 达标" if s["pass"] else "❌ 未达标：" + "、".join(failed)
    lines = [f"# {ARR_NAMES[arr]}校验报告 · {iter_name}", "", f"**结论：{status}**", "",
             "| 指标 | 本轮 | 上一轮 | 达标线 |", "|---|---|---|---|"]
    rows = [("可弹性违规", "violations", False, "0"),
            ("和弦一致率", "chord_agreement", True, f"≥ {s['chord_target']:.0%}")]
    if arr == "finger":
        rows += [("旋律 F1", "melody_f1", False, f"≥ {s['melody_target']:.2f}"),
                 ("旋律精确率", "melody_precision", True, ""),
                 ("旋律召回率（骨架音）", "melody_recall", True, ""),
                 ("八度完全正确", "melody_octave_exact", True, "")]
    rows.append(("和弦标记不符", "chord_mark_issues", False, "0" if arr == "strum" else "仅报告"))
    rows.append(("和声相似度（均值）", "harmonic_similarity", False, "仅用于定位"))
    if arr == "strum":
        rows.append(("节奏相关度（均值）", "rhythm_correlation", False, "仅报告"))
    rows.append(("低置信参考拍", "low_conf_beats", False, "转人耳确认"))
    for name, key, pct, target in rows:
        pv = _num(prev.get(key)) if prev and key in prev else None
        lines.append(f"| {name} | {_fmt(s[key], pct)} | {_fmt(pv, pct)} | {target} |")

    lines += ["", f"## 可弹性违规（{len(violations)}）"]
    lines += [f"- 第 {v.bar} 小节 · 拍 {v.start - score.bars[v.bar - 1].start + 1:.2f} · `{v.rule}`：{v.detail}"
              for v in violations] or ["- 无"]

    by_bar = {b.bar: b for b in bars}
    lines += ["", f"## 问题小节（{len(problems)}）"]
    if not problems:
        lines.append("- 无")
    for bar in problems:
        b = by_bar.get(bar)
        sec = f" [{b.section}]" if b and b.section else ""
        lines += ["", f"### 第 {bar} 小节{sec}"]
        if b:
            if b.chord_ratio is not None:
                lines.append(f"- 和弦一致 {b.chord_agree}/{b.chord_total}")
                lines += [f"  - {x}" for x in b.chord_issues]
            if b.mel_f1 is not None:
                lines.append(f"- 旋律 F1 {b.mel_f1:.2f}（命中 {b.mel_hit}/{b.mel_est}，骨架覆盖 {b.mel_found}/{b.mel_ref}）")
                lines += [f"  - {x}" for x in b.mel_issues]
            lines += [f"- 和弦标记不符：{x}" for x in b.mark_issues]
            if b.harmonic_sim is not None:
                lines.append(f"- 和声相似度 {b.harmonic_sim:.2f}")
        lines += [f"- 违规 `{v.rule}`：{v.detail}" for v in violations if v.bar == bar]
        if snippets.get(bar):
            lines.append("- 试听：" + " ↔ ".join(snippets[bar]))
        if plots.get(bar):
            lines.append(f"- 色度图：{plots[bar]}")

    low = [b for b in bars if b.low_conf_beats]
    lines += ["", f"## 需人耳确认（参考和弦低置信，共 {s['low_conf_beats']} 拍）"]
    lines += [f"- 第 {b.bar} 小节：拍 {', '.join(map(str, b.low_conf_beats))}" for b in low] or ["- 无"]
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: 实现 `guitarize/cli.py`**

```python
"""guitarize 命令行入口。"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_song_config
from .paths import REPO_ROOT, SongPaths

ARRS = ("strum", "finger")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="guitarize", description="原曲 → 吉他谱（弹唱 / 指弹）+ 自动校验")
    ap.add_argument("--songs-dir", type=Path, default=None, help="songs 目录（默认仓库内 songs/）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_text in (("render", "生成交付物：gp / txt / pdf / html"),
                            ("verify", "构建 + 合成 + 可弹性 + 评分 → 报告")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("slug")
        p.add_argument("--arr", choices=ARRS, required=True)
    p = sub.add_parser("serve", help="从仓库根目录起本地 HTTP 服务，用于打开交互谱")
    p.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)

    if args.cmd == "serve":
        return _serve(args.port)
    paths = SongPaths.for_slug(args.slug, args.songs_dir)
    cfg = load_song_config(paths)
    if args.cmd == "render":
        from .render.pipeline import render_all

        for kind, path in render_all(paths, cfg, args.arr).items():
            print(f"{kind}: {path}")
    elif args.cmd == "verify":
        from .verify.report import run_verify

        out = run_verify(paths, cfg, args.arr)
        print((out / "report.md").read_text(encoding="utf-8"))
        print(f"报告目录：{out}")
    return 0


def _serve(port: int) -> int:
    import functools
    import http.server

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(REPO_ROOT))
    print(f"http://localhost:{port}/songs/  （Ctrl+C 停止）")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
    return 0
```

- [ ] **Step 5: 运行，确认通过**

Run: `$PY -m pytest tests/test_verify_cli.py -v`
Expected: 3 passed

- [ ] **Step 6: 全量测试**

Run: `$PY -m pytest -v`
Expected: 全部通过（约 47 个）

- [ ] **Step 7: Commit**

```bash
git add guitarize/verify/report.py guitarize/cli.py tests/test_verify_cli.py
git commit -m "feat: verification report with snippets, history and CLI"
```

---

### Task 9: 浏览器验收（主会话执行，非子代理）

**Files:** 无新增（临时目录 `songs/demo/` 验收后删除）

- [ ] **Step 1: 准备临时 demo 歌曲并渲染**

```bash
cd /Users/s4146265/Desktop/Guitar
mkdir -p songs && cp -R tests/fixtures/demo songs/demo && mkdir -p songs/demo/analysis
$PY -c "import json,sys; sys.path.insert(0,'tests'); from conftest import demo_leadsheet; open('songs/demo/analysis/leadsheet.json','w').write(json.dumps(demo_leadsheet()))"
$PY -m guitarize render demo --arr finger
$PY -m guitarize render demo --arr strum
```
Expected: 每次打印 `gp: … txt: … pdf: … html: …` 四行

- [ ] **Step 2: 起服务并在浏览器面板检查**

后台运行 `$PY -m guitarize serve --port 8765`，用浏览器面板打开 `http://localhost:8765/songs/demo/out/finger.html`：
- 截图：五线谱 + 六线谱渲染正常，旋律符干向上、低音符干向下
- 点"▶ 合成音"：状态栏不报错，光标移动
- 读控制台：无报错
- 再打开 `strum.html`：和弦图、扫弦箭头显示正常
- 用 Read 工具查看 `songs/demo/out/finger.pdf` 第 1 页：排版不溢出 A4

- [ ] **Step 3: 清理**

停止 serve 进程，`rm -rf songs/demo`。

## Phase 1 完成标准

1. `$PY -m pytest` 全部通过。
2. `guitarize verify demo --arr finger` 在 demo 上输出"✅ 达标"；人为改错 leadsheet 一个小节的和弦，报告能定位到该小节并给出片段。
3. 浏览器面板中打开 demo 的交互谱：谱面正确渲染、合成音可播放（Task 9）。
