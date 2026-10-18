"""song.yaml 与难度档的加载。"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from .paths import SongPaths

PROFILES_PATH = Path(__file__).with_name("profiles.yaml")
ARR_NAMES = {"strum": "弹唱", "finger": "指弹", "melody": "旋律"}
MELODY_SOURCES = ("lead", "vocals")   # 旋律从哪条分轨识别：主唱 stems/lead.wav（去掉伴唱）、整条人声 stems/vocals.wav
FINGER_STYLES = ("full", "skeleton")  # 指弹写法：完整旋律（默认）/ 骨架旋律，见 arrange/finger.py


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
    melody_line: float = 0.90       # 旋律谱（melody）的旋律 F1 达标线：召回按全部参考音算
    bar_chord_flag: float = 0.75
    bar_melody_flag: float = 0.60
    bar_harmonic_flag: float = 0.60
    bar_vocal_flag: float = 0.60    # 旋律谱：与人声分轨的小节色度相似度低于它 → 问题小节
    melody_conf_flag: float = 0.50  # 旋律谱：参考旋律音的置信（平均周期性）低于它 → 需人耳确认


@dataclass(frozen=True)
class SongConfig:
    """song.yaml 的内容。overrides 是人工覆盖项（分析出错时用人耳纠正，重跑分析后仍然生效）：

    - tempo_bpm / downbeat_shift / feel：beats 步骤（见 analyze/beats.py）
    - key：调（如 B、F#m），不再用色度推断
    - chords：人耳纠正的和弦，在 leadsheet 步骤套用（analyze/leadsheet.py 的 build_leadsheet）；
      不改 chords.json，所以 `--steps leadsheet` 就能生效，重跑 chords 也不会丢。键是 leadsheet 的小节号
      （1 起；int 或数字字符串），值有三种写法，所指小节原有的和弦整个被替换：

          overrides:
            chords:
              19: N                          # 一个标签 = 整小节（N = 无和弦）
              21: E F#                       # 空格分隔的标签均分小节：2 个各 2 拍，4 个各 1 拍；
                                             #   个数必须整除每小节拍数（4/4 里写 3 个会报错）
              16: [C#m7, C#m7, F#, F#]       # YAML 列表 = 逐拍一个标签，长度必须等于每小节拍数

      标签须能被 theory.parse_chord 解析（含转位 C/E 和强力和弦 E5），或写 N / N.C. / NC；相邻同名标签合并成一段。
      写错的小节号（< 1 或超过总小节数）、标签、个数都会 ValueError，并指出小节号。覆盖后的和弦段
      confidence 记为 "manual"：评分时按 high 计（manual 的 N 照例跳过），summary 里标 "!" 并写 [manual]。
    - melody：人耳纠正的旋律，同样在 leadsheet 步骤套用（不改 melody.json，重跑 melody 不会丢）。键是小节号，
      值是这一小节的全部旋律音 [拍, 拍数, 音名]，所指小节原有的旋律整个被替换（对照旋律谱的音高图 / 人声试听来写）：

          overrides:
            melody:
              1: [[2.67, 0.33, F#4], [3, 1, F#4], [4, 0.33, B3], [4.33, 1.33, F#4]]
              19: []                         # 这一小节没有人声旋律

      拍从 1 起（小节内），拍数是时值；两者都就近取到 1/12 拍，所以 ⅓（2.67、4.33）和 ¼（1.25）都能写。
      音名带八度（F#4、Bb3；C4 = 中央 C）。写的音 conf 记 1.0、按调标 out_of_key、带 "manual": true
      （校验不再把它列进"需人耳确认"），summary 里该小节旋律末尾标 "!"。写错的小节号、音名、
      非正的时值、拍不在小节内、同一小节里音重叠都会 ValueError 并指出小节号；全部校验完才套用。
    - melody_octave：指弹 / 旋律谱的旋律八度（arrange/capo.py 的 octave_plan）。默认 "song"：整曲一个移位，
      保持人声原有的高低关系；"section"：按段落各自选；整数（12 的倍数）：整曲固定移；{小节: 移位}：从该小节起移。
    - capo：指定变调夹（arrange）
    - finger_style：指弹写法（arrange/finger.py）。默认 "full"：旋律谱的每一个音 + 低音 + 合音，verify 按全部参考音
      评旋律；"skeleton"：旧的骨架旋律 + 低音 / 填充（适合初级），verify 只考骨架音
    - finger_bass：完整旋律指弹第 3 拍（和弦没换时）的低音。默认 "root"：再弹根音（摇滚贝斯多半一直弹根音，
      なんもねえ 原曲贝斯第 3 拍 7/16 是根音、5/16 是五音）；"alternate"：根-五交替低音（民谣 / 乡村味）
    - strum_pattern：弹唱谱整首用同一个节奏型（arrange/patterns.py 的 PATTERNS 名：whole / half / quarter / push / pop /
      eighth）。不写时按段落取与原曲鼓 + 吉他起音最像的型；简单版用 "quarter"（每拍一下下扫）或 "half"
    - melody_source：旋律从哪条音频识别。默认 "lead"：analyze 的 lead 步骤从人声分轨分出的主唱（伴唱分离模型
      + 居中掩码；模型不可用时只用居中掩码，见 stems.separate_lead）；"vocals"：整条人声。melody 步骤把用的分轨
      记进 melody.json 的 source（lead 时还有 lead_method），旋律谱校验对照同一条（改了这一项要重跑 --steps melody）

    language（不在 overrides 里）：歌词语言 en / ja / yue / zh，melody 步骤传给 GAME 分音（analyze/game.py）；
    不写则不传（GAME 不按语言分音）。改了要重跑 --steps melody。
    """
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
    language: str | None = None                    # 歌词语言（GAME 分音用）
    thresholds: Thresholds = Thresholds()


def melody_source(value, where: str = "overrides.melody_source") -> str:
    """旋律识别用的分轨名（MELODY_SOURCES 之一），None 取默认 "lead"。analyze 的 melody 步骤（song.yaml 的
    overrides.melody_source）与 verify 选旋律谱对照的分轨（melody.json 的 source）共用这一份校验；where 写进报错。"""
    if value is None:
        return "lead"
    if value not in MELODY_SOURCES:
        raise ValueError(f'{where} must be "lead" or "vocals", got {value!r}')
    return value


def strum_pattern(overrides: dict) -> str | None:
    """overrides.strum_pattern：弹唱谱整首用的节奏型名（PATTERNS 里的），没写是 None（按段落自动选）。"""
    from .arrange.patterns import PATTERNS

    value = overrides.get("strum_pattern")
    if value is not None and value not in PATTERNS:
        raise ValueError(f"overrides.strum_pattern must be one of {', '.join(PATTERNS)}, got {value!r}")
    return value


FINGER_BASSES = ("root", "alternate")


def finger_bass(overrides: dict) -> str:
    """overrides.finger_bass（默认 "root"）：完整旋律指弹第 3 拍的低音是根音还是五音（交替低音）。"""
    value = overrides.get("finger_bass", "root")
    if value not in FINGER_BASSES:
        raise ValueError(f'overrides.finger_bass must be "root" or "alternate", got {value!r}')
    return value


def finger_style(overrides: dict) -> str:
    """overrides.finger_style（默认 "full"）。arrange 写指弹、verify 评指弹旋律共用这一份，评法跟着写法走。"""
    value = overrides.get("finger_style", "full")
    if value not in FINGER_STYLES:
        raise ValueError(f'overrides.finger_style must be "full" or "skeleton", got {value!r}')
    return value


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
