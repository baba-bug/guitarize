"""文本简谱（音高谱表的 txt，如卡祖笛谱）：每小节一行，数字是唱名，拍与拍之间空一格；记法写在文件开头（LEGEND）。
读的是 alphaTab 构建出的谱面模型（与文本六线谱一样），手改的 atex 也照样反映出来。

八度与网页 / PDF 上的简谱相同：不加记号的中音区从主音起一个八度，起点按 alphaTab 的约定（atex.numbered_one，
主音名放在第 4 组）再按谱号八度记号移（8vb 上移一个八度）。卡祖笛谱把旋律最集中的那个八度放在中音区
（arrange/kazoo.display_ottava：なんもねえ 是 8vb，中音区 B3–A#4）。唱名按调号上的音名算（升号调写升号、降号调写降号），
调外音在数字前写 # / b（B 大调里的 G 是 b6）。小调和 alphaTab 一样：1 是关系大调的主音（小调主音是 6），
中音区从小调主音起。"""
from __future__ import annotations

from .. import theory
from ..arrange import atex
from ..verify.model import Bar, Score
from .ascii_tab import _bar_text

LETTERS = "CDEFGAB"
MAJOR_STEPS = (0, 2, 4, 5, 7, 9, 11)      # 大调 1–7 离主音的半音数
GRIDS = (1, 2, 4, 8, 3, 6, 12)            # 一拍平分成几份：先试直拍（整拍、半拍、¼ 拍），再试三连音
TRIPLET = "³"
EPS = 1e-4

LEGEND = """记法：每行一小节，两条竖线之间是 4 拍，拍与拍之间空一格；行末是段落名和和弦名。
  1–7 是唱名（1={do}），数字前的 # / b 是升 / 降半音，0 是休止。
  ' 是高八度、, 是低八度（两个八度写两次）；不加记号的中音区是 {low}–{high}（从主音起的一个八度，与网页 / PDF 上的简谱一致）。
  一拍里写几个记号，这一拍就平分成几份：5 = 一拍，56 = 各半拍，5612 = 各 ¼ 拍。
  - 是前一个音接着响：单独一组 = 再响一拍；拍里的 - 占一份，如 5--6 = ¾ 拍 + ¼ 拍，-0 = 前一个音再响半拍、停半拍。
  {tri} 开头的一组是三连音：这一拍平分成三份（各 ⅓ 拍），如 {tri}5-6 = ⅔ 拍 + ⅓ 拍。"""
SHUFFLE = "Shuffle：成对的半拍（如 56）唱成前长后短（⅔ + ⅓ 拍），三连音照写的唱。"


class _Key:
    """唱名的参照：调号（升降号数）、大小调、谱号八度记号 → 1 的音名、中音区的起点（MIDI）。"""

    def __init__(self, bar: Bar):
        self.flat = bar.key < 0
        self.do = atex.FIFTHS_MAJOR[bar.key + 7]   # 1 的音名（小调是关系大调的主音）
        self.one = atex.numbered_one(bar.key, bar.minor) - atex.OTTAVA_SHIFT.get(bar.ottava, 0)

    def label(self, midi: int) -> str:
        """MIDI → 唱名：升降号 + 数字 + 八度记号（'5'、'#4'、'7,'、"1'"）。"""
        name = theory.pc_name(midi, self.flat)
        digit = (LETTERS.index(name[0]) - LETTERS.index(self.do[0])) % 7 + 1
        acc = (midi - theory.pc(self.do) - MAJOR_STEPS[digit - 1] + 6) % 12 - 6
        octave = (midi - self.one) // 12
        return ("#" * acc if acc > 0 else "b" * -acc) + str(digit) + ("'" * octave if octave > 0 else "," * -octave)


def _segments(score: Score, bar: Bar, key: _Key) -> list[tuple[float, float, str]]:
    """一小节（声部 1）→ [(小节内记谱起点, 终点, 记号)]：起音是唱名，连音线延续是 '-'，休止是 '0'。"""
    out = []
    for e in sorted((e for e in score.events if e.bar == bar.index and e.voice == 0), key=lambda e: e.written_start):
        a = e.written_start - bar.start
        if e.rest or not e.notes:
            mark = "0"
        elif all(n.tie for n in e.notes):
            mark = "-"
        else:
            mark = key.label(max(n.midi for n in e.notes if not n.tie))
        out.append((a, a + e.written_dur, mark))
    return out


def _grid(edges: list[float]) -> int:
    """拍内的边界（拍内偏移，0–1）都落在格上的最少份数（GRIDS 里找不到时取最细的）。"""
    return next((n for n in GRIDS if all(abs(x * n - round(x * n)) < EPS for x in edges)), GRIDS[-1])


def beat_group(segments: list[tuple[float, float, str]], beat: int) -> str:
    """第 beat 拍（0 起）→ 一组记号：这一拍平分成 n 份，每份是从这里起音的唱名 / 休止 0，或接着响的 '-'
    （休止接着的那一份还是 0）；三连音的组前面加 ³。"""
    n = _grid([a - beat for a, _, _ in segments if beat + EPS < a < beat + 1 - EPS])
    slots = []
    for j in range(n):
        t = beat + j / n
        a, _, mark = next((s for s in segments if s[0] - EPS <= t < s[1] - EPS), (t, t, "0"))
        slots.append(mark if abs(a - t) < EPS or mark == "0" else "-")
    return (TRIPLET if n % 3 == 0 else "") + "".join(slots)


def render_jianpu(score: Score) -> str:
    key = _Key(score.bars[0])
    beats = int(round(score.bars[0].length))
    head = f"1={key.do} {beats}/4" + ("（小调：6 是主音）" if score.bars[0].minor else "")
    info = f"{score.title} · ♩ = {score.tempo:g}"
    if any(b.feel == "triplet8th" for b in score.bars):
        info += " · " + SHUFFLE
    legend = LEGEND.format(do=key.do, low=theory.midi_name(key.one, key.flat),
                           high=theory.midi_name(key.one + 11, key.flat), tri=TRIPLET)
    rows = []
    for bar in score.bars:
        segments = _segments(score, bar, key)
        groups = [beat_group(segments, k) for k in range(int(round(bar.length)))]
        section = bar.section or _bar_text(score, bar)
        chords = [e.chord for e in sorted(score.events, key=lambda e: e.written_start)
                  if e.bar == bar.index and e.chord]
        rows.append((bar, groups, " ".join(([f"[{section}]"] if section else []) + chords)))
    width = max((len(g) for _, groups, _ in rows for g in groups), default=1)
    lines = [f"{bar.index + 1:>3} | " + " ".join(g.ljust(width) for g in groups) + " |" + (f"  {tail}" if tail else "")
             for bar, groups, tail in rows]
    return "\n".join([head, info, legend, "", *lines]) + "\n"
