"""指弹编配草稿，指法由 Viterbi 分配（fretting.assign_frets）。两种写法（style）：

- full（默认，"旋律优先"）：voice 1 是旋律谱的每一个音（melody.melody_line：同样的起音、时值、三连音、连音线、
  八度移位，连奏 / 尾滑在同一根弦上时同样写 h / sl / sod / sou；只有指法可以不同，让左手按着和弦指型）。
  voice 2 是低音与合音：第 1 拍弹和弦的低音（斜杠和弦弹斜杠低音），第 3 拍换和弦弹新低音、不换弹五音
  （交替低音，Shuffle 的 boom-chick；五音放不下换根音），低音响到下一个低音；第 2、4 拍旋律没有正好在拍上起音时
  弹两个和弦内音的短音（STAB：Shuffle 一个三连音八分），在低音之上、比附近的旋律至少低 3 个半音；
  拍上 / 摇摆反拍起音、唱满 1 拍的旋律长音下面加一个低 3–9 个半音的和弦内音（双音，优先三度、六度）。
  合音一律可丢：Viterbi 里放不下就不取，自检违规时先丢合音，旋律和低音从不为合音让路；旋律的音高、时值也从不修改。
- skeleton：旧的骨架旋律（≤ melody_max_per_beat 个 / 拍、短音丢掉、长音最长 2 拍）+ 第 1 / 3 拍的低音与填充。
  Shuffle 下旋律按原唱的三连音格记谱：有 +1/3 起音的拍整拍写成三连音（{tu 3}），其余拍照旧写成摇摆八分对。

给出自检（与 verify 同一套可弹性检查）时按违规逐轮修补：截短仍按着的音、去掉填充（合音）/ 低音；
换把找不到 t 时可修的音时，改在前一个起音处修，最后才动旋律音的位置（skeleton 移八度，full 只换弦、不改音高）。
每处指法降级与修补都写进该小节的 `// bar N` 注释，方便审阅。"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from .. import theory
from ..config import FINGER_STYLES, Profile
from ..skeleton import skeleton_mask
from ..verify.playability import Violation
from . import atex
from .capo import OctavePolicy, effective_profile, octave_plan
from .fretting import FretEvent, assign_frets
from .layout import _at, _voice_lines, annotate, section_bars, shape_marks
from .melody import SLIDE_OUT, Sung, note_effects
from .melody import melody_line as sung_line
from .shapes import OPEN_STRINGS, voicings
from .strum import beat_chords

BASS_LOW = 40        # E2：低音区间 E2–D#3（指法调）
MAX_MELODY_NOTE = 2.0
FILL_GAP = 3         # 填充音至少比它持续期间的最低旋律音低 3 个半音
REPAIR_ROUNDS = 10
HAND_RULES = ("span", "fingers", "shift")   # 校验把仍按着的音算进手型的规则：可以靠截短 / 去音修补
EPS = 1e-6
STAB = {"shuffle": 1 / 3, "straight": 0.5}   # 第 2、4 拍和弦短音的记谱时值：Shuffle 一个三连音八分，直拍一个八分
STAB_TAKE, STAB_CANDIDATES = 2, 4            # 短音取 2 个和弦内音，给 Viterbi 的候选
DOUBLE_GAP = (3, 9)                          # 双音比旋律音低 3–9 个半音
SWEET = (3, 4, 8, 9)                         # 双音优先三度、六度
DOUBLE_CANDIDATES = 3
HARMONY_TOP = 64                             # 附近没有旋律时合音不高于 E4（与填充一样）


def written_position(perf: float, feel: str) -> float:
    """演奏位置（拍）→ 记谱位置（八分格）。Shuffle：2/3 → 反拍八分，1/3 → 拍点；直拍取最近的八分。
    （Shuffle 的骨架旋律不用它：+1/3 写成三连音，见 skeleton_notes。）"""
    whole = int(np.floor(perf + 1e-6))
    frac = perf - whole
    if feel == "shuffle":
        if frac < 1 / 2:
            return float(whole)
        if frac < 5 / 6:
            return whole + 0.5
        return float(whole + 1)
    return round(perf * 2) / 2


def performed_position(pos: float, feel: str) -> float:
    """记谱位置 → 演奏位置（与 alphaTab 一致）：Shuffle 的反拍八分（+1/2，总是八分对的后一个，见 atex.split_for_swing）
    在 +2/3 演奏；拍点、三连音位置（k/3）照记谱演奏；直拍不变。"""
    if feel == "shuffle" and abs(pos % 1 - 0.5) < EPS:
        return pos + 1 / 6
    return pos


def triplet_slot(perf: float) -> tuple[int, int]:
    """Shuffle 的演奏位置（拍）→ (拍, 三连音格 k)：k = 0 拍点、1 = +1/3、2 = +2/3。取最近的格，
    分界与 written_position 一致（+1/2 归 +2/3，5/6 以后归下一拍）。"""
    n = int(np.floor(perf * 3 + 0.5 + 1e-6))
    return n // 3, n % 3


def _keep_per_beat(slots: list[tuple[float, float, int]], limit: int) -> list[tuple[float, float, int]]:
    """同一拍的骨架音 [(位置, 演唱时值, 音高)]（按位置排序）超过 limit 个时：留拍点上的音和唱得最长的音（一样长取靠前的）。"""
    if len(slots) <= limit:
        return slots
    beat = int(slots[0][0] + EPS)
    rank = sorted(slots, key=lambda x: (abs(x[0] - beat) > EPS, -x[1], x[0]))
    return sorted(rank[:limit])


def skeleton_notes(leadsheet: dict, profile: Profile) -> list[tuple[float, int]]:
    """骨架旋律 → [(记谱位置, 音高)]。
    Shuffle：音落在三连音格上（拍点、+1/3、+2/3）。某拍留下的音里有 +1/3 的，整拍写成三连音（记谱 = 演奏 = k/3，
    如 B3（4）后紧跟进副歌的长音 F#4（4+⅓））；其余拍照旧是摇摆八分对（+2/3 记在反拍八分 +1/2）。
    直拍：取最近的八分；长音与前一个音落在同一个八分位时顺延半拍、两个都保留——只留给直拍
    （Shuffle 的 +1/3 有三连音可写，不必顺延；直拍的 +1/4 十六分音符以后再说）。
    每拍最多 profile.melody_max_per_beat 个：多了留拍点上的音和唱得最长的音。"""
    notes = sorted((m for b in leadsheet["bars"] for m in b["melody"]), key=lambda m: m["start_beats"])
    mask = skeleton_mask([m["start_beats"] for m in notes], [m["dur_beats"] for m in notes])
    feel = leadsheet.get("feel", "straight")
    at: dict[float, tuple[float, int]] = {}   # 位置 → (演唱时值, 音高)；Shuffle 的位置是三连音格 k/3
    for m, keep in zip(notes, mask):
        if not keep:
            continue
        if feel == "shuffle":
            beat, k = triplet_slot(m["start_beats"])
            pos = beat + k / 3
        else:
            pos = written_position(m["start_beats"], feel)
            if pos in at and pos + 0.5 not in at and m["dur_beats"] >= 1 - EPS:
                pos += 0.5   # 直拍：长音与前一个音落在同一个八分位，顺延半拍、两个都保留（短的经过音照旧丢掉）
        at.setdefault(pos, (m["dur_beats"], m["midi"]))
    beats: dict[int, list[tuple[float, float, int]]] = {}
    for pos in sorted(at):
        beats.setdefault(int(pos + EPS), []).append((pos, *at[pos]))
    out = []
    for beat, slots in sorted(beats.items()):
        kept = _keep_per_beat(slots, profile.melody_max_per_beat)
        triplet = any(abs(pos - beat - 1 / 3) < EPS for pos, _, _ in kept)
        for pos, _, midi in kept:
            if feel == "shuffle" and not triplet and abs(pos - beat - 2 / 3) < EPS:
                pos = beat + 0.5   # 不是三连音拍：+2/3 记成反拍八分，alphaTab 按八分对后移到 +2/3 演奏
            out.append((pos, midi))
    return out


def melody_line(leadsheet: dict, capo: int, profile: Profile, octave: OctavePolicy = "song") -> list[tuple[float, int]]:
    """骨架旋律 → (记谱位置, 指法调音高)；八度移位见 capo.octave_plan（默认整曲一个移位，保持人声的高低关系）。
    记谱位置见 skeleton_notes（Shuffle 的三连音拍里是 k/3）。"""
    mel = [(pos, p - capo) for pos, p in skeleton_notes(leadsheet, profile)]
    out = list(mel)
    for idx, shift, _ in octave_plan(mel, section_bars(leadsheet), octave):
        for i in idx:
            out[i] = (mel[i][0], mel[i][1] + shift)
    return out


FILL_LOW, FILL_HIGH = 50, 64   # 填充音优先区间 D3–E4


def _parse(label: str) -> tuple[int, str, int | None]:
    m = re.match(r"^([A-G][#b]?)5$", label)
    if m:
        return theory.pc(m.group(1)), "power", None
    return theory.parse_chord(label)


def _bass_and_fill(label: str) -> tuple[int, int]:
    """(低音指法调音高, 填充音 pc)。填充 = 三音；三音已是低音时用根音；强力和弦用五音。"""
    root, quality, bass = _parse(label)
    bass_pc = bass if bass is not None else root
    if quality == "power":
        fill = (root + 7) % 12
    else:
        third = (root + (3 if quality in ("min", "min7", "dim", "min7b5") else 4)) % 12
        fill = root if third == bass_pc else third
    return BASS_LOW + ((bass_pc - BASS_LOW) % 12), fill


def place_fill(pc: int, bass: int, top: int | None) -> int | None:
    """填充音：高于 bass、且比 top（填充持续期间的最低旋律音）至少低 3 个半音，取该音名最高的音，优先落在 50–64。
    无旋律时上限 E4。"""
    limit = 65 if top is None else top - FILL_GAP + 1
    inside = [p for p in range(bass + 1, limit) if p % 12 == pc]
    preferred = [p for p in inside if FILL_LOW <= p <= FILL_HIGH]
    if preferred:
        return max(preferred)
    return max(inside) if inside else None


def melody_floor(melody: list[tuple[float, float, int]], pos: float, dur: float) -> int | None:
    """melody：(位置, 时值, 音高)。[pos, pos + dur) 内发声的最低旋律音（填充音的天花板）；没有则 None。"""
    return min((p for s, d, p in melody if s < pos + dur - EPS and s + d > pos + EPS), default=None)


def bass_events(leadsheet: dict, capo: int, profile: Profile) -> list[tuple[float, float, int, int]]:
    """第 1 拍放低音 + 填充；第 3 拍换和弦则放新和弦的低音 + 填充，和弦不变且难度允许时重复根音 + 填充。"""
    out = []
    for i, b in enumerate(leadsheet["bars"]):
        chords = [c if c == "N" else theory.transpose_label(c, -capo) for c in beat_chords(b)]
        start = i * 4.0
        first, third = chords[0], chords[2]
        if first != "N":
            if third == first and profile.bass_on_beat3:
                out += [(start, 2.0, *_bass_and_fill(first)), (start + 2, 2.0, *_bass_and_fill(first))]
            elif third == first:
                out.append((start, 4.0, *_bass_and_fill(first)))
            else:
                out.append((start, 2.0, *_bass_and_fill(first)))
        if third != "N" and third != first:
            out.append((start + 2, 2.0, *_bass_and_fill(third)))
    return out


def _chord_pcs(label: str) -> set[int]:
    """和弦内音的音名（强力和弦：根音、五音）。"""
    root, quality, _ = _parse(label)
    if quality == "power":
        return {root, (root + 7) % 12}
    return theory.chord_tones(label)


def _low(pc: int) -> int:
    """音名 → 低音区间 E2–D#3 里的音高。"""
    return BASS_LOW + ((pc - BASS_LOW) % 12)


def _fifth(label: str) -> int:
    root, quality, _ = _parse(label)
    return (root + {"dim": 6, "min7b5": 6, "aug": 8}.get(quality, 7)) % 12


def bass_line(leadsheet: dict, capo: int, profile: Profile) -> list[tuple[float, float, int, int | None]]:
    """完整旋律指弹的低音 → [(位置, 时值, 指法调音高, 放不下时换的低音)]。第 1 拍弹和弦的低音（斜杠和弦弹斜杠低音）；
    第 3 拍换和弦弹新和弦的低音；和弦不变且难度允许时弹五音（交替低音），五音放不下换根音（五音就是第 1 拍的低音时
    直接弹根音）。每个低音响到下一个低音（第 3 拍没有低音时响满一小节，第 3 拍无和弦时响到第 3 拍）。"""
    out = []
    for i, b in enumerate(leadsheet["bars"]):
        chords = [c if c == "N" else theory.transpose_label(c, -capo) for c in beat_chords(b)]
        start = i * 4.0
        first, third = chords[0], chords[2]
        if first != "N":
            low, root = _bass_and_fill(first)[0], _low(_parse(first)[0])
            if third == first and profile.bass_on_beat3:
                fifth = _low(_fifth(first))
                out += [(start, 2.0, low, None),
                        (start + 2, 2.0, root, None) if fifth == low else (start + 2, 2.0, fifth, root)]
            else:
                out.append((start, 4.0 if third == first else 2.0, low, None))
        if third not in ("N", first):
            out.append((start + 2, 2.0, _bass_and_fill(third)[0], None))
    return out


def _shape(label: str, profile: Profile) -> tuple | None:
    """该按的和弦指型（与 shape_marks 同一个：代价最低的指型）的品位；无和弦 / 找不到时 None。"""
    try:
        options = voicings(label, profile) if label != "N" else []
    except ValueError:
        return None
    return options[0].frets if options else None


def _shape_pitches(frets: tuple | None) -> dict[int, int]:
    """指型里发声的音：{音高: 弦}。"""
    return {OPEN_STRINGS[s] + f: s + 1 for s, f in enumerate(frets or ()) if f is not None}


def stab_pitches(label: str, frets: tuple | None, bass: int, ceiling: int | None) -> tuple[int, ...]:
    """第 2、4 拍和弦短音的候选（偏好顺序）：和弦内音，高于 bass、比 ceiling（附近最低的旋律音）至少低 FILL_GAP 个半音，
    附近没有旋律时不高于 E4。指型在 2–4 弦上的音在前，其次指型上别的音，再次指型外的和弦内音；同一类里高的在前。"""
    top = HARMONY_TOP if ceiling is None else ceiling - FILL_GAP
    shape, pcs = _shape_pitches(frets), _chord_pcs(label)
    cands = [p for p in range(bass + 1, top + 1) if p % 12 in pcs]
    return tuple(sorted(cands, key=lambda p: (not 2 <= shape.get(p, 0) <= 4, p not in shape, -p))[:STAB_CANDIDATES])


def double_stop_pitches(label: str, frets: tuple | None, melody: int, bass: int | None) -> tuple[int, ...]:
    """长旋律音下面的双音候选（偏好顺序）：和弦内音，比旋律音低 3–9 个半音、高于 bass；三度、六度在前，
    其次落在指型上的，同一类里高的在前。"""
    lo, hi = DOUBLE_GAP
    shape, pcs = _shape_pitches(frets), _chord_pcs(label)
    cands = [p for p in range(melody - hi, melody - lo + 1) if p % 12 in pcs and (bass is None or p > bass)]
    return tuple(sorted(cands, key=lambda p: (melody - p not in SWEET, p not in shape, -p))[:DOUBLE_CANDIDATES])


def _near_floor(line: list[Sung], t: float, dur: float) -> int | None:
    """t 起的和弦短音附近（前半拍到这一拍结束）最低的旋律音：短音要低于它，旋律听起来才在上面。"""
    return min((n.pitch for n in line if n.pos < t + max(1.0, dur) - EPS and n.pos + n.dur > t - 0.5 + EPS),
               default=None)


def _long_note(n: Sung, feel: str) -> bool:
    """加双音的旋律音：起在拍点或摇摆反拍（Shuffle +⅔、直拍 +½），唱满 1 拍。"""
    frac = n.perf % 1
    offbeat = 2 / 3 if feel == "shuffle" else 0.5
    on_grid = frac < EPS or frac > 1 - EPS or abs(frac - offbeat) < EPS
    return on_grid and performed_position(n.pos + n.dur, feel) - n.perf > 1 - EPS


def _sounding_at(items: list[tuple[float, float, int, int | None]], t: float) -> tuple | None:
    """t 时正在响的低音（t 之前起音）。"""
    return next((x for x in items if x[0] < t - EPS and t < x[0] + x[1] - EPS), None)


def _full_voices(leadsheet: dict, capo: int, profile: Profile, octave: OctavePolicy,
                 feel: str) -> tuple[list[Placed], list[Placed], list[Sung], list[tuple[float, str]]]:
    """完整旋律的两个声部（见模块说明）→ (voice 1, voice 2, 旋律音（与 voice 1 一一对应）, 注释 [(记谱位置, 说明)])。
    每个起音时刻（旋律、低音、短音）一个 FretEvent；还在响的旋律音 / 低音作为接着响的音放进去（held_*），
    所以 Viterbi 不会把低音、合音放到旋律长音的弦上，旋律也尽量和响着的低音一起按得住。"""
    line, dropped = sung_line(leadsheet, capo, octave)
    n_bars = len(leadsheet["bars"])
    labels = [c if c == "N" else theory.transpose_label(c, -capo) for b in leadsheet["bars"] for c in beat_chords(b)]
    shapes = {c: _shape(c, profile) for c in set(labels)}
    bass = bass_line(leadsheet, capo, profile)
    stab = STAB.get(feel, 0.5)
    attacks = {round(n.pos, 6): n for n in line}
    stabs = {round(i * 4.0 + k, 6) for i in range(n_bars) for k in (1, 3)
             if labels[i * 4 + k] != "N" and round(i * 4.0 + k, 6) not in attacks}
    bass_at = {round(x[0], 6): x for x in bass}
    changes = {float(k) for k in range(1, len(labels)) if labels[k] != labels[k - 1]}
    times = sorted(set(attacks) | set(bass_at) | stabs)
    events, plan = [], []
    for t in times:
        n, b = attacks.get(t), bass_at.get(t)
        held = None if n else next((x for x in line if x.pos < t - EPS and t < x.pos + x.dur - EPS), None)
        ring = None if b else _sounding_at(bass, t)
        low = (b or ring or (None, None, None))[2]
        label = labels[min(int(t + EPS), len(labels) - 1)]
        harmony, take = (), 0
        if t in stabs and low is not None:
            harmony, take = stab_pitches(label, shapes[label], low, _near_floor(line, t, stab)), STAB_TAKE
        elif n is not None and label != "N" and _long_note(n, feel):
            harmony, take = double_stop_pitches(label, shapes[label], n.pitch, low), 1
        mel = n or held
        events.append(FretEvent(n.perf if n else performed_position(t, feel), None if mel is None else mel.pitch,
                                low, legato=bool(n and n.legato), slide_out=bool(n and n.slide_out in SLIDE_OUT),
                                harmony=harmony, harmony_max=take, shape=shapes[label],
                                bass_alt=b[3] if b else None, held_melody=held is not None, held_bass=ring is not None))
        plan.append((t, n, b))
    assign_frets(events, profile)
    v1, v2, notes = [], [], list(dropped)
    sounding = None   # 正在响的低音
    for (t, n, b), ev in zip(plan, events):
        if n is not None:
            v1.append(Placed(n.pos, n.dur, [ev.melody_pos]))
            if ev.melody_pos is not None:
                n.pitch = _pitch(ev.melody_pos)   # 降级（melody±12）后的实际指法调音高
        if b is not None:
            sounding = Placed(t, b[1], [None, ev.bass_pos]) if ev.bass_pos else None
            v2 += [sounding] if sounding else []
        elif ev.held_bass and ev.bass_pos is None and sounding is not None and sounding.holds(t):
            sounding.cut(_on_beat_stop(sounding, t))
        if ev.harmony_pos:   # 双音响到旋律音的音尾，遇到下一个短音或换和弦先停
            end = min([x for x in stabs | changes if x > t + EPS] + [t + (n.dur if n else stab)])
            v2.append(Placed(t, (stab if t in stabs else end - t), [*ev.harmony_pos, None]))
        if ev.fallback:
            notes.append((t, f"{ev.fallback} {_at(t)}"))
    return v1, v2, line, notes


def _on_third(x: float) -> bool:
    frac = x % 1
    return abs(frac - 1 / 3) < EPS or abs(frac - 2 / 3) < EPS


def _one_grid(times: Iterable[float], feel: str) -> dict[float, float]:
    """伴奏声部的边界（起音、止音）在同一拍里既有反拍八分（+½）又有三连音位置（+⅓、+⅔）时写不出来（要 1/6 拍）：
    Shuffle 把这一拍的 +½ 记成 +⅔（演奏时间一样，见 performed_position）；直拍把 +⅓、+⅔ 提前到拍点
    （伴奏在直拍里只会在三连音位置止音）。→ {原边界: 记谱边界}，只列要改的。"""
    beats: dict[int, list[float]] = {}
    for x in times:
        beats.setdefault(int(np.floor(x + EPS)), []).append(x)
    out = {}
    for beat, xs in beats.items():
        halves = [x for x in xs if abs(x - beat - 0.5) < EPS]
        thirds = [x for x in xs if _on_third(x)]
        if halves and thirds:
            out.update({x: beat + 2 / 3 for x in halves} if feel == "shuffle" else {x: float(beat) for x in thirds})
    return out


def merged_pieces(items: list[Placed], cuts: Iterable[float] = (),
                  feel: str = "straight") -> list[tuple[float, float, list, bool | frozenset]]:
    """伴奏声部的音（可以互相重叠：低音响着时拍 2 的和弦短音）→ 按时间铺开的片段 [(位置, 时值, [(弦, 品)], 连音线)]，
    在每个起音、止音处与 cuts（要挂指型名的位置）切开；同一拍里八分格与三连音格的边界统一到一种（_one_grid）。
    连音线：True 全是接着响的音，False 全是新起音，frozenset = 接着响的那几根弦（同一拍里低音接着响、合音新起音，
    见 atex.Item.held）。"""
    raw = [(it.pos, it.end(k), w) for it in items for k, w in enumerate(it.where)
           if w is not None and it.end(k) > it.pos + EPS]
    grid = _one_grid([x for a, b, _ in raw for x in (a, b)], feel)
    notes = [(grid.get(a, a), grid.get(b, b), w) for a, b, w in raw]
    notes = [(a, b, w) for a, b, w in notes if b > a + EPS]
    edges: list[float] = []
    for x in sorted([x for a, b, _ in notes for x in (a, b)] + list(cuts)):
        if not edges or x > edges[-1] + EPS:
            edges.append(x)
    out = []
    for a, b in zip(edges, edges[1:]):
        on = [(s, w) for s, e, w in notes if s < b - EPS and e > a + EPS]
        if not on:
            continue
        held = frozenset(w[0] for s, w in on if s < a - EPS)
        out.append((a, b - a, sorted(w for _, w in on), True if len(held) == len(on) else held or False))
    return out


def _starts_beat(pieces: list[tuple[float, float, list, bool]], x: float) -> bool:
    """声部在 x 处能挂拍效果：x 不在某个发声片段的中间，也不是连音线延续片段的起点（休止会在 x 处切开）。"""
    for pos, dur, notes, tie in pieces:
        if notes and pos - EPS < x < pos + dur - EPS and (pos < x - EPS or tie is True):
            return False
    return True


@dataclass
class Placed:
    """声部里的一个音 / 和弦（记谱位置、时值，拍）。where：voice 1 为 [旋律]；voice 2 最后一个是低音，前面是填充 / 合音
    （skeleton [填充, 低音]；full 的低音 [None, 低音]、合音 [合音…, None]，可以互相重叠）；None = 不弹。
    stop：where 下标 → 这个音单独提前止音的位置（其余的音接着响，渲染成连音线延续）。moved：修补时已移过位置。"""
    pos: float
    dur: float
    where: list
    stop: dict = field(default_factory=dict)
    moved: bool = False

    def end(self, k: int) -> float:
        return min(self.stop.get(k, self.pos + self.dur), self.pos + self.dur)

    def sounding(self, t: float) -> list[int]:
        """起音早于 t、t 时仍在发声的音（where 下标）。"""
        if self.pos >= t - EPS:
            return []
        return [k for k, w in enumerate(self.where) if w is not None and t < self.end(k) - EPS]

    def holds(self, t: float) -> bool:
        return bool(self.sounding(t))

    def cut(self, t: float) -> None:
        """整体截止到 t。"""
        self.dur = t - self.pos

    def pieces(self) -> list[tuple[float, float, list, bool]]:
        """→ [(位置, 时值, [(弦, 品)], 是否连音线延续)]：在单独止音处切开，后面的片段只留还在响的音、写成连音线；
        都不响了写休止。"""
        end = self.pos + self.dur
        notes = [k for k, w in enumerate(self.where) if w is not None]
        edges = sorted({self.pos, end} | {self.end(k) for k in notes if self.pos + EPS < self.end(k) < end - EPS})
        out = []
        for a, b in zip(edges, edges[1:]):
            on = [self.where[k] for k in notes if self.end(k) > a + EPS]
            out.append((a, b - a, on, bool(on) and a > self.pos + EPS))
        return out


def _strings(it: Placed) -> set[int]:
    return {w[0] for w in it.where if w is not None}


def _on_beat_stop(it: Placed, t: float) -> float:
    """伴奏（声部 2）止音点落在拍内 ⅓ / ⅔（三连音位置）时提前到该拍拍点：伴奏本身都在拍上，
    不必为一个止音在低音声部写出三连音和连音线；提前的那 ⅓ 拍反正被同弦 / 更高的旋律音盖住。"""
    whole = float(np.floor(t + EPS))
    frac = t - whole
    if (abs(frac - 1 / 3) < EPS or abs(frac - 2 / 3) < EPS) and whole > it.pos + EPS:
        return whole
    return t


def _cut_note(it: Placed, k: int, t: float) -> None:
    """只让 where[k] 在 t 止音（伴奏的三连音位置提前到拍点）；剩下的音都比它先停时整体截短。"""
    it.stop[k] = _on_beat_stop(it, t)
    it.dur = max(it.end(j) for j, w in enumerate(it.where) if w is not None) - it.pos
    it.stop = {j: e for j, e in it.stop.items() if e < it.pos + it.dur - EPS}


def _name(it: Placed, k: int, full: bool) -> str:
    """voice 2 的 where[k] 叫什么：最后一个是低音，其余是填充（skeleton）/ 合音（full）。"""
    return "bass" if k == len(it.where) - 1 else "harmony" if full else "fill"


def cut_same_string(v1: list[Placed], v2: list[Placed], full: bool = False) -> list[tuple[float, str]]:
    """同一根弦不能一边按住一边再弹：低音 / 填充持续期间有旋律音弹在它的弦上 → 只截这根弦上的音（"fill cut" /
    "bass cut"），另一个音接着响（写成连音线延续）；旋律音持续期间有低音 / 填充弹在它的弦上：
    是填充 → 去掉该填充（旋律 > 低音 > 填充，进副歌的长音不能被填充截断）；是低音 → 旋律截止到该处。
    full（完整旋律）：填充叫合音；旋律的时值不改，长音的弦上来了低音就去掉低音；伴奏声部自己的音（低音、合音互相重叠）
    弹在还在响的音的弦上 → 截短先响的那个。返回 (记谱位置, 说明)。"""
    out = []
    for it in v2:
        for k, w in enumerate(it.where):
            hit = None if w is None else min((o.pos for o in v1 if k in it.sounding(o.pos) and w[0] in _strings(o)),
                                             default=None)
            if hit is not None:
                _cut_note(it, k, hit)
                out.append((hit, f"{_name(it, k, full)} cut {_at(hit)} (same string)"))
    for it in v1:
        mine = _strings(it)
        for o in sorted(v2, key=lambda o: o.pos):
            for k, w in enumerate(o.where):
                if it.holds(o.pos) and w is not None and w[0] in mine and (full or k < len(o.where) - 1):
                    o.where[k] = None
                    out.append((o.pos, f"drop {_name(o, k, full)} {_at(o.pos)} (same string)"))
        hit = min((o.pos for o in v2 if it.holds(o.pos) and mine & _strings(o)), default=None)
        if hit is not None:
            it.cut(hit)
            out.append((hit, f"melody cut {_at(hit)} (same string)"))
    for it in v2 if full else []:
        for k, w in enumerate(it.where):
            hit = None if w is None else min((o.pos for o in v2 if k in it.sounding(o.pos) and w[0] in _strings(o)),
                                             default=None)
            if hit is not None:
                _cut_note(it, k, hit)
                out.append((hit, f"{_name(it, k, full)} cut {_at(hit)} (same string)"))
    return sorted(out)


def _pitch(w: tuple[int, int]) -> int:
    return OPEN_STRINGS[w[0] - 1] + w[1]


def cut_crossings(v1: list[Placed], v2: list[Placed], full: bool = False) -> list[tuple[float, str]]:
    """填充本来按整段持续期间的最低旋律音放低至少 3 个半音；但指法降级（如 melody-12）会把后面的旋律压下来。
    低音 / 填充（合音）持续期间有旋律音不比它高出 3 个半音 → 截止到这个旋律音。返回 (记谱位置, 说明)。"""
    out = []
    for it in v2:
        hit = min((o.pos for o in v1 if o.where[0] is not None
                   and any(_pitch(o.where[0]) - _pitch(it.where[k]) < FILL_GAP for k in it.sounding(o.pos))),
                  default=None)
        if hit is not None:
            it.cut(_on_beat_stop(it, hit))
            name = "bass" if it.where[-1] is not None else _name(it, 0, full)
            out.append((hit, f"{name} cut {_at(hit)} (crossing)"))
    return out


def _fretted(w: tuple[int, int] | None) -> bool:
    return w is not None and w[1] > 0


def _attacked(items: list[Placed], t: float) -> list[Placed]:
    return [it for it in items if abs(it.pos - t) < EPS and any(w is not None for w in it.where)]


def hand(v1: list[Placed], v2: list[Placed], t: float) -> list[int]:
    """t 时手上按着的品（记谱时间），与校验的 _hand_frets 同一模型：t 时新拨的有品音 + 之前起音、仍在发声的有品音，
    新拨的有品音所在的弦上旧音让位。空表示没有把位。"""
    fresh = {w for it in _attacked(v1 + v2, t) for w in it.where if _fretted(w)}
    strings = {s for s, _ in fresh}
    held = {it.where[k] for it in v1 + v2 for k in it.sounding(t)
            if _fretted(it.where[k]) and it.where[k][0] not in strings}
    return sorted(f for _, f in fresh | held)


def previous_hand(v1: list[Placed], v2: list[Placed], t: float) -> float | None:
    """t 之前最近一个手上有按弦音的起音：校验的换把规则拿 t 的把位和它比。"""
    onsets = {it.pos for it in v1 + v2 if it.pos < t - EPS and _attacked([it], it.pos)}
    return next((o for o in sorted(onsets, reverse=True) if hand(v1, v2, o)), None)


def jump(v1: list[Placed], v2: list[Placed], t: float, feel: str) -> int:
    """模型里 t 的换把幅度：t 的把位（最低按弦品）与 1 拍以内前一个把位之差；t 没有把位、或 1 拍内没有前一个把位则为 0。
    "相隔是否不到 1 拍"按演奏时间算（与校验一致）：三连音 +2/3 到下一拍的反拍八分记谱只差 5/6 拍，演奏正好 1 拍。"""
    here, prev = hand(v1, v2, t), previous_hand(v1, v2, t)
    if not here or prev is None or performed_position(t, feel) - performed_position(prev, feel) > 1 - EPS:
        return 0
    return abs(here[0] - hand(v1, v2, prev)[0])


def _cut_held(items: list[Placed], t: float, voice2: bool = False, role: str | None = None) -> bool:
    """截短第一个 t 之前起音、t 时仍有按弦音在响的音 / 和弦（伴奏的三连音位置提前到拍点）。
    role（完整旋律的伴奏）："bass" 只截低音、"harmony" 只截合音；None 整个截短。"""
    for it in items:
        ks = [k for k in it.sounding(t) if _fretted(it.where[k])
              and (role is None or (k == len(it.where) - 1) == (role == "bass"))]
        if ks and role is None:
            it.cut(_on_beat_stop(it, t) if voice2 else t)
        for k in ks if role else []:
            _cut_note(it, k, t)
        if ks:
            return True
    return False


def _drop(v2: list[Placed], t: float, bass: bool) -> bool:
    """去掉 t 时起音的一个按弦伴奏音：bass 时去低音，否则去第一个按弦的填充 / 合音。"""
    for it in v2:
        ks = [len(it.where) - 1] if bass else range(len(it.where) - 1)
        k = next((k for k in ks if _fretted(it.where[k])), None) if abs(it.pos - t) < EPS else None
        if k is not None:
            it.where[k] = None
            return True
    return False


Step = tuple[float, str, Callable[[list[Placed], list[Placed]], bool]]   # (修补位置, 说明, 动作(v1, v2) → 是否做了)


def _steps_at(t: float, full: bool = False) -> list[Step]:
    """t 时的修补顺序：截短仍按着的低音 / 填充 → 截短仍按着的旋律 → 去掉 t 时起音的填充 → 去掉 t 时起音的低音。
    full（完整旋律）：先动合音（去掉 t 时起音的合音 → 截短仍按着的合音），再截短仍按着的低音、去掉 t 时起音的低音；
    旋律的时值不改。"""
    if full:
        return [(t, "drop harmony", lambda v1, v2: _drop(v2, t, False)),
                (t, "harmony cut", lambda v1, v2: _cut_held(v2, t, True, "harmony")),
                (t, "bass cut", lambda v1, v2: _cut_held(v2, t, True, "bass")),
                (t, "drop bass", lambda v1, v2: _drop(v2, t, True))]
    return [(t, "bass cut", lambda v1, v2: _cut_held(v2, t, True)), (t, "melody cut", lambda v1, v2: _cut_held(v1, t)),
            (t, "drop fill", lambda v1, v2: _drop(v2, t, False)), (t, "drop bass", lambda v1, v2: _drop(v2, t, True))]


def _shift_steps(t: float, prev: float | None, full: bool = False) -> list[Step]:
    """换把的修补顺序：先截短（t 时的两步，再是前一个手上有音的起音 prev 时仍按着的低音 / 填充），
    再去音（t 起音的填充、低音，然后 prev 起音的填充、低音）。
    full：t 与 prev 的合音（去掉、截短）都在前，再截短低音（t、prev）、去掉低音（t、prev）。"""
    at_t = _steps_at(t, full)
    if prev is None:
        return at_t
    if full:
        at_prev = _steps_at(prev, True)
        return at_t[:2] + at_prev[:2] + [at_t[2], at_prev[2], at_t[3], at_prev[3]]
    return (at_t[:2] + [(prev, "bass cut", lambda v1, v2: _cut_held(v2, prev, True))] + at_t[2:]
            + [(prev, "drop fill", lambda v1, v2: _drop(v2, prev, False)),
               (prev, "drop bass", lambda v1, v2: _drop(v2, prev, True))])


def _repair_at(v1: list[Placed], v2: list[Placed], t: float, full: bool = False) -> str | None:
    return next((name for _, name, act in _steps_at(t, full) if act(v1, v2)), None)


def _melody_room(it: Placed, w: tuple[int, int], v2: list[Placed]) -> bool:
    """旋律音换到 w 后，它发声期间响着的低音 / 填充不和它同弦、且都比它低至少 FILL_GAP 个半音。"""
    start, end = it.pos, it.pos + it.dur
    for o in v2:
        for k, x in enumerate(o.where):
            if x is not None and o.pos < end - EPS and o.end(k) > start + EPS:
                if x[0] == w[0] or _pitch(w) - _pitch(x) < FILL_GAP:
                    return False
    return True


def _move_melody(v1: list[Placed], v2: list[Placed], t: float, prev: float,
                 max_fret: int, shifts: tuple[int, ...] = (12, -12)) -> tuple[float, str] | None:
    """(b) 最后一招：t 与 prev 之间的换把只涉及两个旋律音（没有低音 / 填充）时，把其中一个移八度（shifts：±12），
    选让跳动缩得最多的那个（落到空弦则没有把位），品位须在 max_fret 以内。每个音最多移一次。返回 (位置, 说明)。
    shifts = (0,)（完整旋律：音高不能改）：只给它换一根弦（"refret"）。"""
    here, there = hand(v1, v2, t), hand(v1, v2, prev)
    b = next((it for it in _attacked(v1, t) if _fretted(it.where[0])), None)
    a = next((it for it in v1 if _fretted(it.where[0]) and (abs(it.pos - prev) < EPS or it.holds(prev))), None)
    if a is None or b is None or here != [b.where[0][1]] or there != [a.where[0][1]]:
        return None
    old = abs(here[0] - there[0])
    best = None
    for it, other in ((b, a), (a, b)):
        if it.moved:
            continue
        for shift in shifts:
            pitch = _pitch(it.where[0]) + shift
            for s in range(1, 7):
                w = (s, pitch - OPEN_STRINGS[s - 1])
                if w == it.where[0] or not 0 <= w[1] <= max_fret or not _melody_room(it, w, v2):
                    continue
                dist = abs(w[1] - other.where[0][1]) if w[1] > 0 else 0
                key = (dist, max(0, s - 3), w[1])
                if dist < old and (best is None or key < best[0]):
                    best = (key, it, w, shift)
    if best is None:
        return None
    _, it, w, shift = best
    it.where[0], it.moved = w, True
    return it.pos, f"melody{shift:+d}" if shift else "refret"


def _repair_shift(v1: list[Placed], v2: list[Placed], t: float, max_fret: int,
                  touched: set[float], feel: str, full: bool = False) -> tuple[float, str] | None:
    """按 _shift_steps 的顺序，先取第一个能让模型里的跳动（jump）变小的一步；都不能时取第一个做得了的
    （为后面的步骤让路）；什么都做不了时 (b) 移旋律八度（full：给旋律音换弦）。前一个起音本轮已经改过的留到下一轮。"""
    prev = previous_hand(v1, v2, t)
    if prev is not None and prev in touched:
        return None
    old, first = jump(v1, v2, t, feel), None
    for at, name, act in _shift_steps(t, prev, full):
        trial = copy.deepcopy((v1, v2))
        if not act(*trial):
            continue
        if jump(*trial, t, feel) < old:
            act(v1, v2)
            return at, name
        first = first or (at, name, act)
    if first is not None:
        first[2](v1, v2)
        return first[0], first[1]
    return None if prev is None else _move_melody(v1, v2, t, prev, max_fret, (0,) if full else (12, -12))


def attack_positions(v1: list[Placed], v2: list[Placed], feel: str) -> list[tuple[float, float]]:
    """当前渲染出来的起音：[(演奏位置, 记谱位置)]（来自 Placed.pieces，连音线延续不算起音）。"""
    return sorted({(performed_position(a, feel), a) for it in v1 + v2 for a, _, notes, tie in it.pieces()
                   if notes and not tie})


def written_at(perf: float, attacks: list[tuple[float, float]], feel: str) -> float:
    """违规的演奏时间 → 记谱位置：取在这个时刻起音的那个（三连音拍里记谱 = 演奏 = k/3，摇摆反拍 +2/3 → +1/2）；
    没有对得上的起音时按八分格折算（written_position）。"""
    found = next((w for p, w in attacks if abs(p - perf) < 1e-3), None)
    return written_position(perf, feel) if found is None else found


def repair(v1: list[Placed], v2: list[Placed], violations: Iterable[Violation], feel: str,
           max_fret: int, max_shift: int | None = None, full: bool = False) -> list[tuple[float, str]]:
    """按自检违规修补（就地修改 v1 / v2）。Violation.start 是演奏时间，先换回记谱位置 t（本轮修补之前渲染出来的
    起音里在这个时刻起音的那个，见 written_at）。每个 t 每轮只修一处：
    - 跨度 / 手指数：依次 1. 截短 t 之前起音、t 时仍按着的低音 / 填充；2. 截短 t 时仍按着的旋律音；
      3. 去掉 t 时起音的填充；4. 再去掉低音。
    - 换把：同样的顺序，但截短之后先截短前一个手上有音的起音 t′ 时仍按着的低音 / 填充（宁可截短也不丢新起音），
      去音之后再去掉 t′ 起音的填充、低音；优先取能缩小跳动的一步（见 _repair_shift），
      都做不了时 (b) 把 t 或 t′ 的旋律音移八度（melody±12）。
    full（完整旋律）：合音先丢、旋律不截短、不移八度（顺序见 _steps_at / _shift_steps，(b) 改成给旋律音换弦）。
    只动按弦音（空弦不占手型），只修手型类规则。违规来自本轮修补之前的文本：给了 max_shift 时，
    本轮前面的修补已让 t 的跳动 ≤ max_shift 的换把不再修（免得白白去音）。返回 (修补位置, 说明)；空表示无可修。"""
    done, seen, touched = [], set(), set()
    attacks = attack_positions(v1, v2, feel)
    for v in violations:
        t = written_at(v.start, attacks, feel)
        if v.rule not in HAND_RULES or t in seen:
            continue
        seen.add(t)
        if v.rule == "shift":
            if max_shift is not None and jump(v1, v2, t, feel) <= max_shift:
                continue
            found = _repair_shift(v1, v2, t, max_fret, touched, feel, full)
        else:
            what = _repair_at(v1, v2, t, full)
            found = (t, what) if what else None
        if found:
            touched.add(found[0])
            done.append((found[0], f"{found[1]} {_at(found[0])} ({v.rule})"))
    return done


def _skeleton_voices(leadsheet: dict, capo: int, profile: Profile,
                     octave: OctavePolicy) -> tuple[list[Placed], list[Placed], list[tuple[float, str]]]:
    """骨架写法的两个声部 → (voice 1, voice 2, 注释 [(记谱位置, 说明)])。"""
    n_bars = len(leadsheet["bars"])
    mel = melody_line(leadsheet, capo, profile, octave)
    items = []
    for j, (pos, p) in enumerate(mel):
        cap = float(np.floor((pos + MAX_MELODY_NOTE) * 2 + EPS)) / 2   # 最长 2 拍，止在八分格上（三连音位置起的音不另造三连音拍）
        nxt = mel[j + 1][0] if j + 1 < len(mel) else cap
        items.append((pos, min(nxt, cap, n_bars * 4.0) - pos, p))   # 可以跨小节线（连音线）
    bass = bass_events(leadsheet, capo, profile)
    m_at = {pos: p for pos, _, p in items}
    b_at = {pos: (p, place_fill(fill_pc, p, melody_floor(items, pos, dur))) for pos, dur, p, fill_pc in bass}
    starts = sorted(set(m_at) | set(b_at))
    events = assign_frets([FretEvent(s, m_at.get(s), b_at.get(s, (None, None))[0], fill=b_at.get(s, (None, None))[1])
                           for s in starts], profile)
    at = {e.start: e for e in events}
    v1 = [Placed(pos, d, [at[pos].melody_pos]) for pos, d, _ in items]
    v2 = [Placed(pos, d, [at[pos].fill_pos, at[pos].bass_pos]) for pos, d, _, _ in bass]
    return v1, v2, [(e.start, f"{e.fallback} {_at(e.start)}") for e in events if e.fallback]


def _glides(line: list[Sung], v1: list[Placed]) -> dict[float, dict[int, list[str]]]:
    """完整旋律的音符效果（melody.note_effects：同弦连奏 h / sl、尾滑 sod / sou），按 voice 1 当前的指法算。"""
    for n, it in zip(line, v1):
        n.where = it.where[0]
    return note_effects(line)


def generate_finger(leadsheet: dict, capo: int, profile: Profile, title: str,
                    check: Callable[[str], list[Violation]] | None = None, octave: OctavePolicy = "song",
                    style: str = "full") -> str:
    """check：自检（alphaTex → 违规列表）。给出时渲染后按违规修补、重新渲染，最多 10 轮。
    octave：旋律八度策略（song.yaml 的 overrides.melody_octave，见 capo.octave_plan）。
    style：full（默认，完整旋律 + 交替低音 + 合音）/ skeleton（骨架旋律 + 低音与填充），见模块说明。
    谱上标出该按的和弦指型（附和弦图）；旋律 f、伴奏 mf：试听时旋律稍突出（伴奏再轻会让与原曲器乐分轨的和声相似度下降）。"""
    if style not in FINGER_STYLES:
        raise ValueError(f"finger style must be one of {FINGER_STYLES}, got {style!r}")
    profile = effective_profile(profile, capo)
    feel = leadsheet.get("feel", "straight")
    n_bars = len(leadsheet["bars"])
    full = style == "full"
    if full:
        v1, v2, line, found = _full_voices(leadsheet, capo, profile, octave, feel)
    else:
        (v1, v2, found), line = _skeleton_voices(leadsheet, capo, profile, octave), []
    remarks: dict[int, list[str]] = {}

    def remark(found: Iterable[tuple[float, str]]) -> None:
        for t, text in found:
            remarks.setdefault(int(t // 4), []).append(text)

    remark(found)
    remark(cut_same_string(v1, v2, full))
    remark(cut_crossings(v1, v2, full))
    sections = [b.get("section") for b in leadsheet["bars"]]
    marks, diagrams = shape_marks(leadsheet, capo, profile)
    head = atex.header(title, leadsheet["tempo_bpm"], capo, feel, "score tabs", diagrams, key=atex.leadsheet_key(leadsheet))

    def render() -> str:
        pieces1 = [p for it in v1 for p in it.pieces()]
        effects1: dict[float, list[str]] = {0.0: ["dy f"]}
        effects2: dict[float, list[str]] = {0.0: ["dy mf"]}
        for pos, name in marks.items():   # 指型名尽量挂在旋律声部（黑色、醒目）；那里正被连音线延续占着、
            # 或是段落第一拍（段落名与旋律声部的和弦名画在同一行会重叠）时挂在低音声部（另起一行）
            section_start = pos % 4 == 0 and bool(sections[int(pos // 4)])
            target = effects1 if _starts_beat(pieces1, pos) and not section_start else effects2
            target.setdefault(pos, []).insert(0, f"ch {atex.quote(name)}")
        lines1 = _voice_lines(n_bars, pieces1, sections, feel, effects1, note_effects=_glides(line, v1) if full else None)
        lines2 = _voice_lines(n_bars, merged_pieces(v2, effects2, feel), [None] * n_bars, feel, effects2,
                              tie_effects=True)
        return atex.document(head, [annotate(lines1, remarks), lines2])

    text = render()
    for _ in range(REPAIR_ROUNDS if check else 0):
        found = check(text)
        if not found:
            break
        done = repair(v1, v2, found, feel, profile.max_fret, profile.max_shift, full)
        if not done:
            break
        remark(done)
        text = render()
    return text
