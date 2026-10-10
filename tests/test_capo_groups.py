from guitarize.arrange.capo import CapoOption, choose_capo, melody_groups, octave_plan
from guitarize.config import load_profile

MID = load_profile("intermediate")


def test_groups_follow_section_marks():
    # 段落起于第 0、2 小节（0-based）；第 0 段之前没有音
    pos = [0.0, 1.0, 5.0, 8.0, 9.5, 13.0]
    assert melody_groups(pos, [0, 2]) == [[0, 1, 2], [3, 4, 5]]
    # 第一个段落标记之前的音单独成组
    assert melody_groups(pos, [2]) == [[0, 1, 2], [3, 4, 5]]
    assert melody_groups(pos, [1]) == [[0, 1], [2, 3, 4, 5]]


def test_groups_split_phrases_without_sections():
    a = [float(i) for i in range(10)]            # 10 个音，每拍一个
    b = [20.0 + i for i in range(9)]             # 隔 11 拍后 9 个音
    short = [40.0, 41.0, 42.0]                   # 隔 ≥3 拍的 3 个音：并入前一组
    assert melody_groups(a + b + short, []) == [list(range(10)), list(range(10, 22))]
    # 开头的短乐句并入后一组；间隔 < 3 拍不断句
    assert melody_groups([0.0, 1.0] + [10.0 + 2.5 * i for i in range(8)], []) == [list(range(10))]
    assert melody_groups([], []) == []


def test_octave_per_section_with_hysteresis():
    high = [(i, 66 + i % 4) for i in range(8)]               # 已在 G3–D5：不移
    low = [(32 + i, 50 + i % 6) for i in range(8)]           # 太低：整段升八度
    near = [(64 + i, 56 + i % 6) for i in range(10)]         # 单看本段不移更好（中位数近 D4），但沿用 +12 也全在音域内
    plan = octave_plan(high + low + near, [0, 8, 16], policy="section")
    assert [(idx[0], shift) for idx, shift, _ in plan] == [(0, 0), (8, 12), (16, 12)]
    assert plan[2][2] == 0.0
    # 沿用越界超过 0.1 时重新选
    plan = octave_plan(high + low + [(64 + i, 66) for i in range(8)], [0, 8, 16], policy="section")
    assert [shift for _, shift, _ in plan] == [0, 12, 0]


def test_capo_option_has_no_global_melody_shift():
    assert "melody_shift" not in CapoOption.__dataclass_fields__


def test_choose_capo_melody_penalty_uses_section_shifts():
    beats = {"C": 8.0}
    # 第一段太高、第二段太低：整体移八度无解，分段各移一次就都在音域内
    melody = [(float(i), 79) for i in range(8)] + [(32.0 + i, 48) for i in range(8)]
    per_section = next(o for o in choose_capo(beats, MID, melody, section_bars=[0, 8], octave="section") if o.capo == 0)
    global_only = next(o for o in choose_capo(beats, MID, melody, section_bars=[0, 8]) if o.capo == 0)   # 默认整曲统一移位
    assert per_section.cost < global_only.cost


def test_octave_plan_default_keeps_the_whole_song_in_one_octave():
    # 用户反馈：段落之间换八度听起来像"音突然掉下去"——默认整曲一个移位，保持人声原有的高低关系
    high = [(i, 66 + i % 4) for i in range(8)]
    low = [(32 + i, 50 + i % 6) for i in range(8)]
    plan = octave_plan(high + low, [0, 8])
    assert len(plan) == 1 and plan[0][0] == list(range(16)) and plan[0][1] == 0


def test_octave_plan_fixed_and_per_bar_overrides():
    notes = [(float(i), 60) for i in range(8)] + [(32.0 + i, 50) for i in range(8)]
    assert [s for _, s, _ in octave_plan(notes, [], policy=12)] == [12]
    plan = octave_plan(notes, [], policy={9: 12})          # 从第 9 小节（第 32 拍）起升八度，之前不移
    assert [(idx[0], s) for idx, s, _ in plan] == [(0, 0), (8, 12)]
    import pytest
    for bad in ("sideways", 7, {0: 12}, {3: 5}):
        with pytest.raises(ValueError):
            octave_plan(notes, [], policy=bad)
