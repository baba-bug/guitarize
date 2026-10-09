"""指弹草稿修补逻辑（纯函数，不需要 alphaTab）。"""
from guitarize.arrange.finger import Placed, annotate, cut_crossings, cut_same_string, repair
from guitarize.verify.playability import Violation


def V(start: float, rule: str = "span") -> Violation:
    return Violation(int(start // 4) + 1, start, rule, "x")


def voices():
    v1 = [Placed(0.0, 1.0, [(1, 0)]), Placed(1.0, 2.0, [(1, 5)]), Placed(3.0, 1.0, [(2, 3)])]
    v2 = [Placed(0.0, 2.0, [(2, 0), (4, 1)]), Placed(2.0, 2.0, [(3, 2), (5, 3)])]
    return v1, v2


def test_repair_cuts_held_bass_first():
    v1, v2 = voices()
    assert repair(v1, v2, [V(1.0)], "straight") == [(1.0, "bass cut @2 (span)")]
    assert v2[0].dur == 1.0 and v2[0].where == [(2, 0), (4, 1)]


def test_repair_then_drops_fill_then_bass_then_cuts_melody():
    v1, v2 = voices()
    assert repair(v1, v2, [V(2.0, "shift")], "straight") == [(2.0, "drop fill @3 (shift)")]
    assert v2[1].where == [None, (5, 3)]
    assert repair(v1, v2, [V(2.0, "shift")], "straight") == [(2.0, "drop bass @3 (shift)")]
    assert v2[1].where == [None, None]
    assert repair(v1, v2, [V(2.0)], "straight") == [(2.0, "melody cut @3 (span)")]
    assert v1[1].dur == 1.0
    assert repair(v1, v2, [V(2.0)], "straight") == []      # 无可修：留给审阅


def test_open_strings_are_not_repaired():
    # 空弦不占手型：截短 / 去掉空弦音救不了跨度
    v1 = [Placed(0.0, 4.0, [(1, 0)])]
    v2 = [Placed(0.0, 4.0, [(2, 0), (5, 0)])]
    assert repair(v1, v2, [V(1.0), V(0.0)], "straight") == []


def test_repair_uses_written_time_and_one_fix_per_time():
    v1, v2 = voices()
    # Shuffle：演奏时间 4 + 2/3 是第 2 小节第 1 拍反拍（记谱 4.5）
    v2.append(Placed(4.0, 2.0, [None, (6, 3)]))
    done = repair(v1, v2, [V(4 + 2 / 3), V(4 + 2 / 3, "fingers")], "shuffle")
    assert done == [(4.5, "bass cut @1.5 (span)")] and v2[-1].dur == 0.5


def test_non_hand_rules_are_left_alone():
    v1, v2 = voices()
    assert repair(v1, v2, [V(0.0, "bar_length"), V(1.0, "straight_offbeat")], "straight") == []
    assert v2[0].dur == 2.0


def test_cut_same_string_bass_under_melody():
    v1 = [Placed(0.0, 1.0, [(1, 0)]), Placed(1.0, 1.0, [(4, 3)]), Placed(2.0, 1.0, [(4, 5)])]
    v2 = [Placed(0.0, 4.0, [(3, 0), (4, 2)])]
    assert cut_same_string(v1, v2) == [(1.0, "bass cut @2 (same string)")]
    assert v2[0].dur == 1.0


def test_cut_same_string_melody_over_new_bass():
    v1 = [Placed(0.0, 4.0, [(3, 2)])]
    v2 = [Placed(0.0, 2.0, [None, (6, 3)]), Placed(2.0, 2.0, [(3, 0), (5, 3)])]
    assert cut_same_string(v1, v2) == [(2.0, "melody cut @3 (same string)")]
    assert v1[0].dur == 2.0 and v2[0].dur == 2.0


def test_annotate_appends_to_bar_comment():
    lines = ['// bar 1 [A]\n\\section "A" r.1 |', "// bar 2\nr.1 |"]
    out = annotate(lines, {1: ["drop fill @1 (span)", "bass+12 @3"]})
    assert out[0] == lines[0]
    assert out[1] == "// bar 2 · drop fill @1 (span) · bass+12 @3\nr.1 |"


def test_cut_crossings_when_fallback_drops_melody_under_held_fill():
    # 指法降级把旋律压到 A3，而 B3 填充还在响：低音 + 填充截止到这个旋律音
    v1 = [Placed(0.0, 1.0, [(1, 5)]), Placed(1.0, 1.0, [(3, 2)])]
    v2 = [Placed(0.0, 2.0, [(2, 0), (5, 3)])]
    assert cut_crossings(v1, v2) == [(1.0, "bass cut @2 (crossing)")]
    assert v2[0].dur == 1.0
    # 低 3 个半音以上不算
    v1 = [Placed(0.0, 1.0, [(1, 5)]), Placed(1.0, 1.0, [(2, 3)])]
    v2 = [Placed(0.0, 2.0, [(2, 0), (5, 3)])]
    assert cut_crossings(v1, v2) == [] and v2[0].dur == 2.0
