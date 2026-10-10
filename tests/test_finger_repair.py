"""指弹草稿修补逻辑（纯函数，不需要 alphaTab）。"""
import pytest

from guitarize.arrange.finger import Placed, annotate, cut_crossings, cut_same_string, hand, jump, repair
from guitarize.verify.playability import Violation


def V(start: float, rule: str = "span") -> Violation:
    return Violation(int(start // 4) + 1, start, rule, "x")


def voices():
    v1 = [Placed(0.0, 1.0, [(1, 0)]), Placed(1.0, 2.0, [(1, 5)]), Placed(3.0, 1.0, [(2, 3)])]
    v2 = [Placed(0.0, 2.0, [(2, 0), (4, 1)]), Placed(2.0, 2.0, [(3, 2), (5, 3)])]
    return v1, v2


def test_repair_cuts_held_bass_first():
    v1, v2 = voices()
    assert repair(v1, v2, [V(1.0)], "straight", 12) == [(1.0, "bass cut @2 (span)")]
    assert v2[0].dur == 1.0 and v2[0].where == [(2, 0), (4, 1)]


def test_repair_order_at_t_cuts_held_notes_before_dropping_new_ones():
    # t 时依次：截短仍按着的低音 / 填充 → 截短仍按着的旋律 → 去掉 t 时起音的填充 → 再去掉低音
    v1, v2 = voices()
    v2[0].dur = 3.0                  # 第 1 拍的低音 + 填充按过第 3 拍
    steps = [repair(v1, v2, [V(2.0)], "straight", 12) for _ in range(5)]
    assert steps == [[(2.0, "bass cut @3 (span)")], [(2.0, "melody cut @3 (span)")],
                     [(2.0, "drop fill @3 (span)")], [(2.0, "drop bass @3 (span)")], []]   # 最后无可修：留给审阅
    assert v2[0].dur == 2.0 and v1[1].dur == 1.0 and v2[1].where == [None, None]


def test_held_melody_is_cut_before_a_new_chord_loses_its_bass():
    # 旋律二分音符跨过第 3 拍的换和弦：截短旋律，第 3 拍的低音 + 填充保留
    v1 = [Placed(0.0, 1.0, [(1, 0)]), Placed(1.0, 2.0, [(1, 7)])]
    v2 = [Placed(0.0, 2.0, [None, (6, 0)]), Placed(2.0, 2.0, [(3, 2), (5, 3)])]
    assert repair(v1, v2, [V(2.0)], "straight", 12) == [(2.0, "melody cut @3 (span)")]
    assert v1[1].dur == 1.0 and v2[1].where == [(3, 2), (5, 3)]


def test_open_strings_are_not_repaired():
    # 空弦不占手型：截短 / 去掉空弦音救不了跨度
    v1 = [Placed(0.0, 4.0, [(1, 0)])]
    v2 = [Placed(0.0, 4.0, [(2, 0), (5, 0)])]
    assert repair(v1, v2, [V(1.0), V(0.0)], "straight", 12) == []


def test_repair_uses_written_time_and_one_fix_per_time():
    v1, v2 = voices()
    # Shuffle：演奏时间 4 + 2/3 是第 2 小节第 1 拍反拍（记谱 4.5）
    v2.append(Placed(4.0, 2.0, [None, (6, 3)]))
    done = repair(v1, v2, [V(4 + 2 / 3), V(4 + 2 / 3, "fingers")], "shuffle", 12)
    assert done == [(4.5, "bass cut @1.5 (span)")] and v2[-1].dur == 0.5


def test_non_hand_rules_are_left_alone():
    v1, v2 = voices()
    assert repair(v1, v2, [V(0.0, "bar_length"), V(1.0, "straight_offbeat")], "straight", 12) == []
    assert v2[0].dur == 2.0


def test_cut_same_string_bass_under_melody():
    # 旋律弹在低音的弦上：只截低音，填充（3 弦）接着响，写成连音线延续
    v1 = [Placed(0.0, 1.0, [(1, 0)]), Placed(1.0, 1.0, [(4, 3)]), Placed(2.0, 1.0, [(4, 5)])]
    v2 = [Placed(0.0, 4.0, [(3, 0), (4, 2)])]
    assert cut_same_string(v1, v2) == [(1.0, "bass cut @2 (same string)")]
    assert v2[0].dur == 4.0 and v2[0].sounding(2.0) == [0]
    assert v2[0].pieces() == [(0.0, 1.0, [(3, 0), (4, 2)], False), (1.0, 3.0, [(3, 0)], True)]


def test_cut_same_string_fill_only_keeps_bass_ringing():
    v1 = [Placed(0.0, 1.0, [(1, 0)]), Placed(1.0, 1.0, [(2, 1)])]
    v2 = [Placed(0.0, 2.0, [(2, 0), (4, 1)])]
    assert cut_same_string(v1, v2) == [(1.0, "fill cut @2 (same string)")]
    assert v2[0].dur == 2.0 and v2[0].holds(1.5) and v2[0].sounding(1.5) == [1]
    assert v2[0].pieces() == [(0.0, 1.0, [(2, 0), (4, 1)], False), (1.0, 1.0, [(4, 1)], True)]


def test_cut_same_string_fill_then_bass():
    # 填充先在第 2 拍被截，低音再在第 2 拍反拍被截：整个和弦到 1.5 为止
    v1 = [Placed(0.0, 1.0, [(1, 0)]), Placed(1.0, 0.5, [(2, 1)]), Placed(1.5, 0.5, [(4, 3)])]
    v2 = [Placed(0.0, 2.0, [(2, 0), (4, 1)])]
    assert cut_same_string(v1, v2) == [(1.0, "fill cut @2 (same string)"), (1.5, "bass cut @2.5 (same string)")]
    assert v2[0].dur == 1.5 and not v2[0].holds(1.5)
    assert v2[0].pieces() == [(0.0, 1.0, [(2, 0), (4, 1)], False), (1.0, 0.5, [(4, 1)], True)]


def test_cut_same_string_lone_fill_cuts_the_item():
    v1 = [Placed(0.0, 1.0, [(1, 0)]), Placed(1.0, 1.0, [(2, 1)])]
    v2 = [Placed(0.0, 2.0, [(2, 0), None])]
    assert cut_same_string(v1, v2) == [(1.0, "fill cut @2 (same string)")]
    assert v2[0].dur == 1.0 and v2[0].pieces() == [(0.0, 1.0, [(2, 0)], False)]


def test_cut_same_string_melody_over_new_fill():
    # 第 3 拍新和弦的填充弹在旋律长音那根弦上：去掉填充，旋律接着响（以前是截旋律）
    v1 = [Placed(0.0, 4.0, [(3, 2)])]
    v2 = [Placed(0.0, 2.0, [None, (6, 3)]), Placed(2.0, 2.0, [(3, 0), (5, 3)])]
    assert cut_same_string(v1, v2) == [(2.0, "drop fill @3 (same string)")]
    assert v1[0].dur == 4.0 and v2[1].where == [None, (5, 3)]


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


def test_hand_model_counts_held_notes_like_verify():
    v1 = [Placed(0.0, 1.0, [(1, 0)]), Placed(1.0, 1.0, [(4, 5)]), Placed(2.0, 1.0, [(1, 0)])]
    v2 = [Placed(0.0, 2.0, [(3, 2), (4, 3)])]
    assert hand(v1, v2, 0.0) == [2, 3]
    assert hand(v1, v2, 1.0) == [2, 5]      # 4 弦新拨 5 品，同弦的 3 品让位；3 弦 2 品还按着
    assert hand(v1, v2, 2.0) == []          # 低音 + 填充到第 3 拍为止，空弦旋律不占手型


def test_shift_with_nothing_at_t_cuts_voice2_held_at_previous_attack():
    # (a) t = 2 只有 1 弦 8 品；前一个手上有音的起音是 1.5（空弦旋律，第 1 拍的低音 + 填充 2–3 品还按着）
    v1 = [Placed(0.0, 1.0, [(1, 0)]), Placed(1.5, 0.5, [(2, 0)]), Placed(2.0, 1.0, [(1, 8)])]
    v2 = [Placed(0.0, 2.0, [(3, 2), (5, 3)])]
    assert hand(v1, v2, 1.5) == [2, 3] and hand(v1, v2, 2.0) == [8]
    assert repair(v1, v2, [V(2.0, "shift")], "straight", 12) == [(1.5, "bass cut @2.5 (shift)")]
    assert v2[0].dur == 1.5 and v2[0].where == [(3, 2), (5, 3)]


def test_shift_with_nothing_at_t_drops_fill_then_bass_at_previous_attack():
    v1 = [Placed(1.0, 1.0, [(1, 0)]), Placed(2.0, 1.0, [(1, 8)])]
    v2 = [Placed(1.0, 1.0, [(3, 2), (5, 3)])]
    assert repair(v1, v2, [V(2.0, "shift")], "straight", 12) == [(1.0, "drop fill @2 (shift)")]
    assert repair(v1, v2, [V(2.0, "shift")], "straight", 12) == [(1.0, "drop bass @2 (shift)")]
    assert v2[0].where == [None, None]


def test_shift_between_melody_notes_moves_one_an_octave():
    # (b) 1 弦 1 品 F4 → 1 弦 8 品 C5，没有低音 / 填充参与：C5 降八度到 2 弦 1 品，跳动从 7 品变 0
    v1 = [Placed(0.0, 1.0, [(1, 1)]), Placed(1.0, 1.0, [(1, 8)])]
    assert repair(v1, [], [V(1.0, "shift")], "straight", 12) == [(1.0, "melody-12 @2 (shift)")]
    assert v1[1].where == [(2, 1)] and v1[0].where == [(1, 1)]
    assert repair(v1, [], [V(1.0, "shift")], "straight", 12) == []     # 已经不跳：不再移


def test_shift_moves_whichever_melody_note_shrinks_the_jump():
    # E5（1 弦 12 品）→ F#4（1 弦 2 品）：F#4 移八度最多缩到 3 品；E5 降八度落到 1 弦空弦，跳动消失
    v1 = [Placed(0.0, 1.0, [(1, 12)]), Placed(1.0, 1.0, [(1, 2)])]
    assert repair(v1, [], [V(1.0, "shift")], "straight", 12) == [(0.0, "melody-12 @1 (shift)")]
    assert v1[0].where == [(1, 0)] and v1[1].where == [(1, 2)]


def test_melody_octave_needs_a_fret_in_range_and_room_above_voice2():
    v1 = [Placed(0.0, 1.0, [(1, 1)]), Placed(1.0, 1.0, [(1, 10)])]
    assert repair(v1, [], [V(1.0, "shift")], "straight", 2) == []      # 品位上限 2：两个音都放不下
    assert v1[1].where == [(1, 10)]
    # C4 降到 2 弦会和空弦 B3 填充同弦、且只高一个半音；F4 升八度要 13 品：都不行
    v1 = [Placed(0.0, 1.0, [(1, 1)]), Placed(1.0, 1.0, [(1, 8)])]
    v2 = [Placed(0.0, 2.0, [(2, 0), (6, 0)])]
    assert repair(v1, v2, [V(1.0, "shift")], "straight", 12) == []


def test_shift_cuts_held_bass_at_previous_attack_before_dropping_a_new_bass():
    # 第 1 拍的低音（4 弦 1 品）在空弦旋律 1.5 时还按着，第 3 拍新低音在 5 弦 6 品：
    # 截短前一个起音时仍按着的低音，第 3 拍的低音保留
    v1 = [Placed(1.0, 0.5, [(2, 4)]), Placed(1.5, 1.5, [(1, 0)]), Placed(3.0, 1.0, [(1, 5)])]
    v2 = [Placed(0.0, 2.0, [(2, 0), (4, 1)], stop={0: 1.0}), Placed(2.0, 2.0, [(2, 0), (5, 6)])]
    assert hand(v1, v2, 1.5) == [1] and hand(v1, v2, 2.0) == [6]
    assert repair(v1, v2, [V(2.0, "shift")], "shuffle", 12) == [(1.5, "bass cut @2.5 (shift)")]
    assert v2[0].dur == 1.5 and v2[1].where == [(2, 0), (5, 6)]


def test_shift_prefers_a_step_that_shrinks_the_jump():
    # t = 0.5：1 弦 6 品旋律 + 3 弦 7 品填充（低音空弦），前一个起音只有 5 弦 1 品低音。
    # 去掉 t 的填充不改变把位（还是 6 品），跳过；去掉前一个起音的低音才有用
    v1 = [Placed(0.0, 0.5, [(1, 0)]), Placed(0.5, 0.5, [(1, 6)])]
    v2 = [Placed(0.0, 0.5, [None, (5, 1)]), Placed(0.5, 1.5, [(3, 7), (5, 0)])]
    assert hand(v1, v2, 0.0) == [1] and hand(v1, v2, 0.5) == [6, 7]
    assert repair(v1, v2, [V(0.5, "shift")], "straight", 12) == [(0.0, "drop bass @1 (shift)")]
    assert v2[0].where == [None, None] and v2[1].where == [(3, 7), (5, 0)]


def test_shift_falls_back_to_the_first_step_that_applies():
    # 去掉哪个音都缩不小跳动（低音、填充都在 8–10 品）：按顺序先去掉填充，为后面的步骤让路
    v1 = [Placed(0.0, 1.0, [(1, 1)]), Placed(1.0, 1.0, [(1, 8)])]
    v2 = [Placed(1.0, 1.0, [(3, 10), (5, 8)])]
    assert repair(v1, v2, [V(1.0, "shift")], "straight", 12) == [(1.0, "drop fill @2 (shift)")]


def test_shift_already_fixed_this_round_is_not_repaired_again():
    # 违规来自本轮修补之前的文本：同一轮里前面的修补已经让 t 的跳动 ≤ max_shift 时，不能再白白去音 / 截短
    v1 = [Placed(1.0, 1.0, [(1, 0)]), Placed(2.0, 1.0, [(1, 3)])]
    v2 = [Placed(1.0, 1.0, [(3, 2), (5, 3)])]
    assert repair(v1, v2, [V(2.0, "shift")], "straight", 12, max_shift=4) == []
    assert v2[0].where == [(3, 2), (5, 3)] and v2[0].dur == 1.0


def test_held_melody_beats_a_fill_on_its_string():
    # 进副歌的长音（2 弦 3 品）还在响，下一拍的填充音要弹同一根弦：去掉填充，旋律不截（旋律 > 低音 > 填充）
    v1 = [Placed(0.0, 3.0, [(2, 3)])]
    v2 = [Placed(2.0, 2.0, [(2, 0), (6, 3)])]
    assert cut_same_string(v1, v2) == [(2.0, "drop fill @3 (same string)")]
    assert v1[0].dur == 3.0 and v2[0].where == [None, (6, 3)]


def test_held_melody_still_yields_to_a_bass_on_its_string():
    v1 = [Placed(0.0, 3.0, [(5, 3)])]
    v2 = [Placed(2.0, 2.0, [None, (5, 0)])]
    assert cut_same_string(v1, v2) == [(2.0, "melody cut @3 (same string)")]
    assert v1[0].dur == 2.0


def triplet_voices():
    # 第 4 拍是三连音拍：B3 空弦（4，1/3 拍）→ 1 弦 7 品（4+⅓）；第 3 拍起的低音（5 弦 2 品）还按着 → 跨度 2–7
    v1 = [Placed(2.0, 1.0, [(1, 0)]), Placed(3.0, 1 / 3, [(2, 0)]), Placed(3 + 1 / 3, 2 / 3, [(1, 7)])]
    v2 = [Placed(2.0, 2.0, [None, (5, 2)])]
    return v1, v2


def test_repair_maps_a_triplet_violation_to_its_attack():
    # 三连音拍记谱 = 演奏（k/3）：4+⅓ 的违规修在 4+⅓ 的起音上，不是折回拍点 4
    v1, v2 = triplet_voices()
    assert hand(v1, v2, 3 + 1 / 3) == [2, 7]
    assert repair(v1, v2, [V(3 + 1 / 3)], "shuffle", 12) == [(3 + 1 / 3, "bass cut @4+⅓ (span)")]
    assert v2[0].dur == pytest.approx(4 / 3)


def test_repair_maps_two_thirds_to_the_triplet_attack_not_the_offbeat_eighth():
    v1 = [Placed(3.0, 1 / 3, [(2, 0)]), Placed(3 + 1 / 3, 1 / 3, [(2, 0)]), Placed(3 + 2 / 3, 1 / 3, [(1, 7)])]
    v2 = [Placed(2.0, 2.0, [None, (5, 2)])]
    assert repair(v1, v2, [V(3 + 2 / 3)], "shuffle", 12) == [(3 + 2 / 3, "bass cut @4+⅔ (span)")]
    assert v2[0].dur == pytest.approx(5 / 3)


def test_repair_still_maps_a_swung_offbeat_to_the_written_eighth():
    # 摇摆八分对的后一个（记谱 +1/2）演奏在 +2/3：照旧换回记谱位置
    v1 = [Placed(2.0, 0.5, [(1, 0)]), Placed(2.5, 1.5, [(1, 7)])]
    v2 = [Placed(2.0, 2.0, [None, (5, 2)])]
    assert repair(v1, v2, [V(2 + 2 / 3)], "shuffle", 12) == [(2.5, "bass cut @3.5 (span)")]


def test_jump_measures_the_gap_in_performed_time():
    # 三连音 3+⅔ → 下一拍的反拍八分（记谱 3.5，演奏 3+⅔）：记谱只差 5/6 拍，演奏正好 1 拍——校验不算快速换把
    v1 = [Placed(2.0, 2 / 3, [(1, 0)]), Placed(2 + 2 / 3, 5 / 6, [(1, 1)]), Placed(3.5, 0.5, [(1, 8)])]
    assert jump(v1, [], 3.5, "shuffle") == 0
    assert jump(v1, [], 3.5, "straight") == 7
