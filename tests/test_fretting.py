from guitarize.arrange.fretting import FretEvent, _position, _transition, assign_frets
from guitarize.config import load_profile

MID = load_profile("intermediate")


def test_first_position_melody_with_bass():
    evs = assign_frets([FretEvent(0, 59, 43), FretEvent(1, 60, 48), FretEvent(2, 62, None)], MID)
    assert [(e.melody_pos, e.bass_pos) for e in evs] == [((2, 0), (6, 3)), ((2, 1), (5, 3)), ((2, 3), None)]
    assert all(e.fallback is None for e in evs)


def test_unplayable_event_falls_back_without_crashing():
    ev = assign_frets([FretEvent(0, 71, 43)], MID)[0]
    assert ev.fallback == "bass+12" and ev.melody_pos == (1, 7) and ev.bass_pos == (3, 0)


def test_bass_above_melody_is_dropped_an_octave():
    ev = assign_frets([FretEvent(0, 57, 60)], MID)[0]
    assert ev.melody_pos is not None and ev.bass_pos is not None
    m_string, b_string = ev.melody_pos[0], ev.bass_pos[0]
    assert m_string < b_string


def test_fill_sits_between_melody_and_bass():
    ev = assign_frets([FretEvent(0, 66, 43, fill=59)], MID)[0]
    assert ev.fallback is None
    assert ev.melody_pos == (1, 2) and ev.fill_pos == (2, 0) and ev.bass_pos == (6, 3)


def test_fill_is_kept_when_bass_moves_up_an_octave():
    ev = assign_frets([FretEvent(0, 71, 43, fill=59)], MID)[0]
    assert ev.fallback == "bass+12"
    assert ev.melody_pos == (1, 7) and ev.fill_pos == (2, 0) and ev.bass_pos == (3, 0)


def test_melody_prefers_top_three_strings():
    evs = assign_frets([FretEvent(t, p, None) for t, p in enumerate([55, 57, 59, 60])], MID)
    assert all(e.melody_pos[0] <= 3 for e in evs)


def test_hand_box_intervals_and_transition():
    assert _position(((6, 3),), 3) == (1, 3)
    assert _position(((2, 1), (5, 3)), 3) == (1, 1)
    assert _position(((1, 0), (2, 0)), 3) is None
    assert _transition(_position(((6, 3),), 3), _position(((2, 1), (5, 3)), 3), 1.0, MID) == 0
    assert _position(((2, 8),), 3) == (5, 8)
    assert _transition((1, 1), (5, 8), 1.0, MID) == 8.0
    assert _transition(None, (5, 8), 0.5, MID) == 0


def test_shift_rule_is_a_hard_cost_within_one_beat():
    # 最低品 1 → 6（差 5 > max_shift 4）：一拍以内是校验硬规则，代价 +1000；一拍及以上只按距离计
    assert _transition((1, 1), (3, 6), 0.5, MID) >= 1000
    assert _transition((1, 1), (3, 6), 1.0, MID) < 1000
    assert _transition((1, 1), (2, 5), 0.5, MID) < 1000     # 差 4 不超限


def test_open_only_event_keeps_previous_hand_position():
    # 12 品 → 空弦 → A4：空弦音不占把位，手仍在 12 品附近，应选 2 弦 10 品而不是 1 弦 5 品
    evs = assign_frets([FretEvent(0, 76, None), FretEvent(1, 64, None), FretEvent(2, 69, None)], MID)
    assert [e.melody_pos for e in evs] == [(1, 12), (1, 0), (2, 10)]


def test_no_hard_shift_across_open_melody_note():
    # 评审回归（capo 4 第 48 小节）：B7/D# 低音 D#3，反拍空弦旋律，第 3 拍再弹低音，第 4 拍旋律要 5 品。
    # 旧代价模型：4 弦 1 品 →（空弦，代价 0）→ 5 弦 6 品，半拍内 1→6 换把违规。空弦音继承手型后不能这样跳。
    evs = assign_frets([FretEvent(0, None, 51, fill=59), FretEvent(1.5, 64, None),
                        FretEvent(2, None, 51, fill=59), FretEvent(3, 69, None)], MID)
    assert evs[1].melody_pos == (1, 0)
    assert abs(evs[0].bass_pos[1] - evs[2].bass_pos[1]) <= MID.max_shift


def test_silently_dropped_fill_is_recorded():
    ev = assign_frets([FretEvent(0, 60, 50, fill=48)], MID)[0]
    assert ev.fill_pos is None
    assert ev.fallback is not None and "fill" in ev.fallback


def test_legato_pair_prefers_one_string():
    # D4 → E4：平时 E4 弹 1 弦空弦（换弦重新拨）；连奏（同一音节里滑上去）时留在 2 弦 3 → 5 品，能击弦
    plain = assign_frets([FretEvent(0, 62, None), FretEvent(1, 64, None)], MID)
    assert [e.melody_pos for e in plain] == [(2, 3), (1, 0)]
    legato = assign_frets([FretEvent(0, 62, None), FretEvent(1, 64, None, legato=True)], MID)
    assert [e.melody_pos for e in legato] == [(2, 3), (2, 5)]


def test_legato_slide_moves_the_hand_for_free_but_keeps_the_shift_rule():
    # A3 按 2 拍后滑到 D4（3 弦 2 → 7 品）：滑音本身就是换把，不计手型移动的距离代价
    legato = assign_frets([FretEvent(0, 57, None), FretEvent(2, 62, None, legato=True)], MID)
    assert [e.melody_pos for e in legato] == [(3, 2), (3, 7)]
    # 半拍内 2 → 7 品超过 max_shift（校验硬规则）：只是偏好，不是硬约束——换弦、照常拨弦
    quick = assign_frets([FretEvent(0, 57, None), FretEvent(0.5, 62, None, legato=True)], MID)
    assert [e.melody_pos for e in quick] == [(3, 2), (2, 3)]


def test_legato_without_a_melody_before_it_is_ignored():
    evs = assign_frets([FretEvent(0, None, 43), FretEvent(1, 64, None, legato=True)], MID)
    assert evs[1].melody_pos == (1, 0) and evs[1].fallback is None


def test_slide_out_note_prefers_a_fretted_position():
    # 尾滑（音尾滑出）要按着弦才滑得出去：E4 平时弹 1 弦空弦，带尾滑时放到 2 弦 5 品
    assert assign_frets([FretEvent(0, 64, None)], MID)[0].melody_pos == (1, 0)
    assert assign_frets([FretEvent(0, 64, None, slide_out=True)], MID)[0].melody_pos == (2, 5)
    evs = assign_frets([FretEvent(0, 62, None), FretEvent(1, 64, None, slide_out=True)], MID)
    assert [e.melody_pos for e in evs] == [(2, 3), (2, 5)]
