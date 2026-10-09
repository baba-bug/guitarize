import numpy as np

from guitarize.analyze.key import MAJOR, MINOR, estimate_key, parse_key, scale_pcs


def test_estimate_key():
    assert estimate_key(np.roll(MAJOR, 11))[:2] == ("B", "major")
    assert estimate_key(np.roll(MINOR, 9))[:2] == ("A", "minor")


def test_scale_and_parse():
    assert scale_pcs("B", "major") == {11, 1, 3, 4, 6, 8, 10}
    assert scale_pcs("A", "minor") == {9, 11, 0, 2, 4, 5, 7}
    assert parse_key("B") == ("B", "major")
    assert parse_key("G#m") == ("G#", "minor")
    assert parse_key("Bb") == ("A#", "major")


def _segs(labels):
    return [{"bar": i + 1, "beat": 1, "dur_beats": 4, "label": l} for i, l in enumerate(labels)]


def test_diatonic_fraction_counts_triads_power_roots_and_skips_n():
    from guitarize.analyze.key import diatonic_fraction

    # G 大调：C#7 的三和弦 C#-F-G# 不在调内；B5 只看根音；N 不计
    segs = _segs(["G", "Em7", "D/F#", "B5", "C#7", "N"])
    assert diatonic_fraction(segs, "G", "major") == 4 / 5
    assert diatonic_fraction(_segs(["N"]), "G", "major") == 0.0


def test_cross_check_prefers_chord_key_over_wrong_chroma_key():
    from guitarize.analyze.key import cross_check_key

    segs = _segs(["G", "C", "D", "Em", "G", "C", "Am", "D7"] * 2)
    kc = cross_check_key(("D", "major"), segs)
    assert (kc["tonic"], kc["mode"]) == ("G", "major")  # E 小调同分：主和弦 G 出现得多，取 G 大调
    # D 大调：G、D、Em 是调内；C、Am 不是；D7 按真实和弦音（含 C）也不是 → 4/8（以前按 D 大三和弦算成 5/8）
    assert kc["score"] == 1.0 and kc["chroma_key_score"] == 0.5 and kc["used"] is True


def test_cross_check_keeps_chroma_key_when_close():
    from guitarize.analyze.key import cross_check_key

    segs = _segs(["G", "C", "D", "Em"] + ["D"] * 7)  # G 大调 1.0，D 大调 10/11：差不到 0.1
    kc = cross_check_key(("D", "major"), segs)
    assert kc["score"] - kc["chroma_key_score"] < 0.1 and kc["used"] is False


def test_chosen_key_and_report():
    from guitarize.analyze.key import chosen_key, key_report

    old = {"key": {"tonic": "D", "mode": "major", "r": 0.8}}
    assert chosen_key(old) == ("D", "major")  # 旧 chords.json 没有 key_from_chords
    kc = {"tonic": "G", "mode": "major", "score": 1.0, "chroma_key_score": 0.625, "used": True}
    new = {**old, "key_from_chords": kc}
    assert chosen_key(new) == ("G", "major")
    assert "G major from chords" in key_report(new, overridden=False)
    assert chosen_key({**old, "key_from_chords": {**kc, "used": False}}) == ("D", "major")


def _web_segs(pairs):
    """(解析后的标签, 网上谱原标签) → 带 votes["web"] 的段落。"""
    return [{"bar": i + 1, "beat": 1, "dur_beats": 4, "label": lab, "confidence": "high",
             "votes": {"template": lab, "web": web}} for i, (lab, web) in enumerate(pairs)]


def test_diatonic_judges_chords_by_their_real_tones():
    from guitarize.analyze.key import diatonic_fraction

    # sus2 / sus4 / dim / m7b5 / 斜线和弦按真实和弦音判断，而不是先化成大小三和弦
    assert diatonic_fraction(_segs(["Dsus2"]), "C", "major") == 1.0       # D E A 全在 C 大调里（大三和弦 D F# A 不在）
    assert diatonic_fraction(_segs(["Dsus2"]), "A", "major") == 1.0
    assert diatonic_fraction(_segs(["Gsus4"]), "C", "major") == 1.0       # G C D
    assert diatonic_fraction(_segs(["Bdim"]), "C", "major") == 1.0        # B D F（majmin 当成 B 小三和弦，D F# 不全在）
    assert diatonic_fraction(_segs(["Bm7-5"]), "C", "major") == 1.0       # B D F A
    assert diatonic_fraction(_segs(["C/E"]), "C", "major") == 1.0
    assert diatonic_fraction(_segs(["G/F#"]), "C", "major") == 0.0        # 低音 F# 在调外
    assert diatonic_fraction(_segs(["Dsus2"]), "D", "minor") == 1.0       # D E A
    assert diatonic_fraction(_segs(["Bdim"]), "G", "major") == 0.0        # B D F：F 自然音不在 G 大调


def test_chord_key_does_not_switch_on_sus2_chords():
    """C Dsus2 G … 的 Dsus2 当大三和弦（D F# A）会让 G 大调显得更"调内"，实际 D E A 在 C 大调里也全是调内音。"""
    from guitarize.analyze.key import chosen_key, cross_check_key, diatonic_fraction

    segs = _segs("C Dsus2 G C Am Dsus2 G C Em Am Dsus2 G".split())
    assert diatonic_fraction(segs, "C", "major") == 1.0
    kc = cross_check_key(("C", "major"), segs)
    assert kc["used"] is False and kc["chroma_key_score"] == 1.0
    assert chosen_key({"key": {"tonic": "C", "mode": "major", "r": 0.8}, "key_from_chords": kc}) == ("C", "major")


def test_power_chord_beats_score_by_root_only():
    """网上谱的强力和弦被调性解析成了大 / 小三和弦（偏向色度调）：这些拍只按根音是否在音阶里计分。"""
    from guitarize.analyze.key import cross_check_key, diatonic_fraction

    # A5 被解析成 A 大三（A C# E）：按三和弦在 G 大调里不是调内，按根音 A 是
    assert diatonic_fraction(_web_segs([("A", "A5")]), "G", "major") == 1.0
    assert diatonic_fraction(_web_segs([("A", "A")]), "G", "major") == 0.0   # 真正的 A 大三和弦仍按三和弦判断
    assert diatonic_fraction(_web_segs([("Am", "A5")]), "D", "major") == 1.0  # Am（A C E）里的 C 不在 D 大调，但 A5 只看根音
    # 没有 votes 的旧段落照旧
    assert diatonic_fraction([{"bar": 1, "beat": 1, "dur_beats": 4, "label": "A"}], "G", "major") == 0.0

    # D5 A5 G5 D5：色度调 G 大调把 A5 解析成了 Am。D 大调按根音全在调内，不该因为那个 Am 扣分
    segs = _web_segs([("D", "D5"), ("Am", "A5"), ("G", "G5"), ("D", "D5")] * 2)
    kc = cross_check_key(("G", "major"), segs)
    assert (kc["tonic"], kc["mode"], kc["score"]) == ("D", "major", 1.0)
