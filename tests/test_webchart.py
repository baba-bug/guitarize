import pytest

from guitarize.analyze.webchart import normalize_label, parse_webchart

CHART = """# source: example chart (chords only)
# BPM≒120 legend line
| G5 > | C |
# Verse
| D/F# ---- Em ---- | /B ↓
| N.C. -- /A - /B - |
Am D G N.C.
| Cmaj7 Cmaj7 | xyz |
"""


def test_normalize_label():
    assert normalize_label("G5") == "G5"
    assert normalize_label("D#/G") == "D#/G"
    assert normalize_label("/B") == "N"
    assert normalize_label("N.C.") == "N"
    assert normalize_label("E↓") == "E"
    assert normalize_label("C#5>") == "C#5"
    assert normalize_label("xyz") == "N"


def test_parse_webchart_bars_weights_sections():
    bars = parse_webchart(CHART)
    labels = [[c for c, _ in b.chords] for b in bars]
    assert labels == [["G5"], ["C"], ["D/F#", "Em"], ["N"], ["N"], ["Am"], ["D"], ["G"], ["N"], ["Cmaj7"], ["N"]]
    assert [w for _, w in bars[2].chords] == pytest.approx([0.5, 0.5])
    assert bars[0].section is None and bars[2].section == "Verse" and bars[5].section == "Verse"
    assert all(abs(sum(w for _, w in b.chords) - 1) < 1e-9 for b in bars)
