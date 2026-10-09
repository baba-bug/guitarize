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
