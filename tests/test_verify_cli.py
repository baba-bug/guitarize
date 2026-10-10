import csv
import json

from guitarize.cli import main


def test_verify_twice_creates_iterations(demo_song, capsys):
    songs = str(demo_song.root.parent)
    assert main(["--songs-dir", songs, "verify", "demo", "--arr", "finger"]) == 0
    assert main(["--songs-dir", songs, "verify", "demo", "--arr", "finger"]) == 0
    root = demo_song.report_root("finger")
    assert (root / "iter-01" / "report.md").exists() and (root / "iter-02" / "report.md").exists()
    s = json.loads((root / "iter-02" / "scores.json").read_text(encoding="utf-8"))["summary"]
    assert s["pass"] is True and s["violations"] == 0
    assert s["chord_agreement"] == 1.0 and s["melody_f1"] == 1.0
    rows = list(csv.DictReader((root / "history.csv").open(encoding="utf-8")))
    assert [r["iter"] for r in rows] == ["iter-01", "iter-02"]
    report = (root / "iter-02" / "report.md").read_text(encoding="utf-8")
    row = next(ln for ln in report.splitlines() if ln.startswith("| 和弦一致率"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1] == "100.0%" and cells[2] == cells[1]  # 上一轮 = 本轮
    assert "指弹校验报告" in report and "✅ 达标" in report
    assert "小节数（编配 / 参考）" in report
    assert "旋律召回率（全部参考音）" in report and s["finger_style"] == "full"   # 默认完整旋律：考每一个音
    assert "指弹校验报告" in capsys.readouterr().out


def test_skeleton_finger_style_is_verified_against_the_skeleton(demo_song):
    cfg = demo_song.config.read_text(encoding="utf-8")
    demo_song.config.write_text(cfg + "\noverrides: {finger_style: skeleton}\n", encoding="utf-8")
    assert main(["--songs-dir", str(demo_song.root.parent), "verify", "demo", "--arr", "finger"]) == 0
    it = demo_song.report_root("finger") / "iter-01"
    assert json.loads((it / "scores.json").read_text(encoding="utf-8"))["summary"]["finger_style"] == "skeleton"
    assert "旋律召回率（骨架音）" in (it / "report.md").read_text(encoding="utf-8")


def test_verify_flags_problem_bar_with_snippet(demo_song):
    ls = json.loads(demo_song.leadsheet_json.read_text(encoding="utf-8"))
    ls["bars"][1]["chords"] = [{"beat": 1, "dur_beats": 4, "label": "D", "confidence": "high"}]
    demo_song.leadsheet_json.write_text(json.dumps(ls), encoding="utf-8")
    main(["--songs-dir", str(demo_song.root.parent), "verify", "demo", "--arr", "strum"])
    it = demo_song.report_root("strum") / "iter-01"
    s = json.loads((it / "scores.json").read_text(encoding="utf-8"))
    assert s["problems"] == [2] and s["summary"]["pass"] is False
    assert (it / "snippets" / "bar-002_synth.wav").exists()
    report = (it / "report.md").read_text(encoding="utf-8")
    assert "### 第 2 小节" in report and "参考 D" in report


def test_verify_snippets_for_strum_stop_at_the_cap(demo_song, monkeypatch):
    # 弹唱 / 指弹：只有前 MAX_SNIPPETS 个问题小节出试听片段和色度图（旋律谱不截断，见 test_verify_melody）
    from guitarize.verify import report

    ls = json.loads(demo_song.leadsheet_json.read_text(encoding="utf-8"))
    for i in (1, 3):
        ls["bars"][i]["chords"] = [{"beat": 1, "dur_beats": 4, "label": "D", "confidence": "high"}]
    demo_song.leadsheet_json.write_text(json.dumps(ls), encoding="utf-8")
    monkeypatch.setattr(report, "MAX_SNIPPETS", 1)
    main(["--songs-dir", str(demo_song.root.parent), "verify", "demo", "--arr", "strum"])
    it = demo_song.report_root("strum") / "iter-01"
    assert json.loads((it / "scores.json").read_text(encoding="utf-8"))["problems"] == [2, 4]
    assert sorted(p.name for p in (it / "snippets").iterdir()) == ["bar-002_synth.wav"]


def test_verify_without_reference_is_not_a_pass(demo_song):
    demo_song.leadsheet_json.unlink()
    assert main(["--songs-dir", str(demo_song.root.parent), "verify", "demo", "--arr", "strum"]) == 0
    it = demo_song.report_root("strum") / "iter-01"
    report = (it / "report.md").read_text(encoding="utf-8")
    assert "没有参考" in report and "未达标" in report and "参考不足" in report
    s = json.loads((it / "scores.json").read_text(encoding="utf-8"))["summary"]
    assert s["pass"] is False and s["checks"]["reference"] is False


def test_render_cli(demo_song, capsys):
    assert main(["--songs-dir", str(demo_song.root.parent), "render", "demo", "--arr", "strum"]) == 0
    printed = capsys.readouterr().out
    assert "pdf:" in printed and demo_song.out("strum", "html").exists()


def test_history_migrates_old_header(tmp_path):
    from guitarize.verify.report import HISTORY_FIELDS, _append_history

    path = tmp_path / "history.csv"
    path.write_text("iter,time,violations\niter-01,t,2\n", encoding="utf-8")
    _append_history(path, "iter-02", {"violations": 0, "scored_beats": 8})
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert list(rows[0]) == HISTORY_FIELDS
    assert rows[0]["violations"] == "2" and rows[1]["scored_beats"] == "8"
