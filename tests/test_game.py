import os
import subprocess
import sys

import pytest

from guitarize.analyze import game

CSV = "onset,offset,pitch\n0.460,0.830,66.009\n0.830,1.080,66.386\n1.460,1.560,57.099\n"
ROWS = [(0.46, 0.83, 66.009), (0.83, 1.08, 66.386), (1.46, 1.56, 57.099)]


def _game_root(tmp_path, model=True):
    """假的 GAME 仓库：src/infer.py + models/large/GAME-1.0-large/model.pt（真的 GAME 不在测试里跑）。"""
    root = tmp_path / "game"
    (root / "src").mkdir(parents=True)
    (root / "src" / "infer.py").write_text("")
    if model:
        (root / game.GAME_MODEL).parent.mkdir(parents=True)
        (root / game.GAME_MODEL).write_bytes(b"x")
    return root


def _fake_run(monkeypatch, calls, csv_text=CSV, returncode=0, stderr=b""):
    """假的 subprocess.run：记下命令和 cwd，像 infer.py extract 一样把 <wav 名>.csv 写进 --output-dir。"""
    def run(cmd, cwd=None, capture_output=False, **kw):
        calls.append((cmd, cwd))
        assert capture_output                                  # GAME 的输出很吵：收起来，失败时只报最后一行
        if csv_text is not None:
            out = cmd[cmd.index("--output-dir") + 1]
            with open(os.path.join(out, os.path.basename(cmd[3]).rsplit(".", 1)[0] + ".csv"), "w") as f:
                f.write(csv_text)
        return subprocess.CompletedProcess(cmd, returncode, b"", stderr)

    monkeypatch.setattr(game.subprocess, "run", run)


def _wav(tmp_path, data=b"RIFF lead"):
    wav = tmp_path / "stems" / "lead_raw.wav"
    wav.parent.mkdir(exist_ok=True)
    wav.write_bytes(data)
    return wav


def test_game_dir_defaults_to_tools_game(tmp_path, monkeypatch):
    from guitarize.paths import REPO_ROOT

    monkeypatch.delenv("GUITARIZE_GAME_DIR")
    assert game.game_dir() == REPO_ROOT / "tools" / "game"
    monkeypatch.setenv("GUITARIZE_GAME_DIR", str(tmp_path))
    assert game.game_dir() == tmp_path


def test_game_notes_runs_infer_extract_and_reads_the_csv(tmp_path, monkeypatch):
    root = _game_root(tmp_path)
    monkeypatch.setenv("GUITARIZE_GAME_DIR", str(root))   # 不传 root：取环境变量
    calls = []
    _fake_run(monkeypatch, calls)
    wav = _wav(tmp_path)
    assert game.game_notes(wav, tmp_path / "analysis" / "game_notes.csv", "ja") == ROWS
    (cmd, cwd), = calls
    out = cmd[cmd.index("--output-dir") + 1]
    assert cmd == [sys.executable, "infer.py", "extract", str(wav), "-m", str(root / game.GAME_MODEL), "-l", "ja",
                   "--output-formats", "csv", "--pitch-format", "number", "--output-dir", out]
    assert cwd == root / "src"
    assert not os.path.exists(out)                         # 临时输出目录清掉了


def test_game_notes_without_a_language_omits_the_flag(tmp_path, monkeypatch):
    # GAME 的 -l 默认 None（= 不按语言分音），song.yaml 没写 language 时不传
    calls = []
    _fake_run(monkeypatch, calls)
    assert game.game_notes(_wav(tmp_path), tmp_path / "game_notes.csv", None, root=_game_root(tmp_path)) == ROWS
    assert "-l" not in calls[0][0]


def test_game_notes_are_cached_per_wav_and_language(tmp_path, monkeypatch):
    root, cache = _game_root(tmp_path), tmp_path / "analysis" / "game_notes.csv"
    calls = []
    _fake_run(monkeypatch, calls)
    wav = _wav(tmp_path)
    assert game.game_notes(wav, cache, "ja", root=root) == ROWS
    assert cache.read_text().startswith("# ") and "lead_raw.wav" in cache.read_text().splitlines()[0]
    assert game.game_notes(wav, cache, "ja", root=root) == ROWS
    assert len(calls) == 1                                    # 同一条主唱、同一语言：读缓存
    game.game_notes(wav, cache, "zh", root=root)
    assert len(calls) == 2                                    # 语言变了：重跑
    _wav(tmp_path, b"RIFF a new lead")
    game.game_notes(wav, cache, "zh", root=root)
    assert len(calls) == 3                                    # 主唱重分过（大小 / 修改时间变了）：重跑
    cache.write_text("garbage")
    game.game_notes(wav, cache, "zh", root=root)
    assert len(calls) == 4 and game.game_notes(wav, cache, "zh", root=root) == ROWS   # 读不了的缓存当作没有
    assert not [p for p in cache.parent.iterdir() if p.name != cache.name]       # 没有留下临时文件


@pytest.mark.parametrize("setup, says", [
    (lambda tmp: tmp / "nowhere", "nowhere"),
    (lambda tmp: _game_root(tmp, model=False), "model.pt"),
])
def test_game_notes_is_none_with_one_warning_when_game_is_missing(tmp_path, monkeypatch, capsys, setup, says):
    calls = []
    _fake_run(monkeypatch, calls)
    cache = tmp_path / "game_notes.csv"
    assert game.game_notes(_wav(tmp_path), cache, "ja", root=setup(tmp_path)) is None
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "GAME" in err and says in err
    assert calls == [] and not cache.exists()


def test_game_notes_is_none_with_one_warning_when_infer_fails(tmp_path, monkeypatch, capsys):
    calls = []
    _fake_run(monkeypatch, calls, csv_text=None, returncode=1,
              stderr=b"Traceback (most recent call last):\n  ...\nValueError: Language 'xx' not supported\n\n")
    cache = tmp_path / "game_notes.csv"
    assert game.game_notes(_wav(tmp_path), cache, "xx", root=_game_root(tmp_path)) is None
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "exit 1" in err and "Language 'xx' not supported" in err
    assert not cache.exists()


def test_game_notes_is_none_when_infer_writes_no_csv(tmp_path, monkeypatch, capsys):
    _fake_run(monkeypatch, [], csv_text=None)
    assert game.game_notes(_wav(tmp_path), tmp_path / "game_notes.csv", None, root=_game_root(tmp_path)) is None
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "lead_raw.csv" in err
