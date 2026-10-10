"""GAME（唱歌 → 音符的模型）分音：在主唱上跑 tools/game 的 infer.py extract，得到 (起点秒, 终点秒, 浮点 MIDI 音高)，
缓存成 analysis/game_notes.csv。旋律 v7（analyze/notes.py）以它为骨架，再用主唱的信号检查漏音、换字和长音。"""
from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from ..paths import REPO_ROOT

GAME_MODEL = Path("models") / "large" / "GAME-1.0-large" / "model.pt"   # 相对 GAME 目录
CACHE_MARK = "# "   # 缓存第一行：# {签名 JSON}，后面是 GAME 原样的 CSV（onset,offset,pitch）


def game_dir() -> Path:
    """GAME 仓库所在目录：环境变量 GUITARIZE_GAME_DIR，否则仓库内 tools/game（不入 git）。"""
    env = os.environ.get("GUITARIZE_GAME_DIR")
    return Path(env) if env else REPO_ROOT / "tools" / "game"


def _warn(reason: str) -> None:
    print(f"warning: GAME 不可用（{reason}）", file=sys.stderr)


def parse_game_csv(text: str) -> list[tuple[float, float, float]]:
    """GAME 的 CSV（--pitch-format number）→ [(起点秒, 终点秒, 浮点 MIDI 音高)]。"""
    return [(float(r["onset"]), float(r["offset"]), float(r["pitch"])) for r in csv.DictReader(io.StringIO(text))]


def run_game(wav: Path, language: str | None = None, root: Path | None = None) -> str | None:
    """在 wav 上跑 GAME，返回它写出的 CSV 原文。language 是 GAME lang_map.json 里的语言（en / ja / yue / zh），
    None 时不传 -l（GAME 不按语言分音）。没有 GAME 目录 / 模型、infer.py 失败或没写出 CSV 时 stderr 打一行警告，返回 None。"""
    root = Path(root) if root is not None else game_dir()
    model = root / GAME_MODEL
    if not (root / "src" / "infer.py").exists():
        _warn(f"{root} 里没有 src/infer.py")
        return None
    if not model.exists():
        _warn(f"没有模型 {model}")
        return None
    with tempfile.TemporaryDirectory(prefix="guitarize-game-") as tmp:
        cmd = [sys.executable, "infer.py", "extract", str(Path(wav).resolve()), "-m", str(model.resolve()),
               *(["-l", language] if language else []),
               "--output-formats", "csv", "--pitch-format", "number", "--output-dir", tmp]
        proc = subprocess.run(cmd, cwd=root / "src", capture_output=True)
        if proc.returncode != 0:
            lines = [ln.strip() for ln in (proc.stderr or b"").decode("utf-8", "replace").splitlines() if ln.strip()]
            _warn(f"infer.py exit {proc.returncode}: {lines[-1][:300] if lines else 'no stderr'}")
            return None
        out = Path(tmp) / f"{Path(wav).stem}.csv"
        if not out.exists():
            _warn(f"infer.py 没有写出 {out.name}")
            return None
        return out.read_text(encoding="utf-8")


def _signature(wav: Path, model: Path, language: str | None) -> str:
    """缓存签名：主唱文件名、大小、修改时间，模型名，语言。"""
    st = Path(wav).stat()
    return json.dumps({"stem": Path(wav).name, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                       "model": model.parent.name, "language": language}, sort_keys=True)


def _read_cache(cache: Path, signature: str) -> list[tuple[float, float, float]] | None:
    """签名相同时读出缓存的音符；没有缓存、签名不同、读不了都是 None。"""
    try:
        head, _, body = cache.read_text(encoding="utf-8").partition("\n")
        return parse_game_csv(body) if head == CACHE_MARK + signature else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_cache(cache: Path, signature: str, text: str) -> None:
    """先写同目录的临时文件再 os.replace：写到一半出错不会留下半个缓存。"""
    cache.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=cache.parent, prefix=f".{cache.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"{CACHE_MARK}{signature}\n{text}")
        os.replace(tmp, cache)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def game_notes(wav: Path, cache: Path, language: str | None = None,
               root: Path | None = None) -> list[tuple[float, float, float]] | None:
    """run_game 带缓存（analysis/game_notes.csv）：还是同一条主唱（文件名、大小、修改时间）、同一模型和语言就直接读，
    否则重跑并写缓存。GAME 不可用时 None（已打过警告），melody 步骤据此退回 crepe 逐格量化。"""
    root = Path(root) if root is not None else game_dir()
    signature = _signature(wav, root / GAME_MODEL, language)
    found = _read_cache(Path(cache), signature)
    if found is not None:
        return found
    text = run_game(wav, language, root)
    if text is None:
        return None
    rows = parse_game_csv(text)
    _write_cache(Path(cache), signature, text)
    return rows
