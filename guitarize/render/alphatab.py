"""调用 tools/alphatab/cli.mjs：alphaTex → 模型 JSON / MIDI / GP / SVG。"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..paths import REPO_ROOT

NODE_CLI = REPO_ROOT / "tools" / "alphatab" / "cli.mjs"


class AlphaTexError(RuntimeError):
    pass


def build(atex: Path, out_dir: Path, name: str, svg_width: int = 700) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["node", str(NODE_CLI), "build", str(atex), str(out_dir), name, "--svg-width", str(svg_width)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AlphaTexError(f"alphaTex build failed for {atex}:\n{proc.stderr.strip()}")
    last = proc.stdout.strip().splitlines()[-1]
    return {k: Path(v) for k, v in json.loads(last).items()}
