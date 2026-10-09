"""guitarize 命令行入口。"""
from __future__ import annotations

import argparse
import functools
import http.server
import os
import re
from pathlib import Path

from .config import load_song_config
from .paths import REPO_ROOT, SongPaths

ARRS = ("strum", "finger")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="guitarize", description="原曲 → 吉他谱（弹唱 / 指弹）+ 自动校验")
    ap.add_argument("--songs-dir", type=Path, default=None, help="songs 目录（默认仓库内 songs/）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_text in (("render", "生成交付物：gp / txt / pdf / html"),
                            ("verify", "构建 + 合成 + 可弹性 + 评分 → 报告")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("slug")
        p.add_argument("--arr", choices=ARRS, required=True)
    p = sub.add_parser("serve", help="从仓库根目录起本地 HTTP 服务，用于打开交互谱")
    p.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)

    if args.cmd == "serve":
        return _serve(args.port)
    paths = SongPaths.for_slug(args.slug, args.songs_dir)
    cfg = load_song_config(paths)
    if args.cmd == "render":
        from .render.pipeline import render_all

        for kind, path in render_all(paths, cfg, args.arr).items():
            print(f"{kind}: {path}")
    elif args.cmd == "verify":
        from .verify.report import run_verify

        out = run_verify(paths, cfg, args.arr)
        print((out / "report.md").read_text(encoding="utf-8"))
        print(f"报告目录：{out}")
    return 0


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    """静态文件服务，支持单段 HTTP Range（音频跳转需要 206 响应）。"""

    _range_left: int | None = None

    def send_head(self):
        header = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not header or not os.path.isfile(path):
            return super().send_head()
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
        if not m or (not m.group(1) and not m.group(2)):
            return super().send_head()
        size = os.path.getsize(path)
        if m.group(1):
            start = int(m.group(1))
            end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
        else:
            start, end = max(0, size - int(m.group(2))), size - 1
        if start >= size or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self._range_left = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        left = self._range_left
        if left is None:
            return super().copyfile(source, outputfile)
        while left > 0:
            chunk = source.read(min(64 * 1024, left))
            if not chunk:
                break
            outputfile.write(chunk)
            left -= len(chunk)
        self._range_left = None


def _serve(port: int) -> int:
    handler = functools.partial(RangeRequestHandler, directory=str(REPO_ROOT))
    print(f"http://localhost:{port}/songs/  （Ctrl+C 停止）")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
    return 0
