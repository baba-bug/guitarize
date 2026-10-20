"""guitarize 命令行入口。"""
from __future__ import annotations

import argparse
import functools
import http.server
import os
import re
from pathlib import Path

import yaml

from .config import ARR_NAMES, load_song_config
from .paths import REPO_ROOT, SongPaths

ARRS = tuple(ARR_NAMES)   # strum / finger / melody / kazoo
SHOW_VIOLATIONS = 5


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="guitarize", description="原曲 → 吉他谱（弹唱 / 指弹）+ 自动校验")
    ap.add_argument("--songs-dir", type=Path, default=None, help="songs 目录（默认仓库内 songs/）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_text in (("render", "生成交付物：gp / txt / pdf / html"),
                            ("verify", "构建 + 合成 + 可弹性 + 评分 → 报告")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("slug")
        p.add_argument("--arr", choices=ARRS, required=True)
    p = sub.add_parser("analyze", help="分析原曲：分轨 / 主唱 / 节拍 / 和弦 / 旋律 / leadsheet")
    p.add_argument("slug")
    p.add_argument("--steps", default="stems,lead,beats,chords,melody,leadsheet",
                   help="逗号分隔，可选 stems,lead,beats,chords,melody,leadsheet（lead：只重做主唱 / 伴唱分离）")
    p = sub.add_parser("audit-melody", help="旋律对照原唱（模型主唱 + 伴唱 + crepe）：覆盖 / 音准 / 伴唱音 / "
                                            "切碎的长音 / 节奏 / 重复段一致，写 reports/melody_audit.json")
    p.add_argument("slug")
    p.add_argument("--melody", type=Path, nargs="+", action="extend", default=[],
                   help="另外要比的旋律 json（analysis/melody.json 在的话总是第一行）")
    p = sub.add_parser("summary", help="打印 leadsheet 的逐小节摘要")
    p.add_argument("slug")
    p.add_argument("--capo", type=int, default=0)
    p = sub.add_parser("new", help="新建一首歌的 song.yaml")
    p.add_argument("slug")
    p.add_argument("--title", required=True)
    p.add_argument("--artist", required=True)
    p = sub.add_parser("acquire", help="获取音源并标准化为 source/original.wav（下载前须征得用户同意）")
    p.add_argument("slug")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--url")
    src.add_argument("--raw", type=Path)
    p.add_argument("--trim", type=float, nargs=2, metavar=("START", "END"))
    p = sub.add_parser("arrange", help="生成编配草稿 arrangements/<arr>.atex")
    p.add_argument("slug")
    p.add_argument("--arr", choices=ARRS, required=True)
    p.add_argument("--capo", type=int)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("serve", help="从仓库根目录起本地 HTTP 服务，用于打开交互谱")
    p.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)

    if args.cmd == "serve":
        return _serve(args.port)
    if args.cmd == "new":
        return _new(SongPaths.for_slug(args.slug, args.songs_dir), args.title, args.artist)
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
    elif args.cmd == "analyze":
        from .analyze.leadsheet import summary_table
        from .analyze.pipeline import run_analysis

        steps = [s.strip() for s in args.steps.split(",") if s.strip()]
        run_analysis(paths, cfg, steps)
        if "leadsheet" in steps:
            import json

            print(summary_table(json.loads(paths.leadsheet_json.read_text(encoding="utf-8"))))
    elif args.cmd == "audit-melody":
        from .analyze.melody_audit import audit_melody, format_missing, format_table

        rows = audit_melody(paths, cfg, args.melody)
        print(format_table(rows))
        print(format_missing(rows[0]))
        print(f"报告：{paths.reports_dir / 'melody_audit.json'}")
    elif args.cmd == "summary":
        import json

        from .analyze.leadsheet import summary_table

        print(summary_table(json.loads(paths.leadsheet_json.read_text(encoding="utf-8")), capo=args.capo))
    elif args.cmd == "acquire":
        from .acquire import acquire

        print(acquire(paths, url=args.url, raw=args.raw, trim=tuple(args.trim) if args.trim else None))
    elif args.cmd == "arrange":
        from .arrange.pipeline import arrange

        target, option, violations = arrange(paths, cfg, args.arr, capo=args.capo, force=args.force)
        print(target)
        if option is not None:   # 卡祖笛谱没有变调夹
            names = ", ".join(f"{k}→{v.name}" for k, v in option.shapes.items())
            print(f"capo {option.capo}（代价 {option.cost}）：{names}")
        if isinstance(violations, str):   # 自检本身没跑成（alphaTab 构建失败 / 找不到 node）：草稿已写出
            print(violations)
            return 0
        print(f"自检：{len(violations)} 处违规")   # 不让命令失败，但必须看得见
        for v in violations[:SHOW_VIOLATIONS]:
            print(f"  bar {v.bar} {v.rule}：{v.detail}")
        if len(violations) > SHOW_VIOLATIONS:
            print(f"  …另有 {len(violations) - SHOW_VIOLATIONS} 处")
    return 0


def _new(paths: SongPaths, title: str, artist: str) -> int:
    if paths.config.exists():
        raise FileExistsError(paths.config)
    paths.root.mkdir(parents=True, exist_ok=True)
    data = {"slug": paths.slug, "title": title, "artist": artist, "version": "tv", "level": "intermediate",
            "arrangements": ["strum", "finger"], "outputs": ["html", "pdf", "txt"],
            "source": {"url": None, "kind": None, "trim": None}, "reference_charts": [], "overrides": {}}
    paths.config.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(paths.config)
    return 0


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    """静态文件服务，支持单段 HTTP Range（音频跳转需要 206 响应）。"""

    _range_left: int | None = None
    ALLOWED_HOSTS = ("localhost", "127.0.0.1", "[::1]")

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        name = host[: host.index("]") + 1] if host.startswith("[") else host.rsplit(":", 1)[0] if ":" in host else host
        return name in self.ALLOWED_HOSTS

    def do_GET(self):
        if not self._host_ok():  # 防 DNS rebinding：只认本机主机名
            return self.send_error(403, "Forbidden host")
        return super().do_GET()

    def do_HEAD(self):
        if not self._host_ok():
            return self.send_error(403, "Forbidden host")
        return super().do_HEAD()

    def list_directory(self, path):
        self.send_error(404, "No permission to list directory")
        return None

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
