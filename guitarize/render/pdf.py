"""SVG 片段 → PDF：套 Bravura 字体后用无头 Chrome 打印（独立临时配置目录，不碰用户的 Chrome）。

无头 Chrome 在部分 macOS 环境里写完 PDF 后不会自行退出（进程停在 NSApplication 事件循环），
所以不能等它的退出码：轮询 PDF 写完（大小稳定且带 %%EOF）后主动终止进程组。
"""
from __future__ import annotations

import html
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from .synth import ALPHATAB_DIST

CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
FONT = ALPHATAB_DIST / "font" / "Bravura.woff2"
TIMEOUT = 120
POLL = 0.2
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>
@font-face{{font-family:"alphaTab";src:url("{font}") format("woff2")}}
.at{{font-family:"alphaTab";speak:none;font-style:normal;font-weight:normal;line-height:1;font-size:34px}}
@page{{size:A4;margin:12mm}}
body{{margin:0;background:#fff}}
svg{{display:block;break-inside:avoid;margin:0 auto}}
</style></head><body>{body}</body></html>"""


def _pdf_size(pdf: Path) -> int | None:
    """PDF 已完整写出（文件尾含 %%EOF）时返回文件大小，否则 None。"""
    try:
        size = pdf.stat().st_size
        with pdf.open("rb") as f:
            f.seek(max(0, size - 1024))
            tail = f.read()
    except OSError:
        return None
    return size if size > 0 and b"%%EOF" in tail else None


def _stop(proc: subprocess.Popen) -> None:
    """终止 Chrome 及其辅助进程（独立会话，进程组 id 即 proc.pid）。"""
    if proc.poll() is not None:
        return
    for sig, wait in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def svg_to_pdf(svg_fragment: Path, pdf: Path, title: str) -> Path:
    if not CHROME.exists():
        raise RuntimeError(f"Google Chrome not found at {CHROME}")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.unlink(missing_ok=True)  # 避免把旧文件当成这次的结果
    page = pdf.with_suffix(".print.html")
    page.write_text(
        PAGE.format(title=html.escape(title), font=FONT.as_uri(), body=svg_fragment.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    try:
        with tempfile.TemporaryDirectory(prefix="guitarize-chrome-") as profile:
            log = Path(profile) / "chrome.log"
            with log.open("wb") as log_fh:
                proc = subprocess.Popen(
                    [str(CHROME), "--headless=new", "--disable-gpu", f"--user-data-dir={profile}",
                     "--no-pdf-header-footer", f"--print-to-pdf={pdf}", "--virtual-time-budget=5000", page.as_uri()],
                    stdin=subprocess.DEVNULL,
                    stdout=log_fh,
                    stderr=log_fh,
                    start_new_session=True,
                )
                try:
                    deadline = time.monotonic() + TIMEOUT
                    last: int | None = None
                    while True:
                        size = _pdf_size(pdf)
                        if size is not None and size == last:
                            break  # 连续两次轮询大小不变且已收尾：写完了
                        last = size
                        if proc.poll() is not None:
                            if _pdf_size(pdf) is not None:
                                break
                            tail = log.read_text(errors="replace")[-800:]
                            raise RuntimeError(f"Chrome exited ({proc.returncode}) without writing {pdf}:\n{tail}")
                        if time.monotonic() > deadline:
                            tail = log.read_text(errors="replace")[-800:]
                            raise RuntimeError(f"Chrome did not produce {pdf} within {TIMEOUT}s:\n{tail}")
                        time.sleep(POLL)
                finally:
                    _stop(proc)
    finally:
        page.unlink(missing_ok=True)
    return pdf
