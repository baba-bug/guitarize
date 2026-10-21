"""一首歌的总页面 out/index.html：顶上一排按钮切换这首歌已生成的各个编配页面（弹唱 / 指弹 / 旋律 / 卡祖笛……），
下面用 iframe 显示选中的那个，旁边给 PDF / 文本谱的链接。每次渲染任何一个编配都会重写它，新编配自动出现；
编配的变体（如旧版弹唱 strum_old，见 config.split_arr）排在所属编配后面。"""
from __future__ import annotations

import html
import json
from pathlib import Path

from ..config import ARR_NAMES, VARIANT_TAG, SongConfig, arr_label
from ..paths import SongPaths

PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · 全部谱子</title>
<style>
  :root {{ --bg: #fafaf7; --fg: #1d1d1b; --muted: #6b6b66; --accent: #c2410c; --line: #ddd; }}
  html, body {{ margin: 0; height: 100%; background: var(--bg); color: var(--fg); font: 15px/1.4 -apple-system, "PingFang SC", sans-serif; }}
  body {{ display: flex; flex-direction: column; }}
  nav {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; padding: 8px 16px; border-bottom: 1px solid var(--line); }}
  nav h1 {{ font-size: 16px; margin: 0 12px 0 0; }}
  nav button {{ font: inherit; padding: 4px 12px; border: 1px solid #ccc; border-radius: 6px; background: #fff; cursor: pointer; }}
  nav button[aria-pressed="true"] {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
  #files {{ margin-left: auto; color: var(--muted); font-size: 13px; }}
  #files a {{ color: var(--accent); margin-left: 8px; }}
  iframe {{ flex: 1; width: 100%; border: 0; }}
</style>
</head>
<body>
<nav><h1>{title}</h1><span id="tabs"></span><span id="files"></span></nav>
<iframe id="view" title="谱子"></iframe>
<script>
  const PAGES = {pages};
  if (location.protocol === "file:") {{   // 直接双击打开的文件：里面的谱子页面出不来，提示改用本地服务
    const m = location.pathname.match(new RegExp("/songs/.*$"));
    const url = "http://localhost:8766" + (m ? m[0] : "/");
    document.body.insertAdjacentHTML("afterbegin", '<div style="padding:12px 16px;background:#fde7dc;color:#7c2d12;' +
      'border-bottom:1px solid #f4b8a0">要通过本地服务打开才能看谱、播放：双击仓库里的「打开谱子.command」，或在终端运行 ' +
      '<code>python -m guitarize serve --port 8766</code>，然后打开 <a href="' + url + location.hash + '">' + url + '</a></div>');
  }}
  const tabs = document.getElementById("tabs"), view = document.getElementById("view"), files = document.getElementById("files");
  function show(key) {{
    const p = PAGES.find((x) => x.key === key) || PAGES[0];
    view.src = p.html;
    tabs.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.key === p.key)));
    files.innerHTML = p.files.map((f) => `<a href="${{f.url}}" target="_blank">${{f.label}}</a>`).join("");
    history.replaceState(null, "", "#" + p.key);   // 刷新后还停在这个编配
  }}
  PAGES.forEach((p) => {{
    const b = document.createElement("button");
    b.textContent = p.label; b.dataset.key = p.key; b.onclick = () => show(p.key);
    tabs.appendChild(b);
  }});
  show(location.hash.slice(1));
</script>
</body>
</html>
"""
FILE_LABELS = (("pdf", "PDF"), ("txt", "文本谱"), ("gp", "Guitar Pro"))


def _variants(paths: SongPaths, arr: str) -> list[str]:
    """out/ 里 arr 的变体页面名（out/<arr>_<标签>.html），按名字排序。"""
    return sorted(p.stem for p in paths.out_dir.glob(f"{arr}_*.html") if VARIANT_TAG.fullmatch(p.stem[len(arr) + 1:]))


def hub_pages(paths: SongPaths, cfg: SongConfig) -> list[dict]:
    """已经生成了网页的编配，按 song.yaml 的 arrangements 顺序（其余的按 ARR_NAMES 顺序排在后面），每个编配的变体紧跟在它后面。"""
    order = [a for a in cfg.arrangements if a in ARR_NAMES] + [a for a in ARR_NAMES if a not in cfg.arrangements]
    pages = []
    for name in (n for arr in order for n in (arr, *_variants(paths, arr))):
        if not paths.out(name, "html").exists():
            continue
        pages.append({"key": name, "label": arr_label(name), "html": paths.out(name, "html").name,
                      "files": [{"label": label, "url": paths.out(name, ext).name}
                                for ext, label in FILE_LABELS if paths.out(name, ext).exists()]})
    return pages


def build_hub(paths: SongPaths, cfg: SongConfig) -> Path | None:
    """写 out/index.html；一个编配网页都没有时不写，返回 None。"""
    pages = hub_pages(paths, cfg)
    if not pages:
        return None
    out = paths.out_dir / "index.html"
    out.write_text(PAGE.format(title=html.escape(cfg.title), pages=json.dumps(pages, ensure_ascii=False)),
                   encoding="utf-8")
    return out
