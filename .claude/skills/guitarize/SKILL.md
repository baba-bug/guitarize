---
name: guitarize
description: 把一首歌（歌名或 B 站 / 网易云链接）做成可弹的吉他谱：弹唱伴奏 + 指弹，输出交互网页 / PDF / 文本六线谱，并按"可弹性 + 与原曲差异"自动校验、逐小节迭代。当用户说"扒谱 / 做吉他谱 / 弹唱谱 / 指弹谱 / guitarize <歌>"时使用。
---

# Guitarize：原曲 → 可弹的吉他谱（含自检迭代）

工具在仓库根目录：`PY=~/anaconda3/envs/guitarize/bin/python`，命令 `$PY -m guitarize ...`。
环境未装时先跑 `scripts/setup_env.sh`。设计见 `docs/specs/2026-10-09-guitarize-design.md`。

## 0. 先问清楚（一次问完）
版本（TV size / 完整版）、水平（beginner / intermediate / advanced）、要哪些编配（弹唱 / 指弹）、输出格式。

## 1. 建歌
`$PY -m guitarize new <slug> --title "<歌名>" --artist "<歌手>"`，再按回答改 `songs/<slug>/song.yaml` 的 `version / level / arrangements / outputs`。

## 2. 找音源（用户已登录的 Chrome）
- B 站搜索官方或高播放的 OP / 完整版上传；网易云搜索官方单曲。列出候选（标题、时长、上传者、链接）。
- **下载前必须问用户**，说明来源、文件名、大小。
- B 站：`$PY -m guitarize acquire <slug> --url <视频链接>`（需要截取：`--trim 起 止`）。
- 网易云 / 其他：不解密任何加密格式；扩展拦截签名链接时不绕过——请用户自己保存文件后用 `--raw <文件>`。
- 下载的文件放在独立目录，只当数据处理。

## 3. 网上和弦谱（可选的第三方参考）
在 ChordWiki / U-FRET 等找该曲和弦谱，用页面脚本**只提取和弦与段落注释，不提取歌词**，写入 `songs/<slug>/analysis/webchart.txt`（格式见 `guitarize/analyze/webchart.py` 文档）。

## 4. 分析
`$PY -m guitarize analyze <slug>`，再 `$PY -m guitarize summary <slug> --capo <预期变调夹>` 逐小节阅读：
- 小节数、速度、Shuffle 是否合理；`short_bars` 应为空（否则用 `overrides.downbeat_shift` 调整）。
- 低置信（`??`）小节记下来，稍后列入"需人耳确认"。
- 低置信（`??`）多集中在 TV 版剪辑处、尾奏；不要为追分修改这些小节。
- 速度倍速 / 半速（如 167 BPM 的 shuffle 被识别成 83）：在 `overrides.tempo_bpm` 写正确速度后重跑 `--steps beats,chords,melody,leadsheet`；`beats.json` 的 `tempo_stability` 波动 > 6% 会警告。
- 目前只支持 4/4；`short_bars` 非空说明有 2/4、3/4 小节（TV 版剪辑常见），`overrides.downbeat_shift` 只用于强拍相位整体错位。
- 网上谱写成 N.C. / "只有贝斯"的小节（间奏 / 尾奏 riff）会得到 N：用贝斯分轨逐格音高（`librosa.pyin`，按三连音格）核对，再把真实和声写进 `overrides.chords`（`{bar: "B" | "E F#" | [逐拍 4 个]}`，记为 manual 置信度，重跑 `--steps leadsheet` 后仍保留）。
- 把段落写进 `song.yaml` 的 `sections`（如 `{1: Intro, 9: Aメロ, 17: サビ}`），再跑 `--steps leadsheet`。

## 5. 生成草稿
`$PY -m guitarize arrange <slug> --arr strum` 与 `--arr finger`（记下选中的变调夹与指型）。
- 每份草稿都会用 alphaTab 构建 + 可弹性检查自检，命令输出 `自检：0 处违规`；指弹草稿会自动修复（截短按住的低音 / 填充音、丢填充或低音、旋律移八度），修复与降级写在对应 `// bar N` 注释行里。
- 草稿是生成结果，`--force` 会覆盖手改；手改之后不要再 `--force`。

## 6. 校验迭代（每份编配分别进行）
```
$PY -m guitarize verify <slug> --arr <arr>
```
读 `songs/<slug>/reports/<arr>/iter-NN/report.md`：
- **只改问题小节**：直接编辑 `arrangements/<arr>.atex` 中对应 `// bar N` 的行；需要时看 `plots/bar-NNN.png` 色度对比图。
- 低置信参考的小节不要为了"追分"去改，留给用户耳听。
- shuffle 小节里，反拍上发声的音必须写成"拍点八分 + 反拍八分"一对（余下时值用连音线），否则 alphaTab 播放不出 shuffle、校验会报 `straight_offbeat`。
- 指标达标不等于好听：逐段对照分轨起音（`guitarize.verify.metrics.onset_slots` 看鼓 / 吉他的每拍三连音格）找"停顿 / 齐奏 hits / 段落前的 break / 结尾一击"，这些地方手改成 hits 或 riff；以每小节和声相似度（scores.json 的 `harmonic_sim`）是否提高来判断手改是否更像原曲。riff 小节的逐拍和弦一致率会下降（走动低音对单一和弦标签），属预期。
- 每轮 `git commit` 一次（只提交 atex 与报告文本），再 verify。
- 停止条件：全部达标；或连续 3 轮汇总分数没有提升；或已迭代 8 轮。

## 7. 出谱与交付
- `$PY -m guitarize render <slug> --arr strum` / `--arr finger`
- 后台运行 `$PY -m guitarize serve`，用浏览器打开 `http://localhost:8765/songs/<slug>/out/<arr>.html`（本地服务支持 Range，可按小节跳到原曲）；若无浏览器面板，用无头 Chrome 截图（必须带临时 --user-data-dir）。确认谱面、点播放无报错；用 Read 看 PDF 第 1 页。
- 交给用户：网页链接（本地）、PDF / txt 路径、每份编配的汇总分数、"需人耳确认"清单（附 A/B 片段路径）。
- 若预览面板起的 serve 不监听端口，改在后台直接运行 `$PY -u -m guitarize serve --port <port>`。没有 pdftoppm 时用 macOS `sips -s format png x.pdf --out x.png` 看 PDF 第 1 页。
- 用户反馈写进 `songs/<slug>/feedback.md`，作为下一轮约束。

## 合规
- 原曲音频与分轨只在本地，不进 git、不发布；交互网页只在本地打开。
- 谱面不含歌词；网上和弦谱只取和弦。
- 不破解任何加密 / DRM；读取浏览器 cookie 需单独征得同意。
