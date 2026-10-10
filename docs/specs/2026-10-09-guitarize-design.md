# Guitarize 设计文档：从原曲音频到可弹的吉他谱（弹唱 + 指弹），含自动校验迭代

- 日期：2026-10-09
- 状态：待用户评审
- 首个目标曲：忘れらんねえよ「なんもねえ」（TV 动画『ヤニねこ』OP），TV size

---

## 1. 目标与范围

**做什么**

1. 输入一首歌（歌名，或 B 站 / 网易云链接）、版本（TV size / 完整版）和难度等级。
2. 产出两份编配：
   - **弹唱伴奏**：和弦 + 节奏型 + 变调夹保持原调
   - **指弹**：主旋律 + 低音 + 少量填充
3. 每份编配输出三种格式：**网页交互谱**（可播放、可对照原曲）、**PDF**、**文本六线谱**。
4. **自动自检并迭代**：可弹性检查 + 与原曲的逐小节差异评分 → 问题报告 → 修改 → 复检，直到达标或收敛，最后交给用户耳听确认。
5. **可复用**：命令行工具 + Claude Code skill，换一首歌只改配置，不改代码。

**本期不做（非目标）**

- 还原原曲乐队吉他声部（Riff / Solo / 失真音色）
- 完整版：等 TV size 跑通后再做，流程相同
- 谱内歌词：版权原因，只留空白栏位
- 简化现有谱子：作为后续独立项目，见 §11
- 零人工的全自动：编配取舍由 Claude 决定，最终听感由用户确认

## 2. 首曲已知信息与用户设定

| 项 | 值 | 来源 / 置信度 |
|---|---|---|
| 调性 | B 大调 | ChordWiki 开头和弦为 B、E、D#、C#，**待音频分析确认** |
| 速度 / 律动 | BPM≈168，Shuffle 4/4 | ChordWiki 标注，**待确认** |
| 参考和弦谱 | ChordWiki「なんもねえ」 | 网友可编辑，只作为交叉参考 |
| 用户水平 | 中级：横按没问题，弹过简单指弹 | 用户 |
| 输出格式 | 网页交互谱 + PDF + 文本六线谱 | 用户 |
| 伴奏类型 | 弹唱伴奏（用变调夹保持原调） | 用户 |

## 3. 总体架构

流水线分 5 个阶段。每个阶段的产物都落盘缓存，修改某一阶段后只需从该阶段重跑。

```
song.yaml（配置）
   │
   ▼
[1 获取 acquire] Chrome 内定位音源 → 下载 → 标准化 wav → 截取 TV size
   │  source/original.wav
   ▼
[2 分析 analyze] 分轨 → 节拍 / 小节线 → 调性 → 参考和弦（三方投票）→ 人声旋律 → 段落
   │  analysis/{stems/, beats.json, chords.json, melody.json, leadsheet.json}
   ▼
[3 编配 arrange] 草稿生成器（确定性代码）出 v1 → Claude 逐小节审阅修改
   │  arrangements/{strum,finger}.atex（alphaTex，源文件）
   ▼
[4 渲染 render] alphaTex → 解析为谱面模型 JSON / MIDI / SVG→PDF / ASCII / 网页
   │  out/{strum,finger}.{html,pdf,txt}, build/{*.json,*.mid,*.wav}
   ▼
[5 校验 verify] 可弹性规则 + 与参考的逐小节评分 → 报告
   │  reports/iter-NN/{report.md, scores.json, snippets/, plots/}
   └──► 未达标：Claude 改问题小节 → 回到 [4]（每轮几秒）
        达标 / 收敛：交给用户耳听 → 反馈进入下一轮
```

阶段之间只通过上面列出的文件通信。每个阶段可以单独运行、单独测试。

## 4. 各阶段设计

### 4.1 配置 `song.yaml`

```yaml
slug: nanmonee
title: なんもねえ
artist: 忘れらんねえよ
version: tv              # tv | full
level: intermediate      # beginner | intermediate | advanced → 对应 profiles.yaml
arrangements: [strum, finger]
outputs: [html, pdf, txt]
source:
  url: null              # 获取阶段填入
  kind: null             # bilibili | netease
  trim: null             # [start_s, end_s]，TV size 截取区间
reference_charts:        # 可选，网上和弦谱，仅作交叉参考
  - https://ja.chordwiki.org/wiki/なんもねえ
overrides: {}            # 人工覆盖项，如 {tempo_bpm: 168, key: B, capo: 4}
```

### 4.2 获取（acquire）

- **定位**：用 Claude in Chrome（用户已登录的 Chrome）在 B 站 / 网易云搜索音源。TV size 优先选官方无字幕 OP（ノンクレジット OP）或官方上传。选定的 URL 写回 `song.yaml`。
- **下载前向用户确认**：说明来源、文件名和大致大小。
- **B 站**：用 `yt-dlp` 只下音频流。只有在需要登录才能拿到音质时，才在用户明确同意后使用 `--cookies-from-browser chrome`（macOS 会弹钥匙串授权）。
- **网易云**：只有网页播放器能正常获取音频流时才使用（从浏览器网络请求里拿到媒体地址）。VIP 加密格式（.ncm 等）**不解密**，改用 B 站音源。
- **标准化**：用 ffmpeg 转成 44.1kHz 立体声 wav；按 `trim` 截取；记录 URL、时间区间和 sha256 到 `source/meta.json`。

### 4.3 分析（analyze）

| 步骤 | 方法 | 产物 |
|---|---|---|
| 分轨 | Demucs `htdemucs_6s`：人声 / 鼓 / 贝斯 / 吉他 / 钢琴 / 其他（Apple M4 的 MPS 加速） | `stems/*.wav` |
| 主唱 / 伴唱 | 人声分轨 → becruily 的 Mel-Band Roformer 伴唱分离模型（audio-separator）→ 主唱再过一遍 L/R 居中掩码；模型不可用时只用居中掩码 | `stems/lead.wav`、`stems/backing.wav`、`stems/lead.json` |
| 节拍 / 小节线 | `beat_this` 模型。检查速度稳定性，判断 Shuffle 律动。`overrides` 可强制指定速度 | `beats.json` |
| 调性 | 色度谱 + Krumhansl 调性模板，再和和弦结果交叉确认 | 写入 `leadsheet.json` |
| 参考和弦 | **三方投票**，见下文 | `chords.json` |
| 人声旋律 | 主唱分轨 → 基频提取（优先 RMVPE，备选 torchcrepe）→ 稳定音高切段 → 按节拍网格量化（Shuffle 用三连音网格）→ 修正八度跳错、标出调外音 | `melody.json` |
| 段落 | 按小节的色度自相似矩阵找重复段落，再由 Claude 结合参考谱标注 Intro / A / B / サビ | 写入 `leadsheet.json` |

**参考和弦的三方投票**

- 投票来源：
  - (a) 模板匹配：去掉人声和鼓的和声分轨 → 逐拍 CQT 色度 → 大三 / 小三 / 七和弦模板 → HMM 平滑
  - (b) 预训练和弦识别模型（如果能在本机装上；装不上就退化为两方投票）
  - (c) 网上和弦谱：只提取和弦序列，丢弃歌词，再与小节对齐
- 置信度按当拍实际可用的来源数 N 判定（不是每首歌都有网上和弦谱，模型也可能装不上）：
  - N = 3：三方一致 → `high`；两方一致 → `medium`；全不一致 → `low`
  - N = 2：两方一致 → `high`；不一致 → `low`
  - N = 1（只有模板匹配）：HMM 后验概率 ≥ 0.7 → `medium`，否则 `low`
- `low` 的拍**不参与自动评分**，统一列入"需人耳确认"。

### 4.4 编配（arrange）

**变调夹 / 指法调选择**：对变调夹 0–7 品 × 常用指法调（C、G、D、A、E 及其关系小调）逐一打分。

- 弹唱：开放和弦比例、横按个数、难度档的限制。
- 指弹：额外考虑旋律音域能否落在 1–3 弦、12 品以内。

首曲预期结果是**夹 4 品按 G 调**。B→G、E→C、D#→B7、C#→A、G#m→Em，基本都是开放和弦。最终以打分结果为准。

**弹唱草稿生成器**

- 和弦：取自 `leadsheet.json` 的逐拍和弦，换算到指法调，从和弦指型库选形（受难度档约束）。
- 节奏型：每个段落从节奏型模板库（Shuffle 八分为主）中选一个。选法是比较模板重音和原曲在三连音八分格上的起音强度（来自鼓和吉他分轨），取相关性最高的。

**指弹草稿生成器**

- 旋律骨架化：保留强拍音、长音和乐句首尾音；按难度档限制密度，删掉装饰音。
- 八度安置：把旋律放到 1–3 弦的音域。
- 低音：每小节第 1 拍（必要时加第 3 拍）放根音或五音。
- 旋律休止处用和弦内音填充。
- 指法分配：用 Viterbi 动态规划为每个音选（弦，品）。代价包括手指跨度、换把距离、同弦冲突，空弦音给奖励。

**Claude 审阅**：草稿只是起点。Claude 直接编辑 alphaTex 文件，一小节一行，带 `// bar 12 [サビ]` 注释，处理音乐性取舍。

**难度档 `profiles.yaml`**：beginner / intermediate / advanced 三档参数，见 §4.6。生成器和校验器共用这一份参数。§11 的简化项目也会复用它。

### 4.5 渲染（render）

- **源文件**：alphaTex（`.atex`）。头部写元信息（标题、速度、变调夹、调弦、三连音感觉记号），一小节一行。
- **解析**：Node 脚本调用 `@coderline/alphatab`，把 alphaTex 解析成谱面模型 → 导出：
  - `build/<arr>.json`：小节 → 拍 → 音（弦、品、开始拍、时值、连音线等）和和弦名，供 Python 校验使用
  - `build/<arr>.mid`
  - `out/<arr>.gp`：附带产物，可用 Guitar Pro / TuxGuitar 打开
- **PDF**：alphaTab 在 Node 里渲染 SVG → 套上 Bravura 字体的 HTML → 无头 Chrome 打印成 PDF，与网页谱面一致。spike 已验证，不需要 MuseScore。
- **文本六线谱**：Python 从 `build/<arr>.json` 生成。每行 4 小节，带小节号、段落标记和和弦名。
- **合成音频**：MIDI → FluidSynth + GM 音色库（钢弦吉他）→ `build/<arr>.wav`，供校验使用。
- **网页交互谱** `web/viewer.html`：
  - alphaTab 从 jsDelivr 加载，固定版本号
  - 功能：播放 / 暂停、光标跟随、调速、选段循环
  - **原曲对照**：页面内嵌原曲 `<audio>`。用 `beats.json` 的"小节 → 原曲时间"映射，点击任一小节即可跳到原曲对应位置播放，可在合成音和原曲之间切换
  - 只用本地 HTTP 服务打开，不发布在线页面
- 谱面记法：Shuffle 记为直八分音符，加"三连音感觉"标记（`\tf triplet-8th`），这是 J-rock 谱子的通行写法。指弹谱用两个声部：voice 1 为旋律（符干向上），voice 2 为低音和填充（符干向下）。
- spike 结论（2026-10-09，alphaTab 1.8.4）：变调夹、三连音感觉、和弦图、扫弦方向、刷弦、三连音、段落、多声部、连音线、附点都能正确解析；导出的 MIDI 已包含变调夹移调和 swing；Guitar Pro 导出和 SVG 渲染可用；alphaTab 自带 sonivox.sf2 音色库，可直接用于 FluidSynth 合成。

### 4.6 校验（verify）

**A. 可弹性：硬规则，任一违反即不合格**（intermediate 档参数）

| 规则 | intermediate 档 |
|---|---|
| 同一弦同时出两个音 | 禁止 |
| 左手跨度：一个四品格的把位范围（最高品 − 最低品，不含空弦） | ≤ 3 |
| 同时按弦手指数（横按计 1 指） | ≤ 4 |
| 最高品位 | ≤ 12 |
| 指弹同时拨弦数（p / i / m / a） | ≤ 4 |
| 快速换把：相邻两次按弦间隔不到一拍、手位却移动超过 N 品（按实际秒数算，168 BPM 下一拍约 0.36 秒） | N = 4，超过即违规 |

**软指标**（只报告、不判失败）：每小节难度分、整曲横按和弦个数、换和弦频率。

**B. 与原曲的差异：逐小节评分，0–1**

| 指标 | 定义 | 适用 |
|---|---|---|
| 和弦一致率 | 编配和弦由**实际发声的音符**推断（每拍窗口内的音名集合，含延音，做模板匹配），不用谱上写的和弦名。换算回音乐会音高后，与参考和弦按"根音 + 大小三"逐拍比对。`low` 置信拍不计入分母。另做一项检查：谱上写的和弦名与实际音符推断结果不符时报错 | 两者 |
| 旋律 F1 | 指弹旋律声部（alphaTex 的 voice 1）与参考旋律逐音符匹配：起音容差为 1/3 拍，**音名相同即算对，允许八度差**。精确率以编配的全部旋律音为分母；召回率以参考旋律的**骨架音**为分母（骨架规则与编配生成器共用同一个函数，因为指弹本来就只保留骨架音）。另报告"八度完全正确"比例 | 指弹 |
| 和声相似度 | 合成音频与原曲和声分轨（去人声和鼓），逐小节 CENS 色度均值的余弦相似度。用 `beats.json` 把两边按小节对齐 | 两者 |
| 节奏相关度 | 扫弦重音向量与原曲鼓 + 吉他分轨在三连音八分格上的起音强度的相关系数，按段落平均 | 弹唱 |

**C. 达标线**（初始值，可在 `song.yaml` 中覆盖）

- 可弹性：0 硬违规（两者）
- 弹唱：和弦一致率 ≥ 90%，节奏相关度只报告不卡线
- 指弹：旋律 F1 ≥ 0.80，和弦一致率 ≥ 85%
- 和声相似度：只用于定位问题小节，不卡线

**D. 报告** `reports/iter-NN/`

- `scores.json`：逐小节全部指标，以及全曲汇总
- `report.md`：
  - 汇总表，与上一轮对比
  - 问题小节列表：哪项不达标、参考值是什么、编配是什么
  - "需人耳确认"列表
- `snippets/bar-012_{orig,synth}.wav`：问题小节的原曲 / 合成片段，**给用户听**
- `plots/bar-012.png`：原曲与合成的色度对比图，**给 Claude 看**
- `reports/history.csv`：每一轮的汇总分数，用于判断是否收敛

### 4.7 迭代协议

1. 生成器出 v1，然后渲染 + 校验。
2. Claude 读 `report.md`，**只修改问题小节**，在 git 中提交为新版本，然后渲染 + 校验。
3. 停止条件，满足任一即停：
   - 全部达标
   - 连续 3 轮汇总分数没有提升
   - 已迭代 8 轮
4. 交付给用户：网页交互谱，以及问题小节的 A/B 片段和"需人耳确认"清单。
5. 用户反馈（在对话里说，或写进 `songs/<slug>/feedback.md`，如"第 12 小节不像"）作为新约束进入下一轮。

**原则**：参考数据为 `low` 置信的地方不自动"修正"编配，避免把错误的参考当成标准答案。

## 5. 数据格式（阶段间契约）

时间统一用两种量：**秒**，以及**从第 1 小节第 1 拍起算的拍数**（可以是分数，三连音格为 1/3）。

```jsonc
// beats.json
{ "tempo_bpm": 168.2, "feel": "shuffle", "time_signature": [4, 4],
  "beats": [ { "t": 0.512, "bar": 1, "beat": 1 }, ... ] }

// chords.json
{ "segments": [ { "bar": 1, "beat": 1, "dur_beats": 2, "label": "B",
                  "votes": { "template": "B", "model": "B", "web": "B" },
                  "confidence": "high" } ] }

// melody.json
{ "grid": "triplet8", "notes": [ { "start_beats": 16.667, "dur_beats": 0.333,
                                   "midi": 71, "conf": 0.92, "out_of_key": false } ] }

// leadsheet.json：合并上述三者，加上调性、段落、小节 → 原曲时间映射
{ "key": "B", "mode": "major", "bars": [ { "bar": 1, "section": "Intro",
   "t_start": 0.512, "t_end": 1.94, "chords": [...], "melody": [...] } ] }

// build/<arr>.json：alphaTex 解析结果，校验器的输入
{ "capo": 4, "tuning": ["E4","B3","G3","D3","A2","E2"], "bars": [ { "bar": 1,
   "chord": "G", "beats": [ { "start_beats": 0, "dur_beats": 1,
     "notes": [ { "string": 6, "fret": 3 } ], "strum": "down" } ] } ] }
```

## 6. 目录结构

```
Guitar/
  guitarize/                 Python 包；CLI：guitarize {new,acquire,analyze,arrange,render,verify} <slug>
    acquire.py  analyze/  arrange/  render/  verify/  profiles.yaml
  tools/alphatab/            Node 脚本：alphaTex 解析、导出 JSON / MIDI / GP / SVG
  web/viewer.html            交互谱模板
  tests/                     单元测试 + 闭环测试
  songs/<slug>/
    song.yaml  feedback.md
    source/                  原曲音频（不入 git）
    analysis/                stems/（不入 git）、beats / chords / melody / leadsheet.json
    arrangements/            strum.atex、finger.atex
    build/                   中间产物（不入 git）
    reports/iter-NN/         报告（片段音频不入 git）
    out/                     html、pdf、txt、gp
  .claude/skills/guitarize/SKILL.md
  docs/specs/
```

## 7. 环境与依赖

- conda 环境 `guitarize`（Python 3.11，避开当前 3.13 的兼容问题），包括：
  - torch（MPS）、demucs、beat_this、librosa、mir_eval、soundfile、pretty_midi
  - pyyaml、yt-dlp、pytest
  - 基频模型（RMVPE 权重或 torchcrepe）
- Homebrew：`fluid-synth`、`gh`；PDF 用系统已安装的 Google Chrome（无头模式）
- Node：`@coderline/alphatab`，装在 `tools/alphatab/`
- 磁盘：约 3–5GB（环境 + 模型权重）

## 8. 测试策略

- **单元测试**（pytest，先写测试再实现）：
  - 指法分配（Viterbi）、可弹性规则、各项指标（手工构造的小例子）
  - 变调夹 / 指法调选择、ASCII 渲染
  - alphaTex → 谱面模型 JSON 的解析结果
- **闭环测试（已知答案，校准"耳朵"）**：
  - 用代码写一首 16 小节的公有领域旋律（如《小星星》），配上和弦、贝斯、简单鼓和一条"人声"主旋律，用 FluidSynth 合成音频
  - 跑分析阶段，要求还原出的节拍、和弦、旋律与真值相比达到阈值
  - 再跑编配 + 校验全流程
  - 作用：分析结果是校验的基准，它本身必须先被验证
- **首曲实跑**：「なんもねえ」TV size。

## 9. 风险与对策

| 风险 | 对策 |
|---|---|
| alphaTex 表达力不足，或 alphaTab 在 Node 下无法导出 MIDI / SVG | **已通过 spike 验证（见 §4.5）**，风险关闭 |
| 失真吉他混音下和弦识别出错 | 三方投票 + 置信度；`low` 不参与自动修改，转人耳确认 |
| 168 BPM Shuffle 下日语快速音节导致旋律切分不准 | 指弹本来就做旋律骨架化，只要求骨架音准确；八度允许差异 |
| 找不到 TV size 音源，只有完整版 | 用 OP 视频音轨与完整版做互相关，定位截取区间；都不行就请用户指定时间段 |
| Demucs / beat_this 在 MPS 上的兼容问题 | 退回 CPU 运行，TV size 只有约 90 秒，耗时可接受 |
| 指标好看但听感不对 | 指标只用来定位问题，最终由用户耳听把关；保留 A/B 片段，方便快速核对 |

## 10. 验收标准

1. 首曲两份编配：可弹性 0 硬违规；指标达标，或剩余问题小节已列明并经用户确认可以接受。
2. 三种格式均生成且可用：网页可播放、可按小节跳转原曲对照；PDF 排版正确；txt 对齐正确。
3. 闭环测试通过。
4. 用户实际弹奏、试听后确认"像原曲、能弹下来"。
5. **复用验证**：用户另选一首歌，只写 `song.yaml`、调用 skill 就能跑通到出谱，过程中不修改代码（修 bug 除外）。

## 11. 后续独立项目：谱子简化（本期不做）

用户提出：现有的谱子太难时，自动简化成更容易弹的版本。为它预留的复用点：

- **导入器**（新）：Guitar Pro / MusicXML / alphaTex / ASCII → 本项目的谱面模型 JSON
- **简化操作**（新）：
  - 换调或调整变调夹，挪到开放和弦友好的调
  - 用开放和弦或部分和弦替换横按
  - 减少声部、旋律骨架化、节奏简化
- **复用本期模块**：`profiles.yaml` 难度档、可弹性校验、指法 Viterbi、渲染、和声 / 旋律评分。参考换成"原谱"而不是原曲音频。

本期只需保证谱面模型、难度档和校验器不绑定"音频转谱"这一种输入。

## 12. 合规边界

- 下载的原曲音频和分轨只存在本地，只用于个人练习转谱，不进 git，不发布，不传给第三方。
- 不破解任何 DRM 或加密格式。
- 谱面不包含歌词。读取网上和弦谱时只提取和弦序列。
- 下载前向用户确认来源；需要读取浏览器 cookie 时单独征得同意。
- 网页交互谱只在本地打开（页面内含原曲音频）。
