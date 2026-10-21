---
name: guitarize
description: 把一首歌（歌名或 B 站 / 网易云链接）做成可弹的吉他谱：弹唱伴奏 + 指弹，输出交互网页 / PDF / 文本六线谱，并按"可弹性 + 与原曲差异"自动校验、逐小节迭代。当用户说"扒谱 / 做吉他谱 / 弹唱谱 / 指弹谱 / guitarize <歌>"时使用。
---

# Guitarize：原曲 → 可弹的吉他谱（含自检迭代）

工具在仓库根目录：`PY=~/anaconda3/envs/guitarize/bin/python`，命令 `$PY -m guitarize ...`。
环境未装时先跑 `scripts/setup_env.sh`。设计见 `docs/specs/2026-10-09-guitarize-design.md`。

## 0. 先问清楚（一次问完）
版本（TV size / 完整版）、水平（beginner / intermediate / advanced）、要哪些编配（弹唱 / 指弹 / 旋律谱 / 卡祖笛独奏 / 强力和弦弹唱）、输出格式。

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
`$PY -m guitarize analyze <slug>`（步骤 stems → lead → beats → chords → melody → leadsheet），再 `$PY -m guitarize summary <slug> --capo <预期变调夹>` 逐小节阅读：
- lead 步骤把人声分轨拆成主唱 `stems/lead.wav` + 伴唱 `stems/backing.wav`（becruily 的 Mel-Band Roformer 伴唱分离模型，再过一遍居中掩码；整首几分钟），模型的主唱原样（立体声、没过掩码）留作 `stems/lead_raw.wav`，旋律只从主唱识别；用的方法记在 `stems/lead.json` 与 melody.json 的 `lead_method`。模型放在 `tools/karaoke/models/`（或环境变量 `GUITARIZE_KARAOKE_DIR`）；不在或没装 audio-separator 时会警告并退回只用居中掩码（`lr-mask`，没有伴唱分轨）。只重做主唱：`--steps lead,melody,leadsheet`，不必重跑 Demucs。
- melody 步骤默认用旋律 v7（melody.json 的 `method: v7:game+signal`）：GAME（唱歌 → 音符模型，`tools/game/`，或环境变量 `GUITARIZE_GAME_DIR`）在 `lead_raw.wav` 上分音，再用主唱信号去掉伴唱漏音、把没换字的同音并成一个长音（只拨一下）、换音不换字记 legato、按当时和弦吸附音高、量化到三连音格。`song.yaml` 写 `language: ja`（en / ja / yue / zh）给 GAME 分音用。GAME 分音缓存在 `analysis/game_notes.csv`，crepe 基频照旧在 `analysis/vocal_f0.npz`。没有 `lead_raw.wav` / `backing.wav`（模型不可用）或 GAME 不可用时会警告并退回 crepe 逐格量化（`method: crepe-quantize`）。
- **每版旋律都要和原唱比**：改了旋律（重跑 melody、调参数、换算法）之后跑 `$PY -m guitarize audit-melody <slug> [--melody 旧版.json ...]`，对照这首歌自己的模型主唱 + 伴唱 + crepe，一版一行：`coverage` 覆盖、`pitch_ok` / `wrong` 音准、`backing_or_empty` 伴唱音、`chopped_held` 切碎的长音、`onset_recall` / `pluck_ms` 节奏、`repeat_f1` 同名段落（如两遍サビ）是否一致；写进 `reports/melody_audit.json`。新版要在伴唱音、切碎的长音、重复段一致上不比旧版差，再往下做。表后面还按小节列出**漏音候选**（`missing`：主唱稳稳唱着、谱上没有或写的音差 ≥ 0.75 半音、≥ 80 毫秒）——用户最在意漏音，交给人耳之前先逐个看：≥ 1 格的一般已被 v7 的最后核对补上，剩下的多是 ≤ 140 毫秒的短音、唱在两个音中间或滑音；人耳确认是漏音的用 `overrides.melody` 改那一小节。
- 小节数、速度、Shuffle 是否合理；`short_bars` 应为空（否则用 `overrides.downbeat_shift` 调整）。
- 低置信（`??`）小节记下来，稍后列入"需人耳确认"。
- 低置信（`??`）多集中在 TV 版剪辑处、尾奏；不要为追分修改这些小节。
- 速度倍速 / 半速（如 167 BPM 的 shuffle 被识别成 83）：在 `overrides.tempo_bpm` 写正确速度后重跑 `--steps beats,chords,melody,leadsheet`；`beats.json` 的 `tempo_stability` 波动 > 6% 会警告。
- 目前只支持 4/4；`short_bars` 非空说明有 2/4、3/4 小节（TV 版剪辑常见），`overrides.downbeat_shift` 只用于强拍相位整体错位。
- 网上谱写成 N.C. / "只有贝斯"的小节（间奏 / 尾奏 riff）会得到 N：用贝斯分轨逐格音高（`librosa.pyin`，按三连音格）核对，再把真实和声写进 `overrides.chords`（`{bar: "B" | "E F#" | [逐拍 4 个]}`，记为 manual 置信度，重跑 `--steps leadsheet` 后仍保留）。
- 把段落写进 `song.yaml` 的 `sections`（如 `{1: Intro, 9: Aメロ, 17: サビ}`），再跑 `--steps leadsheet`。

## 5. 生成草稿
`$PY -m guitarize arrange <slug> --arr strum` 与 `--arr finger`（记下选中的变调夹与指型）。
- 每份草稿都会用 alphaTab 构建 + 可弹性检查自检，命令输出 `自检：0 处违规`；指弹草稿会自动修复（先丢 / 截短合音，再截短按住的低音、丢低音；旋律的音高和时值不动，换把只给旋律音换弦），修复与降级写在对应 `// bar N` 注释行里。
- 指弹默认是完整旋律（`overrides.finger_style: full`）：旋律声部就是旋律谱的每一个音（同样的节奏、三连音、h / sl，只有指法可以不同）；伴奏声部第 1 拍弹和弦低音、第 3 拍换和弦弹新低音、不换弹五音（交替低音），第 2、4 拍旋律不在拍上起音时弹两个和弦内音的短音，拍上 / 摇摆反拍唱满 1 拍的长音下面加三度 / 六度的双音。所以先把旋律谱核对准再编指弹。verify 按全部参考音评指弹旋律；第 3、4 拍低音是五音时逐拍和弦推断常听成属和弦，和弦一致率会比旧写法低，以和声相似度和人耳为准。旧的骨架写法（骨架旋律 + 低音 / 填充，适合初级）用 `finger_style: skeleton`。
- 弹唱默认 `overrides.strum_rhythm: guitar`：变调夹、开放和弦指型照旧，节奏型与强力和弦谱同一套、按乐句只看吉他分轨选（`verse` / `verse_end` / `chorus` / `hit`，写在 `// bar N · 节奏型 …` 注释里）；D / A 扫整个指型（A 加重音 `ac`），M 只闷指型最低 3 根发声弦（`pm`），U 上扫 1–3 弦、低音弦用连音线接着响，d 轻扫；刷弦 `bd 60` / `bu 60`、下扫 `dy f`、上扫 / 轻扫 `dy mp`。verify 的节奏只对照吉他分轨、按力度加权。旧的弹唱（鼓 + 吉他选型、`sd` / `su`、没有力度）用 `strum_rhythm: drums`。`strum_pattern` 整首一个型：guitar 节奏两套型名都能写（旧的型也刷弦、写力度），drums 只有旧的型。
- 草稿是生成结果，`--force` 会覆盖手改：把手改写成 `songs/<slug>/hand_edits.py`（按 `// bar N` 替换行，见 なんもねえ 的例子），生成器改进后 `arrange --force` 再重放手改脚本，再 verify。
- 指弹旋律默认整曲一个八度（保持人声高低关系；段落间换八度会让旋律在交界处"突然掉下去"）。确需分段换八度时用 `overrides.melody_octave: section` 或 `{小节: 12}`。
- 指弹谱自动标出该按的和弦指型（指型名 + 和弦图）；伴奏 mf、旋律 f。

### 卡祖笛独奏（`--arr kazoo`，"卡祖笛来独奏"）
- `$PY -m guitarize arrange <slug> --arr kazoo`，并把 `kazoo` 加进 `song.yaml` 的 `arrangements`。卡祖笛靠哼唱发声，没有指法：谱上是确认过的旋律（leadsheet 的 melody，与旋律谱同一份、同一个节奏）按**实际音高**写——不加变调夹、不移八度，就是歌手唱的八度。所以先把旋律谱核对准。
- 谱面：五线谱 + 简谱（`\staff {score numbered}`，音高谱表 `\tuning (piano)`，音符写音名 `C#4.4`，升降号跟调号走），音色 GM 59（Muted Trumpet）。连奏（legato）的音写连线（`slur`，不重新吐音），其余每个音重新吐音。每小节第 1 拍标和弦名（实际音高、不画和弦图，转入无和弦标 N.C.），段落名写成文字。简谱不加点的八度放在旋律最集中的八度：男声旋律用 8vb 谱号（记谱高八度，播放音高不变）。
- 手改 atex 时：连音线延续写 `C#4{-}.4`（`-.4` 在音高谱表上是语法错误），和别的音符效果同一个花括号 `C#4{- slur s1}`；连线按 id 配对、id 不能重用；`\section "名"` 后面紧跟音名会被当成参数（写 `\section ("名")`）；alphaTab 1.8.4 的简谱画不了全休止 `r.1`（写 `r.2 r.2`）。
- `verify --arr kazoo`：不查可弹性，只查记谱（小节时值、Shuffle 反拍）；旋律 F1（全部参考音，达标线 0.9）+ 小节数相等即达标；另报音域（最低 / 最高音、跨度）和换气提醒（连续超过 `thresholds.breath_seconds`（默认 8）秒没有休止的乐句，按小节列出）。不对照音频，没有试听片段。
- `render --arr kazoo` 的 txt 是文本简谱（`1=B 4/4`，每行一小节，记法写在文件开头：`'` / `,` 高低八度、`-` 接着响、`0` 休止、`³` 三连音拍）；网页同样能选参考音频、从当前小节播放、光标跟随。

### 强力和弦弹唱（`--arr power`，原曲是失真电吉他扫强力和弦时）
- `$PY -m guitarize arrange <slug> --arr power`，并把 `power` 加进 `song.yaml` 的 `arrangements`。按**实际音高**、不夹变调夹：和弦按根音换成强力和弦 X5（斜杠和弦按和弦根音不按低音：D#/G → D#5、B/D# → B5；小三、七和弦一样是 X5），根音在 6 / 5 弦 0–9 品（加五音和高八度，f = 0 是开放 E5 / A5），每个和弦全曲一个指型，选换和弦时把位移动最少的一套；和弦图在 5 品以上从最低按弦品起画。
- 节奏型按段落、**只看吉他分轨**的起音选（鼓 + 吉他一起选会被鼓带走）：`verse`（1 闷、2 空弦重音、3 闷 + 反拍短促上扫、4）、`verse_end`（1、2、2&，然后延音）、`chorus`（1 重扫 + 延音的音墙，2、3& 轻扫）、`hit`（一击 + 余音）。段落在无和弦小节处断开，段落内前后差得多时切一刀（各至少 3 小节）；最像的型相关度也不到 0.5（持续的音墙）就是 hit。选的型写在 `// bar N · 节奏型 …` 注释里；不对时用 `overrides.power_patterns: {小节: 型}`（用到该乐句末尾）。
- 记谱：刷弦 `bd 60` / `bu 60`（只有刷弦才把各弦错开，`sd` / `su` 只是箭头）、下扫 `dy f`、上扫和轻扫 `dy mp`、空弦重音 `ac`、闷音 `pm`——`ac` / `pm` 是**音符效果**（`(4.3{pm} 4.4{pm} 2.5{pm}).4 {bd 60 dy f}`；写成拍效果 `{pm}` 报 Unrecognized property）。`dy` 会沿用到下一个写了 `dy` 的拍，手改时每下都写上。riff / break / 结尾一击照弹唱谱手改（hand_edits.py 的 `power`）。
- `verify --arr power`：与弹唱同一套，和弦与和弦标记按根音对；节奏只对照吉他分轨、按力度加权（闷音 0.3、mp 0.66），报告末尾按段落列节奏相关度。txt 多一行 `PM` / `>`。

## 6. 校验迭代（每份编配分别进行）
```
$PY -m guitarize verify <slug> --arr <arr>
```
读 `songs/<slug>/reports/<arr>/iter-NN/report.md`：
- **只改问题小节**：直接编辑 `arrangements/<arr>.atex` 中对应 `// bar N` 的行；需要时看 `plots/bar-NNN.png` 色度对比图。
- 低置信参考的小节不要为了"追分"去改，留给用户耳听。
- shuffle 小节里，反拍上发声的音必须写成"拍点八分 + 反拍八分"一对（余下时值用连音线），否则 alphaTab 播放不出 shuffle、校验会报 `straight_offbeat`。
- 休止同理：拍点八分后面从反拍开始、跨过拍线的休止，先写本拍余下的 `r.8` 再写整拍（`0.1.8 r.8 r.2`，不写 `0.1.8 r.2 r.8` / `0.1.8 r.4 {d}`），否则前面的八分按直拍（半拍）播放——校验不报，但听得出来；生成器已这样写。
- 唱在 +1/3 的音写不成八分对：整拍写成三连音（`.8 {tu 3}` = 1/3 拍、`.4 {tu 3}` = 2/3 拍，跨出这一拍的部分用普通时值连音线接上），指弹生成器的旋律已这样写。指型 / 力度和 `tu 3` 写在同一个花括号里（`{ch "C" tu 3}`），手改脚本按指型名匹配时别要求花括号里只有它。
- 旋律谱的滑音（melody.json 的 `legato`：同一音节里滑到下一个音，不重新拨弦）：两个音在同一根弦上时前一个音写音符效果——差 1–2 品（或有一头是空弦）`h`，3 品以上 `sl`；尾滑（`slide_out`）写 `sod` / `sou`（只写在按弦的音上：空弦滑不出去）。音符效果紧跟 品.弦、写在时值前面：`5.3{h}.4 {d}`、和弦里 `(5.3{h} 3.2).4`、连音线延续 `-.3{h}.8`（挂在紧挨着下一个音的那一段上）；写成 `5.3.4 {h}` 会被当成拍效果报错。手改脚本按品.弦匹配时要允许后面跟着 `{…}`。
- 用户优先要求接近原曲：手改以逐小节和声相似度不降为准（为可弹性而降低接近度要先问用户）；骨架写法的指弹伴奏太薄的小节补指型里的空弦 / 顶音（如 C 指型补空弦 G、B7 指型补 1 弦 2 品）——完整旋律写法的合音由生成器配，不用补。
- 指标达标不等于好听：逐段对照分轨起音（`guitarize.verify.metrics.onset_slots` 看鼓 / 吉他的每拍三连音格）找"停顿 / 齐奏 hits / 段落前的 break / 结尾一击"，这些地方手改成 hits 或 riff；以每小节和声相似度（scores.json 的 `harmonic_sim`）是否提高来判断手改是否更像原曲。riff 小节的逐拍和弦一致率会下降（走动低音对单一和弦标签），属预期。
- 每轮 `git commit` 一次（只提交 atex 与报告文本），再 verify。
- 停止条件：全部达标；或连续 3 轮汇总分数没有提升；或已迭代 8 轮。

## 7. 出谱与交付
- `$PY -m guitarize render <slug> --arr strum` / `--arr finger`
- 重新生成前想留旧版对比：把 `arrangements/<arr>.atex` 复制成 `<arr>_old.atex`，`render <slug> --arr <arr>_old`（变体 `<编配>_<标签>` 按所属编配的设置渲染到 `out/<arr>_old.*`，总页面里排在所属编配后面，叫「弹唱（旧版）」；只有 render 认变体）。
- 后台运行 `$PY -m guitarize serve`，用浏览器打开 `http://localhost:8765/songs/<slug>/out/<arr>.html`（本地服务支持 Range，可按小节跳到原曲）；若无浏览器面板，用无头 Chrome 截图（必须带临时 --user-data-dir）。确认谱面、点播放无报错；用 Read 看 PDF 第 1 页。
- 交给用户：网页链接（本地）、PDF / txt 路径、每份编配的汇总分数、"需人耳确认"清单（附 A/B 片段路径）。
- 若预览面板起的 serve 不监听端口，改在后台直接运行 `$PY -u -m guitarize serve --port <port>`。没有 pdftoppm 时用 macOS `sips -s format png x.pdf --out x.png` 看 PDF 第 1 页。
- 用户反馈写进 `songs/<slug>/feedback.md`，作为下一轮约束。

## 合规
- 原曲音频与分轨只在本地，不进 git、不发布；交互网页只在本地打开。
- 谱面不含歌词；网上和弦谱只取和弦。
- 不破解任何加密 / DRM；读取浏览器 cookie 需单独征得同意。
