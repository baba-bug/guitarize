# 指弹校验报告 · iter-20

**结论：❌ 未达标：和弦**

| 指标 | 本轮 | 上一轮 | 达标线 |
|---|---|---|---|
| 可弹性违规 | 0 | 0 | 0 |
| 小节数（编配 / 参考） | 62 / 62 | — | 相等 |
| 和弦一致率 | 76.7% | 89.3% | ≥ 85% |
| 旋律 F1 | 1.000 | 0.995 | ≥ 0.80 |
| 旋律精确率 | 100.0% | — |  |
| 旋律召回率（全部参考音） | 100.0% | — |  |
| 八度完全正确 | 100.0% | — |  |
| 和弦标记不符 | 5 | 7 | 仅报告 |
| 和声相似度（均值） | 0.910 | 0.880 | 仅用于定位 |
| 低置信参考拍 | 21 | — | 转人耳确认 |

## 可弹性违规（0）
- 无

## 问题小节（19）

### 第 4 小节
- 和弦一致 2/4
  - 拍 2：参考 D#/G（指法 B/D#），编配听起来是 G:maj/G:min
  - 拍 3：参考 D#/G（指法 B/D#），编配听起来是 A#:maj/A#:min
- 旋律 F1 1.00（命中 3/3，覆盖 3/3）
- 和声相似度 0.92
- 试听：snippets/bar-004_orig.wav ↔ snippets/bar-004_synth.wav
- 色度图：plots/bar-004.png

### 第 5 小节
- 和弦一致 2/4
  - 拍 3：参考 G#m（指法 Em），编配听起来是 D#:maj/D#:min
  - 拍 4：参考 G#m（指法 Em），编配听起来是 D#:min
- 旋律 F1 1.00（命中 4/4，覆盖 4/4）
- 和声相似度 0.91
- 试听：snippets/bar-005_orig.wav ↔ snippets/bar-005_synth.wav
- 色度图：plots/bar-005.png

### 第 8 小节
- 和弦一致 2/4
  - 拍 3：参考 C#（指法 A），编配听起来是 G#:min
  - 拍 4：参考 C#（指法 A），编配听起来是 G#:min
- 旋律 F1 1.00（命中 4/4，覆盖 4/4）
- 和声相似度 0.92
- 试听：snippets/bar-008_orig.wav ↔ snippets/bar-008_synth.wav
- 色度图：plots/bar-008.png

### 第 9 小节
- 和弦一致 2/3
  - 拍 3：参考 D#（指法 B），编配听起来是 D#:min
- 旋律 F1 1.00（命中 3/3，覆盖 3/3）
- 和声相似度 0.92
- 试听：snippets/bar-009_orig.wav ↔ snippets/bar-009_synth.wav
- 色度图：plots/bar-009.png

### 第 12 小节
- 和弦一致 2/4
  - 拍 1：参考 D#/G（指法 B/D#），编配听起来是 G:maj/G:min
  - 拍 4：参考 D#/G（指法 B/D#），编配听起来是 A#:min
- 旋律 F1 1.00（命中 3/3，覆盖 3/3）
- 和弦标记不符：拍 1 标记 B7/D#，实际音听起来是 G:maj/G:min
- 和声相似度 0.93
- 试听：snippets/bar-012_orig.wav ↔ snippets/bar-012_synth.wav
- 色度图：plots/bar-012.png

### 第 13 小节
- 和弦一致 2/3
  - 拍 3：参考 G#m（指法 Em），编配听起来是 D#:maj/D#:min
- 旋律 F1 1.00（命中 4/4，覆盖 4/4）
- 和声相似度 0.94
- 试听：snippets/bar-013_orig.wav ↔ snippets/bar-013_synth.wav
- 色度图：plots/bar-013.png

### 第 15 小节
- 和弦一致 2/4
  - 拍 2：参考 B/D#（指法 G/B），编配听起来是 D#:min
  - 拍 4：参考 B/D#（指法 G/B），编配听起来是 F#:maj/F#:min
- 旋律 F1 1.00（命中 5/5，覆盖 5/5）
- 和声相似度 0.94
- 试听：snippets/bar-015_orig.wav ↔ snippets/bar-015_synth.wav
- 色度图：plots/bar-015.png

### 第 20 小节
- 和弦一致 2/4
  - 拍 3：参考 B（指法 G），编配听起来是 G#:min
  - 拍 4：参考 B（指法 G），编配听起来是 G#:min
- 和声相似度 0.90
- 试听：snippets/bar-020_orig.wav ↔ snippets/bar-020_synth.wav
- 色度图：plots/bar-020.png

### 第 21 小节
- 和弦一致 2/4
  - 拍 2：参考 E（指法 C），编配听起来是 G#:min
  - 拍 4：参考 F#（指法 D），编配听起来是 A#:min
- 和声相似度 0.96
- 试听：snippets/bar-021_orig.wav ↔ snippets/bar-021_synth.wav
- 色度图：plots/bar-021.png

### 第 33 小节
- 和弦一致 2/4
  - 拍 1：参考 G#m（指法 Em），编配听起来是 C#:maj/C#:min
  - 拍 4：参考 G#m（指法 Em），编配听起来是 D#:maj/D#:min
- 旋律 F1 1.00（命中 3/3，覆盖 3/3）
- 和弦标记不符：拍 1 标记 Em，实际音听起来是 C#:maj/C#:min
- 和声相似度 0.93
- 试听：snippets/bar-033_orig.wav ↔ snippets/bar-033_synth.wav
- 色度图：plots/bar-033.png

### 第 34 小节
- 和弦一致 2/4
  - 拍 3：参考 C#（指法 A），编配听起来是 G#:min
  - 拍 4：参考 C#（指法 A），编配听起来是 G#:min
- 旋律 F1 1.00（命中 5/5，覆盖 5/5）
- 和声相似度 0.89
- 试听：snippets/bar-034_orig.wav ↔ snippets/bar-034_synth.wav
- 色度图：plots/bar-034.png

### 第 35 小节
- 和弦一致 2/4
  - 拍 2：参考 D#（指法 B），编配听起来是 B:maj/G#:min
  - 拍 4：参考 D#（指法 B），编配听起来是 A#:maj/A#:min
- 旋律 F1 1.00（命中 5/5，覆盖 5/5）
- 和声相似度 0.89
- 试听：snippets/bar-035_orig.wav ↔ snippets/bar-035_synth.wav
- 色度图：plots/bar-035.png

### 第 36 小节
- 和弦一致 2/4
  - 拍 2：参考 E（指法 C），编配听起来是 C#:min
  - 拍 4：参考 E（指法 C），编配听起来是 B:maj
- 旋律 F1 1.00（命中 4/4，覆盖 4/4）
- 和声相似度 0.85
- 试听：snippets/bar-036_orig.wav ↔ snippets/bar-036_synth.wav
- 色度图：plots/bar-036.png

### 第 41 小节
- 和弦一致 2/3
  - 拍 3：参考 G#m（指法 Em），编配听起来是 D#:maj/D#:min
- 旋律 F1 1.00（命中 5/5，覆盖 5/5）
- 和声相似度 0.93
- 试听：snippets/bar-041_orig.wav ↔ snippets/bar-041_synth.wav
- 色度图：plots/bar-041.png

### 第 43 小节
- 和弦一致 2/4
  - 拍 1：参考 B/D#（指法 G/B），编配听起来是 D#:min
  - 拍 2：参考 B/D#（指法 G/B），编配听起来是 D#:min
- 旋律 F1 1.00（命中 4/4，覆盖 4/4）
- 和弦标记不符：拍 1 标记 G/B，实际音听起来是 D#:min
- 和声相似度 0.92
- 试听：snippets/bar-043_orig.wav ↔ snippets/bar-043_synth.wav
- 色度图：plots/bar-043.png

### 第 48 小节
- 和弦一致 2/4
  - 拍 2：参考 D#/G（指法 B/D#），编配听起来是 G:maj/G:min
  - 拍 4：参考 D#/G（指法 B/D#），编配听起来是 A#:min
- 旋律 F1 1.00（命中 3/3，覆盖 3/3）
- 和声相似度 0.96
- 试听：snippets/bar-048_orig.wav ↔ snippets/bar-048_synth.wav
- 色度图：plots/bar-048.png

### 第 49 小节
- 和弦一致 2/3
  - 拍 3：参考 G#m（指法 Em），编配听起来是 D#:maj/D#:min
- 旋律 F1 1.00（命中 3/3，覆盖 3/3）
- 和声相似度 0.93
- 试听：snippets/bar-049_orig.wav ↔ snippets/bar-049_synth.wav
- 色度图：plots/bar-049.png

### 第 60 小节
- 和弦一致 2/4
  - 拍 3：参考 B（指法 G），编配听起来是 G#:min
  - 拍 4：参考 B（指法 G），编配听起来是 G#:min
- 和声相似度 0.95
- 试听：snippets/bar-060_orig.wav ↔ snippets/bar-060_synth.wav
- 色度图：plots/bar-060.png

### 第 61 小节
- 和弦一致 2/4
  - 拍 2：参考 E（指法 C），编配听起来是 G#:min
  - 拍 4：参考 F#（指法 D），编配听起来是 A#:min
- 和声相似度 0.97
- 试听：snippets/bar-061_orig.wav ↔ snippets/bar-061_synth.wav
- 色度图：plots/bar-061.png

## 需人耳确认（参考和弦低置信，共 21 拍）
- 第 9 小节：拍 4
- 第 13 小节：拍 4
- 第 16 小节：拍 3, 4
- 第 25 小节：拍 3, 4
- 第 29 小节：拍 2, 3, 4
- 第 32 小节：拍 1, 2, 3, 4
- 第 37 小节：拍 3, 4
- 第 40 小节：拍 1, 2, 3, 4
- 第 41 小节：拍 4
- 第 49 小节：拍 4
