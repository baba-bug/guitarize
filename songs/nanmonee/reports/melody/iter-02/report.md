# 旋律校验报告 · iter-02

**结论：✅ 达标**

> 对照的人声分轨：analysis/stems/lead.wav（相似度、试听片段、音高图）

| 指标 | 本轮 | 上一轮 | 达标线 |
|---|---|---|---|
| 可弹性违规 | 0 | 0 | 0 |
| 小节数（编配 / 参考） | 62 / 62 | — | 相等 |
| 旋律 F1 | 1.000 | 1.000 | ≥ 0.90 |
| 旋律精确率 | 100.0% | — |  |
| 旋律召回率（全部参考音） | 100.0% | — |  |
| 八度完全正确 | 100.0% | — |  |
| 人声相似度（均值） | 0.915 | 0.913 | 仅用于定位 |
| 需人耳确认（调外 / 低置信音所在小节） | 28 | — | 转人耳确认 |

## 可弹性违规（0）
- 无

## 问题小节（0）
- 无

## 需人耳确认（参考旋律里调外 / 低置信的音，共 39 个，28 小节）
- 第 4 小节：拍 1+⅓ G4（调外） · 试听：snippets/bar-004_vocal.wav ↔ snippets/bar-004_synth.wav · 音高图：plots/bar-004.png
- 第 5 小节：拍 3& A#4（置信 0.42） · 试听：snippets/bar-005_vocal.wav ↔ snippets/bar-005_synth.wav · 音高图：plots/bar-005.png
- 第 7 小节：拍 3+⅓ B3（置信 0.42） · 试听：snippets/bar-007_vocal.wav ↔ snippets/bar-007_synth.wav · 音高图：plots/bar-007.png
- 第 8 小节：拍 2& D4（调外） · 试听：snippets/bar-008_vocal.wav ↔ snippets/bar-008_synth.wav · 音高图：plots/bar-008.png
- 第 12 小节：拍 2 G4（调外） · 试听：snippets/bar-012_vocal.wav ↔ snippets/bar-012_synth.wav · 音高图：plots/bar-012.png
- 第 19 小节：拍 1+⅓ A4（调外，置信 0.50）；拍 2 G#4（置信 0.40） · 试听：snippets/bar-019_vocal.wav ↔ snippets/bar-019_synth.wav · 音高图：plots/bar-019.png
- 第 20 小节：拍 1+⅓ A#4（置信 0.35）；拍 3+⅓ B4（置信 0.41） · 试听：snippets/bar-020_vocal.wav ↔ snippets/bar-020_synth.wav · 音高图：plots/bar-020.png
- 第 21 小节：拍 1+⅓ B4（置信 0.49） · 试听：snippets/bar-021_vocal.wav ↔ snippets/bar-021_synth.wav · 音高图：plots/bar-021.png
- 第 24 小节：拍 4& E3（置信 0.44） · 试听：snippets/bar-024_vocal.wav ↔ snippets/bar-024_synth.wav · 音高图：plots/bar-024.png
- 第 27 小节：拍 4 C#4（置信 0.49）；拍 4+⅓ A3（调外） · 试听：snippets/bar-027_vocal.wav ↔ snippets/bar-027_synth.wav · 音高图：plots/bar-027.png
- 第 33 小节：拍 1+⅓ C4（调外）；拍 2+⅓ C#4（置信 0.49）；拍 2& A#3（置信 0.37） · 试听：snippets/bar-033_vocal.wav ↔ snippets/bar-033_synth.wav · 音高图：plots/bar-033.png
- 第 34 小节：拍 3 C4（调外） · 试听：snippets/bar-034_vocal.wav ↔ snippets/bar-034_synth.wav · 音高图：plots/bar-034.png
- 第 35 小节：拍 3+⅓ C#4（置信 0.48） · 试听：snippets/bar-035_vocal.wav ↔ snippets/bar-035_synth.wav · 音高图：plots/bar-035.png
- 第 36 小节：拍 4& C4（调外） · 试听：snippets/bar-036_vocal.wav ↔ snippets/bar-036_synth.wav · 音高图：plots/bar-036.png
- 第 37 小节：拍 2& G#4（置信 0.45） · 试听：snippets/bar-037_vocal.wav ↔ snippets/bar-037_synth.wav · 音高图：plots/bar-037.png
- 第 40 小节：拍 2+⅓ G4（调外） · 试听：snippets/bar-040_vocal.wav ↔ snippets/bar-040_synth.wav · 音高图：plots/bar-040.png
- 第 42 小节：拍 3& B4（置信 0.48） · 试听：snippets/bar-042_vocal.wav ↔ snippets/bar-042_synth.wav · 音高图：plots/bar-042.png
- 第 43 小节：拍 3+⅓ C4（调外，置信 0.46） · 试听：snippets/bar-043_vocal.wav ↔ snippets/bar-043_synth.wav · 音高图：plots/bar-043.png
- 第 44 小节：拍 1& B3（置信 0.49）；拍 4 G#3（置信 0.46） · 试听：snippets/bar-044_vocal.wav ↔ snippets/bar-044_synth.wav · 音高图：plots/bar-044.png
- 第 45 小节：拍 2 D4（调外）；拍 4 G4（调外） · 试听：snippets/bar-045_vocal.wav ↔ snippets/bar-045_synth.wav · 音高图：plots/bar-045.png
- 第 50 小节：拍 1& A#4（置信 0.45） · 音高图：plots/bar-050.png
- 第 53 小节：拍 2 C#4（置信 0.43）；拍 2+⅓ G#4（置信 0.45）；拍 3+⅓ A#4（置信 0.41） · 音高图：plots/bar-053.png
- 第 55 小节：拍 2+⅓ E4（置信 0.45） · 音高图：plots/bar-055.png
- 第 57 小节：拍 2& G#3（置信 0.41） · 音高图：plots/bar-057.png
- 第 59 小节：拍 1& F#4（置信 0.49）；拍 2+⅓ G#4（置信 0.38） · 音高图：plots/bar-059.png
- 第 60 小节：拍 1& F#4（置信 0.44） · 音高图：plots/bar-060.png
- 第 61 小节：拍 4 B4（置信 0.48） · 音高图：plots/bar-061.png
- 第 62 小节：拍 2& C5（调外，置信 0.46）；拍 3+⅓ F#5（置信 0.43） · 音高图：plots/bar-062.png
