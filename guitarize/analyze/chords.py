"""参考和弦：音频模板匹配 + HMM、网上和弦谱对齐，逐拍投票得出标签与置信度（chords.json）。"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .. import theory
from .webchart import POWER_RE, WebBar

LABELS = list(theory.PITCH_CLASSES) + [n + "m" for n in theory.PITCH_CLASSES]  # 与 theory.MAJMIN_LABELS 同序
_TEMPLATES = theory.TEMPLATES / np.linalg.norm(theory.TEMPLATES, axis=1, keepdims=True)
MIN_ALIGN_SIM = 0.5
# 对齐时全 N 小节（N.C. / 只有贝斯）按此相似度计。取值来自经验窗口，不是理论值：
#   · 真实曲目（なんもねえ）的临界值是 0.594：低于它，只有贝斯的尾奏小节会被接到无关的和弦小节上；
#   · 0.60–0.76 之间结果完全相同；
#   · ≥ 0.77 会抢走一个真实的 F# 边界小节（相似度 0.761）；合成场景里 TV 版跳过 N.C. 段后相似度约 0.7 的主歌小节也会被吞进 N 段；
#   · 合成噪声测试在 0.65 稳健。
# 取窗口中部 0.65；上沿由 test_chords.py::test_modest_verse_bars_are_not_absorbed_by_an_n_run 钉住。
N_ALIGN_SIM = 0.65
CONFIDENT_POSTERIOR = 0.7


def template_track(cb: np.ndarray, stay: float = 0.5, temperature: float = 12.0) -> tuple[list[str], np.ndarray]:
    """cb: (12, n) 单位化逐拍色度 → 逐拍标签（Viterbi 路径）及该标签的边缘后验概率（前向-后向）。"""
    log_e = temperature * (_TEMPLATES @ cb)
    log_e = log_e - logsumexp(log_e, axis=0, keepdims=True)
    k, n = log_e.shape
    log_a = np.full((k, k), np.log((1 - stay) / (k - 1)))
    np.fill_diagonal(log_a, np.log(stay))

    delta = log_e[:, 0] - np.log(k)
    back = np.zeros((k, n), dtype=int)
    for i in range(1, n):
        m = delta[:, None] + log_a
        back[:, i] = m.argmax(axis=0)
        delta = m.max(axis=0) + log_e[:, i]
    path = [int(delta.argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[path[-1], i]))
    path = path[::-1]

    fwd = np.zeros((k, n))
    bwd = np.zeros((k, n))
    fwd[:, 0] = log_e[:, 0] - np.log(k)
    for i in range(1, n):
        fwd[:, i] = logsumexp(fwd[:, i - 1][:, None] + log_a, axis=0) + log_e[:, i]
    for i in range(n - 2, -1, -1):
        bwd[:, i] = logsumexp(log_a + (log_e[:, i + 1] + bwd[:, i + 1])[None, :], axis=1)
    post = fwd + bwd
    post = np.exp(post - logsumexp(post, axis=0, keepdims=True))
    return [LABELS[s] for s in path], np.array([post[s, i] for i, s in enumerate(path)])


def chord_tones_any(label: str) -> set[int]:
    m = POWER_RE.match(label)
    if m:
        r = theory.pc(m.group(1))
        return {r, (r + 7) % 12}
    return theory.chord_tones(label)


def web_bar_vector(bar: WebBar) -> np.ndarray:
    v = np.zeros(12)
    for label, w in bar.chords:
        if label != "N":
            for p in chord_tones_any(label):
                v[p] += w
    n = np.linalg.norm(v)
    return v / n if n else v


def _dp_align(sim: np.ndarray, jump: float, any_jump: float) -> tuple[float, list[int]]:
    """每个音频小节 i 对应一个网上小节 j。转移：j−1 → j 不罚；向前跳过若干小节罚 jump；任意跳（含回跳）罚 any_jump。"""
    n, m = sim.shape
    neg = -1e18
    dp = np.full((n, m), neg)
    back = np.zeros((n, m), dtype=int)
    dp[0] = sim[0]
    for i in range(1, n):
        prev = dp[i - 1]
        run_arg = np.zeros(m, dtype=int)
        best = 0
        for j in range(m):
            if prev[j] > prev[best]:
                best = j
            run_arg[j] = best
        glob = int(prev.argmax())
        for j in range(m):
            cands = [(prev[glob] - any_jump, glob)]
            if j >= 1:
                cands.append((prev[j - 1], j - 1))
            if j >= 2:
                cands.append((prev[run_arg[j - 2]] - jump, int(run_arg[j - 2])))
            score, arg = max(cands)
            dp[i, j] = score + sim[i, j]
            back[i, j] = arg
    path = [int(dp[-1].argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    return float(dp[-1].max()), path[::-1]


def align_webchart(audio_bars: np.ndarray, web: list[WebBar], jump: float = 1.0, any_jump: float = 2.0) -> dict:
    """audio_bars: (N, 12) 单位化小节色度。网上谱整体移调 transpose 个半音后与音频最匹配。

    全 N 小节（N.C. / 只有贝斯）的向量为零，相似度恒为 0，DP 永远不会经过它们；
    对齐时把这些列按 N_ALIGN_SIM 计，返回的 sim 仍是真实相似度（交给 reference_chords 的门槛判断）。"""
    w = np.stack([web_bar_vector(b) for b in web])
    all_n = np.array([all(c == "N" for c, _ in b.chords) for b in web])
    best = None
    for t in range(12):
        sim = audio_bars @ np.roll(w, t, axis=1).T
        score, path = _dp_align(np.where(all_n[None, :], N_ALIGN_SIM, sim), jump, any_jump)
        if best is None or score > best[0] + 1e-9:
            best = (score, t, path, sim)
    _, t, path, sim = best
    return {"index": path, "transpose": t, "sim": [float(sim[i, j]) for i, j in enumerate(path)]}


def web_label_at(bar: WebBar, frac: float, transpose: int) -> str:
    """小节内位置 frac（0..1）处的网上和弦，按 transpose 移调。"""
    acc = 0.0
    label = bar.chords[-1][0]
    for lab, w in bar.chords:
        acc += w
        if frac < acc - 1e-9:
            label = lab
            break
    return label if label == "N" or transpose == 0 else theory.transpose_label(label, transpose)


def _root(label: str) -> int | None:
    if label == "N":
        return None
    m = POWER_RE.match(label)
    if m:
        return theory.pc(m.group(1))
    parsed = theory.parse_chord(label)
    return None if parsed is None else parsed[0]


def power_quality_table(web: list[WebBar]) -> dict[int, str]:
    """根音 pc → "" (大三) / "m" (小三)：取网上谱里同根非强力和弦的多数性质；平票不收录。"""
    counts: dict[tuple[int, bool], int] = {}
    for bar in web:
        for label, _ in bar.chords:
            if label == "N" or POWER_RE.match(label):
                continue
            root = _root(label)
            if root is None:
                continue
            is_min = theory.majmin(label).endswith(":min")
            counts[(root, is_min)] = counts.get((root, is_min), 0) + 1
    table = {}
    for (root, is_min), c in counts.items():
        other = counts.get((root, not is_min), 0)
        if c > other:
            table[root] = "m" if is_min else ""
    return table


def resolve_power(label: str, table: dict[int, str], key_pcs: set[int] | None, audio: str) -> str:
    """强力和弦 → 带大小三的标签：网上谱同根多数性质 → 调内三和弦性质 → 音频模板的性质（仅当音频和弦同根，否则大三）。"""
    root = _root(label)
    if root in table:
        quality = table[root]
    elif key_pcs is not None and root in key_pcs and (root + 4) % 12 in key_pcs and (root + 7) % 12 in key_pcs:
        quality = ""
    elif key_pcs is not None and root in key_pcs and (root + 3) % 12 in key_pcs and (root + 7) % 12 in key_pcs:
        quality = "m"
    else:
        # 调外强力和弦多半是借来的大三和弦（bVII/bVI/bIII）：只有音频和弦同根时才采信其性质，否则默认大三
        quality = "m" if _root(audio) == root and audio.endswith("m") else ""
    return theory.pc_name(root) + quality


def vote_beat(audio: str, posterior: float, web: str | None, power_table: dict[int, str] | None = None,
              key_pcs: set[int] | None = None) -> tuple[str, str, dict]:
    """返回 (标签, 置信度, 各方投票)。web=None 表示该拍没有可用的网上和弦（N=1）。"""
    votes = {"template": audio}
    if web is None:
        return audio, ("medium" if posterior >= CONFIDENT_POSTERIOR else "low"), votes
    votes["web"] = web
    if web == "N":
        return "N", "medium", votes
    w = resolve_power(web, power_table or {}, key_pcs, audio) if POWER_RE.match(web) else web
    if _root(w) == _root(audio):
        # 根音一致但大/小三不同：失真的乐队混音里音频模板的大/小三判断不可靠，故采用人工谱的性质，
        # 该拍仍计分但降为 medium（根音都不一致才是 low）
        return w, ("high" if theory.majmin(w) == theory.majmin(audio) else "medium"), votes
    return w, "low", votes


def reference_chords(cb: np.ndarray, beats_json: dict, web: list[WebBar] | None = None,
                    key_pcs: set[int] | None = None) -> dict:
    """cb: 与 beats_json["beats"] 一一对应的逐拍色度 (12, n)。"""
    labels, post = template_track(cb)
    beats = beats_json["beats"]
    bpb = beats_json["time_signature"][0]
    numbers = sorted({b["bar"] for b in beats})
    web_at: list[str | None] = [None] * len(beats)
    alignment = None
    power_table: dict[int, str] = {}
    if web:
        power_table = power_quality_table(web)
        vecs = []
        for n in numbers:
            idx = [i for i, b in enumerate(beats) if b["bar"] == n]
            v = cb[:, idx].mean(axis=1)
            vecs.append(v / (np.linalg.norm(v) + 1e-9))
        alignment = align_webchart(np.stack(vecs), web)
        t = alignment["transpose"]  # 网上谱整体移调后，强力和弦性质表也要同步移调
        power_table = {(r + t) % 12: q for r, q in power_table.items()}
        for k, n in enumerate(numbers):
            wb = web[alignment["index"][k]]
            all_n = all(c == "N" for c, _ in wb.chords)
            consecutive = k > 0 and alignment["index"][k] == alignment["index"][k - 1] + 1
            if alignment["sim"][k] < MIN_ALIGN_SIM and not (all_n and consecutive):
                continue
            for i, b in enumerate(beats):
                if b["bar"] == n:
                    web_at[i] = web_label_at(wb, (b["beat"] - 1) / bpb, alignment["transpose"])
    segments: list[dict] = []
    for i, b in enumerate(beats):
        label, conf, votes = vote_beat(labels[i], float(post[i]), web_at[i], power_table, key_pcs)
        last = segments[-1] if segments else None
        if (last and last["bar"] == b["bar"] and last["label"] == label and last["confidence"] == conf
                and last["votes"] == votes and last["beat"] + last["dur_beats"] == b["beat"]):
            last["dur_beats"] += 1
        else:
            segments.append({"bar": b["bar"], "beat": b["beat"], "dur_beats": 1, "label": label,
                             "confidence": conf, "votes": votes})
    out: dict = {"segments": segments}
    if alignment is not None:
        out["web_alignment"] = [{"bar": n, "web_index": alignment["index"][k], "sim": round(alignment["sim"][k], 3)}
                                for k, n in enumerate(numbers)]
        out["web_transpose"] = alignment["transpose"]
    return out
