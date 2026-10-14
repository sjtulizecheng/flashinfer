"""Export measured FMHA-v2 pipeline diagrams and selected-CTA trace summaries."""

import csv
import gzip
import os
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parent
font_path = os.environ.get("FMHA_PIPELINE_FONT")
if font_path:
    font_manager.fontManager.addfont(font_path)
    font_family = font_manager.FontProperties(fname=font_path).get_name()
else:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    font_family = next(
        (name for name in ("Noto Sans CJK SC", "Noto Sans SC") if name in installed),
        "DejaVu Sans",
    )
plt.rcParams.update(
    {
        "font.family": font_family,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "path",
        "axes.unicode_minus": False,
    }
)
VERSIONS = ["baseline", "tma_padded", "dev3"]
NAMES = ["初版 FMHA-v2", "TMA 中间版", "dev-3 优化版"]
VCOLORS = ["#929cad", "#e9b05a", "#249e89"]
COLORS = {
    "load": "#348ac1",
    "wait": "#b9c1cc",
    "qk": "#8362ae",
    "scale": "#e5af51",
    "softmax": "#299c71",
    "pack": "#92cbb1",
    "pv": "#cd6f61",
    "write": "#5b7c91",
}
LABELS = {
    "load": "load / 预取发起",
    "wait": "等待 / 同步 / 控制",
    "qk": "QK：SMEM load + MMA",
    "scale": "scale / mask",
    "softmax": "online softmax / O 重缩放",
    "pack": "P 转 FP8",
    "pv": "PV：SMEM load + MMA",
    "write": "最终归一化 / 写回",
}
latency = json.loads((ROOT / "latency.json").read_text())
bycase = {r["case"]: r for r in latency}


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(ROOT / f"{name}.{ext}", dpi=170, facecolor="white")
    if name == "cta-full-timeline":
        path = ROOT / f"{name}.svg"
        path.with_suffix(".svg.gz").write_bytes(
            gzip.compress(path.read_bytes(), mtime=0)
        )
        path.unlink()
    plt.close(fig)


def box(ax, x, y, w, h, text, color, fs=11):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.4,rounding_size=1.3",
            facecolor=color,
            edgecolor="none",
            alpha=0.16,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        linespacing=1.65,
    )


def arrow(ax, x1, x2, y):
    ax.annotate(
        "",
        (x2, y),
        (x1, y),
        arrowprops={"arrowstyle": "->", "color": "#667085", "lw": 1.5},
    )


def overview():
    fig = plt.figure(figsize=(16, 12))
    fig.text(0.045, 0.953, "FMHA-v2：优化前后的算子流水线", fontsize=23, weight="bold")
    fig.text(
        0.045,
        0.919,
        "SM120 · RTX PRO 5000 72GB · GLM 第 23 层真实激活 · FP8 attention → BF16 输出",
        color="#546174",
    )
    ax = fig.add_axes([0.025, 0.525, 0.95, 0.36])
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 100)
    ax.axis("off")
    m = bycase["dense4096"]["ms"]
    prep = [
        "逐张量 abs / amax 求 scale\n拼接 K、量化 Q/K/V\n写 BSHD 布局",
        "逐张量 abs / amax 求 scale\n拼接 K、量化 Q/K/V\n写 BSHD 布局",
        "两个融合准备 kernel\n每 head / 64 token 求 scale\n拼接 + 量化，写 head-major",
    ]
    kernel = [
        "cp.async 分片装载\nQK → softmax → PV\n共享内存中转 O",
        "完整 tile + TMA 预取\nQ 驻留寄存器，O 直接写回\n长 causal tile 优先调度",
        "沿用 TMA attention\n读取分块 scale\n独立 Q / KV 长度",
    ]
    output = [
        "前缀 / ragged：取回有效 Q\n无前缀：仅返回视图",
        "前缀 / ragged：取回有效 Q\n无前缀：仅返回视图",
        "直接得到有效 Q 的 O\n取消 Q 补齐与输出 unpack",
    ]
    commits = ["481a4e4a", "28fa8310", "1c73959e"]
    for i, (version, y) in enumerate(zip(VERSIONS, [68, 35, 2], strict=True)):
        ax.text(1, y + 19, NAMES[i], fontsize=13, weight="bold")
        ax.text(1, y + 10, commits[i], fontsize=10, color="#64748b")
        preptime = m["prepare_new" if i == 2 else "prepare_old"]
        box(ax, 21, y, 31, 29, prep[i] + f"\n单测 {preptime:.3f} ms", COLORS["load"])
        box(
            ax,
            56,
            y,
            31,
            29,
            kernel[i] + f"\n单测 {m[version + '_attention']:.3f} ms",
            COLORS["qk"],
        )
        box(
            ax,
            91,
            y,
            28,
            29,
            output[i] + f"\n整条流水线 {m[version + '_pipeline']:.3f} ms",
            VCOLORS[i],
            10.6,
        )
        arrow(ax, 52.5, 55.5, y + 14)
        arrow(ax, 87.5, 90.5, y + 14)
    fig.text(
        0.05,
        0.492,
        "上图耗时：Q = KV = 4096。组件为独立测量，缓存状态不同，不能直接相加作为整条耗时。",
        fontsize=10,
        color="#5c6574",
    )
    ax = fig.add_axes([0.195, 0.126, 0.745, 0.292])
    labels = [
        "Q=KV=1024",
        "Q=KV=4096",
        "Q=KV=5093",
        "Q=3072 / KV=4096",
        "Q=[1024,512]\nKV=[1536,768]",
    ]
    for i, version in enumerate(VERSIONS):
        ys = [j + (i - 1) * 0.23 for j in range(len(latency))]
        vals = [r["ms"][version + "_pipeline"] for r in latency]
        ax.barh(ys, vals, height=0.205, color=VCOLORS[i], label=NAMES[i])
        for y, v in zip(ys, vals, strict=True):
            ax.text(v + 0.015, y, f"{v:.3f}", va="center", fontsize=9)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.85)
    ax.set_xlabel("整条算子流水线耗时（ms，越短越好）")
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01))
    fig.text(
        0.05,
        0.055,
        "4096 token：整条耗时 1.107 → 0.620 ms（−44.0%）；attention 0.782 → 0.514 ms（−34.3%）。",
        fontsize=12,
        weight="bold",
    )
    fig.text(
        0.05,
        0.027,
        "无打点版本 · CUDA Graph 重放 · 每项 5 轮取中位数 · 不包含投影层、MoE、调度与网络；不等于服务 QPS 收益。",
        fontsize=10,
        color="#5c6574",
    )
    save(fig, "pipeline-overview")


def parse_trace(version, case="dense4096"):
    path = ROOT / f"trace-{version}-{case}.json"
    data = json.loads(
        path.read_text()
        if path.exists()
        else gzip.decompress(path.with_suffix(".json.gz").read_bytes())
    )

    def span(records):
        a = [r["cycles"] for r in records if r["step"] == 0 and r["event"] == 0]
        b = [r["cycles"] for r in records if r["step"] == 127 and r["event"] == 4]
        return max(b) - min(a)

    spans = [span(r) for r in data["traces"]]
    idx = sorted(range(len(spans)), key=lambda i: spans[i])[len(spans) // 2]
    records = data["traces"][idx]
    d = {(r["warp"], r["step"], r["event"]): r["cycles"] for r in records}

    # Partition a warp's observed intervals; these include observer overhead.
    def partition(step, warp=0):
        e = {
            event: cycles
            for (w, s, event), cycles in d.items()
            if w == warp and s == step
        }
        pairs = []
        if step == 0:
            pairs = [(0, 1, "load"), (1, 2, "wait")]
        elif step == 127:
            pairs = [(0, 1, "write"), (1, 2, "wait"), (2, 3, "write"), (3, 4, "wait")]
        else:
            for j in range(3):
                a = 2 + 5 * j
                if a in e:
                    pairs += [
                        (a, a + 1, "load"),
                        (a + 1, a + 2, "wait"),
                        (a + 2, a + 3, "qk"),
                        (a + 3, a + 4, "wait"),
                    ]
            pairs += [
                (20, 21, "load"),
                (21, 22, "wait"),
                (22, 23, "qk"),
                (23, 25, "wait"),
                (25, 26, "scale"),
                (26, 27, "softmax"),
                (27, 28, "pack"),
            ]
            if 30 in e:
                pairs += [
                    (30, 31, "load"),
                    (31, 32, "wait"),
                    (32, 33, "pv"),
                    (33, 34, "wait"),
                ]
            pairs += [(40, 42, "wait"), (42, 43, "pv"), (43, 44, "wait")]
        result = []
        cursor = e[0]
        for a, b, kind in pairs:
            if a not in e or b not in e:
                continue
            assert e[a] >= cursor and e[b] >= e[a], (version, step, a, b)
            if e[a] > cursor:
                result.append((cursor, e[a], "wait"))
            result.append((e[a], e[b], kind))
            cursor = e[b]
        return result

    summary = {
        "variant": version,
        "case": case,
        "sample": idx,
        "cta_cycles_samples": spans,
        "cta_cycles_median": statistics.median(spans),
        "observer_ms": data["observer_ms"],
        "observer_overhead_pct": 100
        * (data["observer_ms"]["trace_enabled"] / data["observer_ms"]["clean"] - 1),
        "stages_cycles_warp0": {},
    }
    totals = defaultdict(int)
    for step in [0, *range(1, 65), 127]:
        for a, b, kind in partition(step):
            totals[kind] += b - a
    summary["stages_cycles_warp0"] = dict(totals)
    return d, partition, summary


def cta_figure():
    traces = {v: parse_trace(v) for v in VERSIONS}
    fig = plt.figure(figsize=(17, 15))
    fig.text(
        0.045,
        0.958,
        "一个 CTA 内部：搬运、两次 MMA 与 online softmax",
        fontsize=23,
        weight="bold",
    )
    fig.text(
        0.045,
        0.928,
        "Q=KV=4096 · head 0 · 最后 64 个 Q token · CTA = 128 threads / 4 warps · 遍历 64 个 KV tile",
        color="#546174",
    )
    ax = fig.add_axes([0.035, 0.686, 0.94, 0.208])
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 58)
    ax.axis("off")
    nodes_old = [
        ("cp.async\nQ/K 分 4 片\nload + wait", "load"),
        ("QK MMA\n每片读 SMEM\nQ 重复装载", "qk"),
        ("scale / mask\nonline softmax\nP 转 FP8", "softmax"),
        ("PV MMA\nV 分 2 片\nload + wait", "pv"),
        ("O 归一化\nSMEM 中转\n写回 BF16", "write"),
    ]
    nodes_new = [
        ("TMA Q/K\nQ 初始装载\n寄存器复用", "load"),
        ("QK MMA\n完整 K tile\n提前发起 V", "qk"),
        ("scale / mask\nonline softmax\n预取下一块 K", "softmax"),
        ("PV MMA\nV 逐片读寄存器\n消费整块 V", "pv"),
        ("O 归一化\n寄存器直接\n写回 BF16", "write"),
    ]
    for y, label, nodes in [(32, "初版", nodes_old), (1, "dev-3", nodes_new)]:
        ax.text(0, y + 13, label, fontsize=13, weight="bold")
        for i, (txt, kind) in enumerate(nodes):
            x = 11 + i * 22
            box(ax, x, y, 19, 26, txt, COLORS[kind], 11)
            if i < 4:
                arrow(ax, x + 19.5, x + 21.5, y + 13)
    fig.text(
        0.05,
        0.654,
        "上方是阶段示意，非时间比例；初版也有 cp.async 预取。两版均为 FP8×FP8 MMA、FP32 累加与 softmax。",
        fontsize=10,
        color="#5c6574",
    )
    fig.text(
        0.05,
        0.623,
        "clock64 实测放大：第 33 个 KV tile，取该 CTA 的 warp 0",
        fontsize=14,
        weight="bold",
    )
    maxwidth = 0
    for v in VERSIONS:
        d, _, _ = traces[v]
        maxwidth = max(maxwidth, (d[0, 33, 44] - d[0, 33, 0]) / 1000)
    # Every row uses the same cycle scale. Hatched rows bound asynchronous
    # issue-to-observed-wait windows, not hardware DMA completion times.
    for i, v in enumerate(VERSIONS):
        d, partition, summary = traces[v]
        origin = d[0, 33, 0]
        ax = fig.add_axes([0.155, 0.48 - i * 0.105, 0.79, 0.075])
        for a, b, kind in partition(33):
            a, b = (a - origin) / 1000, (b - origin) / 1000
            ax.barh(
                0.95,
                b - a,
                left=a,
                height=0.43,
                color=COLORS[kind],
                edgecolor="white",
                linewidth=0.35,
            )
            if b - a > 0.42 and kind in ("qk", "softmax", "pv"):
                ax.text(
                    (a + b) / 2,
                    0.95,
                    {"qk": "QK", "softmax": "softmax", "pv": "PV"}[kind],
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                )

        def window(a, b, label, color, y):
            a, b = (a - origin) / 1000, (b - origin) / 1000
            ax.barh(
                y,
                b - a,
                left=a,
                height=0.19,
                facecolor="none",
                edgecolor=color,
                hatch="////",
                linewidth=1,
            )
            ax.text(
                (a + b) / 2,
                y,
                label,
                fontsize=8,
                ha="center",
                va="center",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.85,
                    "edgecolor": "none",
                    "pad": 0.2,
                },
            )

        if v == "baseline":
            window(
                d[0, 33, 21], d[0, 33, 32], "V 首片：issue → wait", COLORS["load"], 0.50
            )
            window(
                d[0, 33, 31], d[0, 33, 42], "V 尾片：issue → wait", COLORS["load"], 0.20
            )
        else:
            window(
                d[0, 33, 21], d[0, 33, 42], "整块 V：issue → wait", COLORS["load"], 0.50
            )
            # Next-K stamp follows K release + issue; next iteration's event 22
            # follows its K wait. This is an envelope, not a pure transfer timer.
            window(
                d[0, 33, 25],
                d[0, 34, 22],
                "下一块 K：issue → 下一轮 wait",
                COLORS["qk"],
                0.20,
            )
        ax.set_xlim(0, maxwidth + 1.1)
        ax.set_ylim(-0.1, 1.3)
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
        ax.text(
            -0.16,
            0.7,
            NAMES[i],
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=11,
        )
        ax.set_xlabel(
            "距本轮开始的 SM clock cycles（×1000）" if i == 2 else "", fontsize=10
        )
        if i < 2:
            ax.set_xticklabels([])
    handles = [
        Patch(facecolor=COLORS[k], label=LABELS[k]) for k in COLORS if k != "write"
    ]
    fig.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.51, 0.211),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.05,
        0.169,
        "虚线纹理表示发起至观察到 wait 返回的窗口，不能当作精确 DMA 时长；QK / PV 段包含 SMEM 读、发射与等待。",
        fontsize=10,
        color="#5c6574",
    )
    boxax = fig.add_axes([0.04, 0.063, 0.93, 0.089])
    boxax.set_xlim(0, 100)
    boxax.set_ylim(0, 20)
    boxax.axis("off")
    box(
        boxax,
        0,
        0,
        31,
        20,
        "共享内存 / CTA\n64 KiB → 48 KiB + 16 B\n允许驻留：1 → 2 CTA / SM",
        COLORS["load"],
        11,
    )
    box(
        boxax,
        34,
        0,
        31,
        20,
        "无打点 NCU 实测\n占用率 8.33% → 16.56%\nlocal load / store 均为 0",
        COLORS["softmax"],
        11,
    )
    box(
        boxax,
        68,
        0,
        31,
        20,
        "观测限制\n打点使整核耗时增加 3.5%～11.0%\n阶段占比不能直接作为优化收益",
        COLORS["scale"],
        10.5,
    )
    fig.text(
        0.05,
        0.029,
        "MMA 指令类型不是这次主要变化；重点是减少切片与 Q 重载、TMA 预取、更小的 SMEM 占用，以及取消流水线外的重复准备。",
        fontsize=10,
        color="#5c6574",
    )
    save(fig, "cta-pipeline")
    results = []
    for case in ("dense4096", "prefix"):
        for v in VERSIONS:
            results.append(parse_trace(v, case)[2])
    (ROOT / "trace-summary.json").write_text(json.dumps(results, indent=2))


def full_cta():
    fig = plt.figure(figsize=(16, 10))
    fig.text(0.05, 0.95, "一个 CTA 的完整生命周期", fontsize=23, weight="bold")
    fig.text(
        0.05,
        0.91,
        "clock64 打点 · Q=KV=4096 · 最后一个逻辑 Q tile / head 0 · 4 个 warp 全部记录",
        color="#546174",
    )
    for i, v in enumerate(VERSIONS):
        d, partition, s = parse_trace(v)
        origin = min(d[w, 0, 0] for w in range(4))
        ax = fig.add_axes([0.13, 0.70 - i * 0.18, 0.80, 0.125])
        for w in range(4):
            segments = []
            for step in [0, *range(1, 65), 127]:
                segments.extend(partition(step, w))
            for a, b, kind in segments:
                ax.barh(
                    w,
                    (b - a) / 1000,
                    left=(a - origin) / 1000,
                    height=0.7,
                    color=COLORS[kind],
                    linewidth=0,
                )
        ax.set_xlim(0, 415)
        ax.set_yticks(range(4), [f"warp {w}" for w in range(4)])
        ax.invert_yaxis()
        ax.set_title(
            f"{NAMES[i]}：选中样本 CTA 跨度 {s['cta_cycles_median'] / 1000:.1f} kcycles",
            fontsize=12,
            loc="left",
        )
        ax.grid(axis="x", alpha=0.13)
        ax.set_axisbelow(True)
        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("距该 CTA 进入被测区域的 SM clock cycles（×1000）")
    handles = [Patch(facecolor=COLORS[k], label=LABELS[k]) for k in COLORS]
    fig.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.52, 0.25),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.05,
        0.18,
        "顺序：初始 Q/K load → 64 次 [QK → online softmax → PV] → 最终归一化与 O 写回。",
        fontsize=12,
    )
    fig.text(
        0.05,
        0.135,
        "选择 7 个样本中 CTA 总跨度居中的一次；各 warp 使用同一 CTA 的相对时钟。图中窄间隔含未分类控制指令。",
        fontsize=10,
        color="#5c6574",
    )
    fig.text(
        0.05,
        0.095,
        "这是一个 CTA 的观测，不是整核耗时：SM 同时运行多少个 CTA、causal 调度的长尾，也会影响整核性能。",
        fontsize=11,
    )
    fig.text(
        0.05,
        0.055,
        "打点改变指令序列并增加开销；不能将这里的周期差直接换算成生产版本的阶段加速比。",
        fontsize=11,
        color="#956122",
    )
    save(fig, "cta-full-timeline")


def ncu_summary():
    results = {}
    for p in sorted(ROOT.glob("ncu-*.csv")):
        lines = [line for line in p.read_text().splitlines() if line.startswith('"')]
        rows = [r for r in csv.DictReader(lines) if r["ID"]]
        assert len(rows) == 1, p
        row = rows[0]
        results[p.stem.removeprefix("ncu-")] = {
            k: row[k]
            for k in row
            if k.startswith(("sm__", "smsp__", "l1tex__"))
            or k
            in (
                "Kernel Name",
                "Block Size",
                "Grid Size",
                "gpu__time_duration.sum",
                "launch__registers_per_thread",
                "launch__shared_mem_per_block_dynamic",
                "launch__occupancy_limit_registers",
                "launch__occupancy_limit_shared_mem",
            )
        }
    if results:
        (ROOT / "ncu-summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    overview()
    cta_figure()
    full_cta()
    ncu_summary()
    print("Saved pipeline-overview / cta-pipeline (PNG, SVG) and measurement summaries")
