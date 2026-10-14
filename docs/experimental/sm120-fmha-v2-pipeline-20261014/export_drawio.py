"""Export the measured diagrams as editable draw.io shapes (no embedded images)."""

import base64
import gzip
import json
import math
import xml.etree.ElementTree as ET
import zlib
from urllib.parse import quote

from visualize import (
    COLORS,
    LABELS,
    NAMES,
    ROOT,
    VCOLORS,
    VERSIONS,
    bycase,
    latency,
    parse_trace,
    partition_timestamps,
)

INK = "#243247"
MUTED = "#56677c"
FONT = "Noto Sans CJK SC,Noto Sans SC,Microsoft YaHei,PingFang SC,Arial"


class Page:
    def __init__(self, name, width, height):
        self.name = name
        self.model = ET.Element(
            "mxGraphModel",
            grid="0",
            gridSize="10",
            guides="1",
            tooltips="1",
            connect="1",
            arrows="1",
            fold="1",
            page="1",
            pageScale="1",
            pageWidth=str(width),
            pageHeight=str(height),
            background="#ffffff",
            math="0",
            shadow="0",
        )
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", id="0")
        ET.SubElement(
            self.root, "mxCell", id="1", parent="0", value="标题、说明与坐标轴"
        )
        self.count = 1

    def uid(self):
        self.count += 1
        return f"c{self.count}"

    def layer(self, label):
        uid = self.uid()
        ET.SubElement(self.root, "mxCell", id=uid, parent="0", value=label)
        return uid

    def rect(
        self,
        x,
        y,
        w,
        h,
        label="",
        fill="none",
        stroke="none",
        size=18,
        parent="1",
        extra="",
        metadata=None,
    ):
        uid = self.uid()
        container = self.root
        attrs = {"id": uid, "value": label}
        if metadata:
            container = ET.SubElement(
                self.root,
                "object",
                id=uid,
                label=label,
                **{key: str(value) for key, value in metadata.items()},
            )
            attrs = {}
        style = (
            f"rounded=0;whiteSpace=wrap;html=0;fillColor={fill};strokeColor={stroke};"
            f"fontColor={INK};fontFamily={FONT};fontSize={size};align=center;"
            f"verticalAlign=middle;spacing=6;{extra}"
        )
        if metadata and "stage" in metadata and not label:
            style = f"fillColor={fill};strokeColor=none;"
        cell = ET.SubElement(
            container,
            "mxCell",
            **attrs,
            vertex="1",
            parent=parent,
            style=style,
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            x=str(round(x, 2)),
            y=str(round(y, 2)),
            width=str(round(w, 2)),
            height=str(round(h, 2)),
            **{"as": "geometry"},
        )
        return uid

    def text(self, x, y, w, h, label, size=18, bold=False, parent="1", color=INK):
        return self.rect(
            x,
            y,
            w,
            h,
            label,
            size=size,
            parent=parent,
            extra=f"align=left;spacing=0;fontColor={color};fontStyle={int(bold)};",
        )

    def box(self, x, y, w, h, label, color, size=18, parent="1"):
        rgb = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
        pale = "#" + "".join(
            f"{round(0.16 * channel + 0.84 * 255):02x}" for channel in rgb
        )
        return self.rect(
            x,
            y,
            w,
            h,
            label,
            pale,
            color,
            size,
            parent,
            extra="rounded=1;arcSize=10;strokeWidth=1;",
        )

    def line(self, x1, y1, x2, y2, color="#d9e0e8", source=None, target=None):
        attrs = {}
        if source:
            attrs.update(source=source, target=target)
        cell = ET.SubElement(
            self.root,
            "mxCell",
            id=self.uid(),
            value="",
            edge="1",
            parent="1",
            style=(
                f"strokeColor={color};strokeWidth={2 if source else 1};"
                f"endArrow={'block' if source else 'none'};endFill=1;"
                "exitX=1;exitY=0.5;entryX=0;entryY=0.5;rounded=0;"
            ),
            **attrs,
        )
        geo = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
        ET.SubElement(geo, "mxPoint", x=str(x1), y=str(y1), **{"as": "sourcePoint"})
        ET.SubElement(geo, "mxPoint", x=str(x2), y=str(y2), **{"as": "targetPoint"})

    def title(self, title, subtitle):
        self.text(36, 24, 1730, 44, title, 30, True)
        self.text(36, 77, 1730, 32, subtitle, 17, color=MUTED)

    def legend(self, x, y, keys, columns=4, width=430):
        for i, kind in enumerate(keys):
            dx, dy = x + (i % columns) * width, y + (i // columns) * 40
            self.rect(dx, dy + 5, 22, 22, fill=COLORS[kind])
            self.text(dx + 32, dy, width - 38, 32, LABELS[kind], 16)


def overview():
    page = Page("01 算子流水线与耗时", 1800, 1400)
    page.title(
        "FMHA-v2：优化前后的算子流水线",
        "SM120 · RTX PRO 5000 72GB · GLM 第 23 层真实激活 · FP8 attention → BF16 输出",
    )
    page.box(
        36,
        125,
        1728,
        66,
        "TMA 中间版是对照版本：已经优化 attention 内核，仍保留旧准备与 Q 补齐。"
        "用于区分内核整组优化和 dev-3 接口 / 准备优化的收益。",
        COLORS["scale"],
        18,
    )
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
    ms = bycase["dense4096"]["ms"]
    for i, version in enumerate(VERSIONS):
        layer = page.layer(NAMES[i])
        y = 220 + i * 160
        page.text(36, y + 30, 190, 32, NAMES[i], 21, True, layer)
        page.text(
            36,
            y + 74,
            190,
            30,
            ["481a4e4a", "28fa8310", "1c73959e"][i],
            17,
            parent=layer,
            color=MUTED,
        )
        preptime = ms["prepare_new" if i == 2 else "prepare_old"]
        a = page.box(
            235,
            y,
            455,
            130,
            prep[i] + f"\n准备单测：{preptime:.6f} ms",
            COLORS["load"],
            parent=layer,
        )
        b = page.box(
            745,
            y,
            455,
            130,
            kernel[i] + f"\nattention 单测：{ms[version + '_attention']:.6f} ms",
            COLORS["qk"],
            parent=layer,
        )
        c = page.box(
            1255,
            y,
            505,
            130,
            output[i] + f"\n整条流水线：{ms[version + '_pipeline']:.6f} ms",
            VCOLORS[i],
            parent=layer,
        )
        page.line(690, y + 65, 745, y + 65, MUTED, a, b)
        page.line(1200, y + 65, 1255, y + 65, MUTED, b, c)
    page.text(
        36,
        690,
        1728,
        34,
        "以上为 Q = KV = 4096。组件独立测量，缓存与执行上下文不同，不能直接相加作为整条耗时。",
        17,
        color=MUTED,
    )
    page.text(36, 742, 1728, 36, "五种输入形状：无打点整条算子流水线耗时", 23, True)
    for i in range(3):
        page.rect(440 + i * 375, 791, 24, 20, fill=VCOLORS[i])
        page.text(475 + i * 375, 784, 320, 32, NAMES[i], 17)
    x0, plot_width, xmax = 410, 1270, 1.85
    for tick in range(10):
        value = tick * 0.2
        x = x0 + value / xmax * plot_width
        page.line(x, 829, x, 1220)
        page.text(x - 14, 1226, 70, 28, f"{value:.1f}", 15)
    for j, row in enumerate(latency):
        y = 838 + j * 77
        q, kv = row["qlens"], row["klens"]
        label = f"Q = KV = {q[0]}" if q == kv and len(q) == 1 else f"Q = {q}\nKV = {kv}"
        page.text(36, y, 350, 62, label, 18)
        for i, version in enumerate(VERSIONS):
            value = row["ms"][version + "_pipeline"]
            width = value / xmax * plot_width
            page.rect(
                x0,
                y + 21 * i,
                width,
                17,
                fill=VCOLORS[i],
                metadata={
                    "case": row["case"],
                    "variant": version,
                    "milliseconds": value,
                    "tooltip": f"{NAMES[i]} | {row['case']} | {value:.9f} ms",
                },
            )
            page.text(x0 + width + 10, y + 21 * i - 5, 90, 27, f"{value:.3f}", 15)
    page.text(410, 1262, 1270, 28, "整条算子流水线耗时（ms，越短越好）", 17)
    page.text(
        36,
        1310,
        1728,
        32,
        "4096 token：初版 → dev-3 整条耗时下降 43.98%；attention 耗时下降 34.29%。",
        20,
        True,
    )
    page.text(
        36,
        1351,
        1728,
        28,
        "CUDA Graph 重放，每项 5 轮取中位数。不包含投影层、MoE、调度与网络；不等于服务 QPS 收益。",
        16,
        color=MUTED,
    )
    return page


def trace_bar(
    page,
    intervals,
    x,
    y,
    scale,
    origin,
    height,
    version,
    warp,
    step,
    parent,
    labels=False,
):
    for start, end, kind in intervals:
        if end <= start:
            continue
        width = (end - start) * scale
        label = (
            {"qk": "QK", "softmax": "softmax", "pv": "PV"}.get(kind, "")
            if labels and width > 85
            else ""
        )
        page.rect(
            x + (start - origin) * scale,
            y,
            width,
            height,
            label,
            COLORS[kind],
            size=15,
            parent=parent,
            extra="fontColor=#ffffff;spacing=0;whiteSpace=nowrap;",
            metadata={
                "variant": version,
                "warp": warp,
                "step": step,
                "stage": kind,
                "start_cycle": start - origin,
                "end_cycle": end - origin,
                "duration_cycles": end - start,
                "placeholders": "1",
                "tooltip": "%variant% | warp %warp% | step %step% | %stage%\n%start_cycle% → %end_cycle% cycles；持续 %duration_cycles% cycles（含打点开销）",
            },
        )


def joint_timeline(page):
    """Place prepare and selected-CTA stamps on the same measured GPU clock."""
    samples = {}
    for version in VERSIONS:
        data = json.loads(
            gzip.decompress((ROOT / f"joint-timeline-{version}.json.gz").read_bytes())
        )
        chosen = data["samples"][data["sample"]]
        d = {(r["warp"], r["step"], r["event"]): r["ns"] for r in chosen["records"]}
        samples[version] = (
            chosen["boundaries_ns"],
            d,
            partition_timestamps(d, version),
        )
    page.text(
        36, 697, 2088, 38, "实测时间轴：prepare 已接入同一条 GPU 流水线", 27, True
    )
    page.text(
        36,
        743,
        2088,
        40,
        "统一读取 GPU globaltimer，单位 µs；黄色是一整个 prepare 区间，后接 Attention 内所选 CTA 的真实阶段。",
        20,
        color=MUTED,
    )
    x0, width = 235, 1720
    xmax = (
        math.ceil(max((b[2] - b[0]) / 1000 for b, _, _ in samples.values()) / 200) * 200
    )
    scale = width / (xmax * 1000)
    for i, version in enumerate(VERSIONS):
        (t0, t1, t2), d, partition = samples[version]
        y = 820 + i * 175
        layer = page.layer(NAMES[i] + "：prepare + CTA，GPU globaltimer")
        page.text(36, y + 20, 185, 42, NAMES[i], 20, True, layer)
        for us in range(0, int(xmax) + 1, 100):
            x = x0 + us * 1000 * scale
            page.line(x, y - 8, x, y + 114)
            if i == 2:
                page.text(x - 15, y + 121, 80, 26, str(us), 16)
        end_x = x0 + (t2 - t0) * scale
        prep_x = x0 + (t1 - t0) * scale
        page.rect(
            prep_x,
            y,
            (t2 - t1) * scale,
            48,
            fill="#edf1f5",
            stroke="#c4ced9",
            parent=layer,
            metadata={
                "variant": version,
                "start_ns": t1 - t0,
                "end_ns": t2 - t0,
                "tooltip": "Attention 全部 CTA 所在区间；灰色表示未展开的工作，不代表 GPU 空闲。",
            },
        )
        page.rect(
            x0,
            y,
            (t1 - t0) * scale,
            48,
            f"prepare  {(t1 - t0) / 1000:.1f} µs",
            "#f2c66d",
            "#bd8824",
            21,
            layer,
            metadata={
                "variant": version,
                "stage": "prepare",
                "start_ns": 0,
                "end_ns": t1 - t0,
                "duration_ns": t1 - t0,
                "tooltip": f"完整 prepare 区间：{(t1 - t0) / 1000:.3f} µs，包含边界打点/调度开销。",
            },
        )
        for step in [0, *range(1, 65), 127]:
            for start, end, kind in partition(step):
                if end > start:
                    page.rect(
                        x0 + (start - t0) * scale,
                        y,
                        (end - start) * scale,
                        48,
                        fill=COLORS[kind],
                        parent=layer,
                        metadata={
                            "variant": version,
                            "warp": 0,
                            "step": step,
                            "stage": kind,
                            "start_ns": start - t0,
                            "end_ns": end - t0,
                            "duration_ns": end - start,
                            "tooltip": f"{version} | warp 0 | KV step {step} | {kind}\n"
                            f"{(start - t0) / 1000:.3f} → {(end - t0) / 1000:.3f} µs",
                        },
                    )
        page.text(
            end_x + 12, y + 7, 165, 37, f"{(t2 - t0) / 1000:.1f} µs", 21, True, layer
        )
        cta_start, cta_end = d[0, 0, 0], d[0, 127, 4]
        page.rect(
            x0 + (cta_start - t0) * scale,
            y + 55,
            (cta_end - cta_start) * scale,
            32,
            "所选 CTA · 64 轮 KV",
            "#ffffff",
            VCOLORS[i],
            16,
            layer,
            extra="dashed=1;strokeWidth=2;",
        )
        page.text(
            prep_x,
            y + 90,
            end_x - prep_x,
            27,
            f"Attention 整体区间：{(t2 - t1) / 1000:.1f} µs（其它 CTA 同时运行）",
            18,
            parent=layer,
            color=MUTED,
        )
    page.text(
        x0,
        1330,
        1720,
        30,
        "从 prepare 起点开始的实际 GPU 时间（µs）；三行使用相同比例，包含打点开销。",
        20,
    )
    page.text(
        36,
        1380,
        2088,
        42,
        "下方放大同一次运行的第 33 个 KV tile。prepare 不在每一轮重复；它已经完整画在上方时间轴的最左侧。",
        22,
        True,
    )
    zoom_max = max(
        (max(d[0, 33, 44], d[0, 34, 22]) if v != "baseline" else d[0, 33, 44])
        - d[0, 33, 0]
        for v, (_, d, _) in samples.items()
    )
    zoom_max = math.ceil(zoom_max / 500) * 500
    zoom_scale = 1720 / zoom_max
    for i, version in enumerate(VERSIONS):
        (t0, _, _), d, partition = samples[version]
        origin = d[0, 33, 0]
        y = 1490 + i * 155
        layer = page.layer(NAMES[i] + "：联合采样的第 33 轮放大")
        page.text(36, y - 4, 185, 38, NAMES[i], 19, True, layer)
        page.text(
            36,
            y + 37,
            185,
            43,
            f"全程 {(origin - t0) / 1000:.3f} µs\n处的这一轮",
            16,
            parent=layer,
            color=MUTED,
        )
        for ns in range(0, int(zoom_max) + 1, 500):
            x = x0 + ns * zoom_scale
            page.line(x, y - 9, x, y + 115)
            if i == 2:
                page.text(x - 12, y + 122, 72, 26, f"{ns / 1000:g}", 16)
        for start, end, kind in partition(33):
            if end > start:
                w = (end - start) * zoom_scale
                label = (
                    {"qk": "QK", "softmax": "softmax", "pv": "PV"}.get(kind, "")
                    if w > 80
                    else ""
                )
                page.rect(
                    x0 + (start - origin) * zoom_scale,
                    y,
                    w,
                    35,
                    label,
                    COLORS[kind],
                    size=17,
                    parent=layer,
                    extra="fontColor=#ffffff;spacing=0;",
                    metadata={
                        "variant": version,
                        "stage": kind,
                        "step": 33,
                        "duration_ns": end - start,
                        "tooltip": f"{kind}: {(end - start) / 1000:.3f} µs",
                    },
                )
        windows = (
            [
                (21, 33, 32, "V 首片：issue → wait", "load"),
                (31, 33, 42, "V 尾片：issue → wait", "load"),
            ]
            if version == "baseline"
            else [
                (21, 33, 42, "整块 V：issue → wait", "load"),
                (25, 34, 22, "下一块 K：issue → 下一轮 wait", "qk"),
            ]
        )
        for j, (a, step, b, label, kind) in enumerate(windows):
            start, end = d[0, 33, a], d[0, step, b]
            page.rect(
                x0 + (start - origin) * zoom_scale,
                y + 45 + j * 36,
                (end - start) * zoom_scale,
                27,
                label,
                "#ffffff",
                COLORS[kind],
                16,
                layer,
                extra="dashed=1;strokeWidth=2;",
            )
    page.text(
        x0,
        1972,
        1800,
        30,
        "距这一轮开始的 GPU 时间（µs）；放大图与上方全程来自同一份 globaltimer 记录。",
        19,
    )
    page.legend(36, 2030, list(COLORS), columns=4, width=520)
    page.text(
        36,
        2130,
        2088,
        36,
        "黄色 prepare：整个准备阶段；彩色细条：一个 CTA 的 warp 0；灰色背景：Attention 内未展示的工作。",
        20,
    )
    page.text(
        36,
        2175,
        2088,
        40,
        "globaltimer 可跨 kernel / SM 对齐；原 clock64 仅用于 CTA 内周期。不能按固定 GPU 频率把旧单测毫秒数拼到 clock64 上。",
        19,
        color=MUTED,
    )
    page.text(
        36,
        2220,
        2088,
        42,
        "这是带打点的观测样本；性能收益仍以上方无打点实测为准。异步框是 issue→wait 窗口，不是 DMA 精确时长。",
        19,
        color=MUTED,
    )


def cta_pipeline(traces, include_preparation=False):
    page = Page(
        "04 准备与 CTA 单轮流水线" if include_preparation else "02 CTA 阶段与单轮放大",
        2160 if include_preparation else 1800,
        2290 if include_preparation else 1520,
    )
    shift = 160 if include_preparation else 0
    page.title(
        "FP8 准备 + 一个 CTA：load → QK → online softmax → PV → 写回"
        if include_preparation
        else "一个 CTA 内部：搬运、两次 MMA 与 online softmax",
        "Q = KV = 4096 · head 0 · 最后 64 个 Q token · 128 threads / 4 warps · 遍历 64 个 KV tile",
    )
    old = [
        ("cp.async\nQ/K 分 4 片\nload + wait", "load"),
        ("QK MMA\n每片读 SMEM\nQ 重复装载", "qk"),
        ("scale / mask\nonline softmax\nP 转 FP8", "softmax"),
        ("PV MMA\nV 分 2 片\nload + wait", "pv"),
        ("O 归一化\nSMEM 中转\n写回 BF16", "write"),
    ]
    new = [
        ("TMA Q/K\nQ 初始装载\n寄存器复用", "load"),
        ("QK MMA\n完整 K tile\n提前发起 V", "qk"),
        ("scale / mask\nonline softmax\n预取下一块 K", "softmax"),
        ("PV MMA\nV 逐片读寄存器\n消费整块 V", "pv"),
        ("O 归一化\n寄存器直接\n写回 BF16", "write"),
    ]
    if include_preparation:
        ms = bycase["dense4096"]["ms"]
        old.insert(
            0,
            (
                "QKV 准备（CTA 外）\n整张量 scale + 拼接 / 量化\nBSHD / 必要时补齐\n"
                f"{ms['prepare_old']:.4f} ms（准备单测）",
                "scale",
            ),
        )
        new.insert(
            0,
            (
                "QKV 准备（CTA 外）\n两个融合 kernel / 分块 scale\n拼接 / 量化 / head-major\n"
                f"{ms['prepare_new']:.4f} ms（准备单测）",
                "scale",
            ),
        )
        page.text(
            36,
            125,
            2088,
            45,
            "量化前上游：RoPE → cache 写入 / 条件性前缀读取 → kv_b_proj 展开（未单独计时，不包含在下方实测中）。",
            21,
            color=MUTED,
        )
        page.text(210, 180, 277, 32, "独立准备 kernel", 20, True, color=COLORS["scale"])
        page.text(
            530,
            180,
            1540,
            32,
            "Attention kernel 内部：初始装载 → 一轮 KV 计算（重复）→ 最终写回",
            20,
            True,
        )
    rows = (
        [(225, "初版", old), (408, "dev-3", new)]
        if include_preparation
        else [(145, "初版", old), (310, "dev-3", new)]
    )
    for y, name, nodes in rows:
        page.text(36, y + 43, 160, 40, name, 23, True)
        previous = None
        for i, (label, kind) in enumerate(nodes):
            x = 210 + i * 313
            node = page.box(
                x, y, 277, 142 if include_preparation else 126, label, COLORS[kind], 19
            )
            if previous:
                center = y + (71 if include_preparation else 63)
                page.line(x - 36, center, x, center, MUTED, previous, node)
            previous = node
    page.text(
        36,
        560 if include_preparation else 452,
        2088 if include_preparation else 1728,
        48,
        "上方为阶段示意，非时间比例；QK → softmax → PV 对每个 KV tile 重复。初版也有 cp.async 预取。\n两版均为 FP8 × FP8 MMA、FP32 累加与 softmax。dev-3 在循环中额外消费分块 scale。",
        16,
        color=MUTED,
    )
    if include_preparation:
        page.box(
            36,
            618,
            2088,
            54,
            f"初版 → dev-3：准备 {ms['prepare_old']:.4f} → {ms['prepare_new']:.4f} ms  ｜  "
            f"Attention {ms['baseline_attention']:.4f} → {ms['dev3_attention']:.4f} ms  ｜  "
            f"连跑 {ms['baseline_pipeline']:.4f} → {ms['dev3_pipeline']:.4f} ms（−43.98%）",
            COLORS["softmax"],
            21,
        )
        joint_timeline(page)
        return page
    page.text(
        36,
        523 + shift,
        1728,
        36,
        "clock64 实测放大：第 33 个 KV tile，warp 0",
        23,
        True,
    )
    maximum = max(
        (max(d[0, 33, 44], d[0, 34, 22]) if version != "baseline" else d[0, 33, 44])
        - d[0, 33, 0]
        for version, (d, _, _) in traces.items()
    )
    xmax = math.ceil(maximum / 1000) * 1000
    x0, width = 235, 1850 if include_preparation else 1480
    scale = width / xmax
    for i, version in enumerate(VERSIONS):
        d, partition, _ = traces[version]
        origin = d[0, 33, 0]
        y = 605 + shift + i * 172
        layer = page.layer(NAMES[i] + "：第 33 个 KV tile")
        page.text(36, y, 185, 45, NAMES[i], 19, True, layer)
        for cycle in range(0, xmax + 1, 1000):
            x = x0 + cycle * scale
            page.line(x, y - 8, x, y + 125)
            if i == 2:
                page.text(x - 10, y + 130, 70, 28, str(cycle // 1000), 15)
        trace_bar(
            page, partition(33), x0, y, scale, origin, 34, version, 0, 33, layer, True
        )
        if version == "baseline":
            windows = [
                (21, 33, 32, "V 首片：issue → wait", "load"),
                (31, 33, 42, "V 尾片：issue → wait", "load"),
            ]
        else:
            windows = [
                (21, 33, 42, "整块 V：issue → wait", "load"),
                (25, 34, 22, "下一块 K：issue → 下一轮 wait", "qk"),
            ]
        for j, (a, step, b, label, kind) in enumerate(windows):
            start, end = d[0, 33, a] - origin, d[0, step, b] - origin
            page.rect(
                x0 + start * scale,
                y + 49 + j * 42,
                (end - start) * scale,
                30,
                label,
                "#ffffff",
                COLORS[kind],
                14,
                layer,
                extra="dashed=1;dashPattern=5 3;strokeWidth=2;",
                metadata={
                    "start_cycle": start,
                    "end_cycle": end,
                    "tooltip": f"{label} | {start} → {end} cycles；这是重叠窗口，不是 DMA 实际耗时",
                },
            )
    page.text(
        x0,
        1120 + shift,
        width,
        30,
        "距本轮开始的 SM clock cycles（×1000）；三行采用相同的周期比例。",
        17,
    )
    page.legend(
        36,
        1168 + shift,
        list(COLORS),
        columns=4,
        width=520 if include_preparation else 430,
    )
    page.text(
        36,
        1255 + shift,
        2088 if include_preparation else 1728,
        43,
        "虚线框为 issue → 观察到 wait 返回的窗口，不是 DMA 精确时长。QK / PV 段包含 SMEM 读取、发射、等待及打点开销。",
        16,
        color=MUTED,
    )
    cards = [
        (
            "共享内存 / CTA\n64 KiB → 48 KiB + 16 B\n共享内存允许驻留：1 → 2 CTA / SM",
            "load",
        ),
        (
            "无打点 NCU 实测\n占用率 8.33% → 16.56%\nlocal load / store 均为 0",
            "softmax",
        ),
        (
            "观测限制\n打点使整核耗时增加约 3.5%～11.0%\n阶段占比不能直接作为单项优化收益",
            "scale",
        ),
    ]
    for i, (label, kind) in enumerate(cards):
        page.box(
            36 + i * (707 if include_preparation else 587),
            1322 + shift,
            674 if include_preparation else 552,
            112,
            label,
            COLORS[kind],
            18,
        )
    page.text(
        36,
        1453 + shift,
        2088 if include_preparation else 1728,
        40,
        "编辑提示：色块是独立对象；悬停查看周期，右键“编辑数据”查看阶段属性。缩放时间条不会自动重算测量数值。",
        16,
        color=MUTED,
    )
    if include_preparation:
        page.text(
            36,
            1660,
            2088,
            32,
            "准备与 Attention 顺序执行；TMA overlap 在 Attention 内。组件单测不能相加替代连跑。ms 为无打点全核耗时，cycles 为单 CTA 打点样本。",
            17,
            color=MUTED,
        )
    return page


def full_timeline(traces):
    page = Page("03 单 CTA 完整时间线", 2000, 1450)
    page.title(
        "一个 CTA 的完整生命周期",
        "clock64 打点 · Q = KV = 4096 · 最后一个逻辑 Q tile / head 0 · 4 个 warp · 64 轮 KV 全部保留",
    )
    page.text(
        36,
        128,
        1915,
        44,
        "时间条为可编辑矩形，按版本分层、按 warp 分组。放大后可选中每个阶段；阶段属性保留版本、warp、轮次和实际周期。",
        17,
        color=MUTED,
    )
    x0, plot_width, xmax = 235, 1690, 415000
    for i, version in enumerate(VERSIONS):
        d, partition, summary = traces[version]
        origin = min(d[w, 0, 0] for w in range(4))
        y = 230 + i * 278
        layer = page.layer(NAMES[i] + "：全部 4 个 warp")
        page.text(
            36,
            y - 40,
            1880,
            35,
            f"{NAMES[i]}：选中样本 CTA 跨度 {summary['cta_cycles_median'] / 1000:.3f} kcycles；样本索引 {summary['sample']}",
            21,
            True,
            layer,
        )
        for cycle in range(0, 400001, 50000):
            x = x0 + cycle / xmax * plot_width
            page.line(x, y - 5, x, y + 186)
            page.text(x - 15, y + 191, 75, 28, str(cycle // 1000), 15)
        for warp in range(4):
            row_y = y + warp * 46
            page.text(95, row_y, 125, 34, f"warp {warp}", 18, parent=layer)
            group = page.rect(
                x0,
                row_y,
                plot_width,
                34,
                parent=layer,
                extra="group;container=1;collapsible=0;recursiveResize=0;",
                metadata={"tooltip": f"{version} | warp {warp} | 全部 64 轮 KV"},
            )
            for step in [0, *range(1, 65), 127]:
                trace_bar(
                    page,
                    partition(step, warp),
                    0,
                    0,
                    plot_width / xmax,
                    origin,
                    34,
                    version,
                    warp,
                    step,
                    group,
                )
    page.text(
        x0,
        1015,
        1690,
        30,
        "距该 CTA 进入被测区域的 SM clock cycles（×1000）；所有版本使用同一横轴比例。",
        17,
    )
    page.legend(36, 1070, list(COLORS), width=480)
    page.text(
        36,
        1180,
        1915,
        36,
        "初始 Q/K load → 64 次 [QK → online softmax → PV] → 最终归一化与 O 写回。",
        22,
        True,
    )
    notes = [
        "每版选择 7 个样本中 CTA 总跨度居中的一次；各 warp 使用同一 CTA 的相对时钟。窄间隔含未分类控制指令。",
        "这是一个 CTA 的观测；整核性能还受同时驻留的 CTA 数量和 causal 调度长尾影响。",
        "打点改变指令序列并增加开销；周期差不能直接当作生产版本的阶段加速比。",
        "步骤属性：step 0 为初始装载，1～64 为 KV 循环，127 为输出。测量原始数据保留在同目录 trace JSON 中。",
    ]
    for i, note in enumerate(notes):
        page.text(36, 1240 + i * 44, 1915, 34, note, 17, color=MUTED)
    return page


def save_drawio(path, pages, uncompressed=False):
    document = ET.Element("mxfile", host="app.diagrams.net", type="device")
    if uncompressed:
        document.set("compressed", "false")
    for index, page in enumerate(pages):
        xml = ET.tostring(page.model, encoding="unicode")
        # draw.io's standard page encoding: URI encode, raw deflate, then base64.
        compressor = zlib.compressobj(level=9, wbits=-15)
        compressed = (
            compressor.compress(quote(xml, safe="~()*!.'-").encode())
            + compressor.flush()
        )
        diagram = ET.SubElement(
            document, "diagram", id=f"fmha-{index + 1}", name=page.name
        )
        if uncompressed:
            diagram.append(ET.fromstring(xml))
        else:
            diagram.text = base64.b64encode(compressed).decode()
    ET.indent(document)
    path.write_bytes(
        ET.tostring(document, encoding="utf-8", xml_declaration=True) + b"\n"
    )
    print(f"{path.name}: {len(pages)} page(s), {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    traces = {version: parse_trace(version) for version in VERSIONS}
    pages = [
        overview(),
        cta_pipeline(traces),
        full_timeline(traces),
        cta_pipeline(traces, include_preparation=True),
    ]
    for name, page in zip(
        (
            "pipeline-overview",
            "cta-pipeline",
            "cta-full-timeline",
            "pipeline-with-preparation",
        ),
        pages,
        strict=True,
    ):
        save_drawio(
            ROOT / f"{name}.drawio",
            [page],
            uncompressed=name == "pipeline-with-preparation",
        )
    save_drawio(
        ROOT / "cta-pipeline-with-preparation.drawio", [pages[-1]], uncompressed=True
    )
    save_drawio(ROOT / "fmha-v2-pipeline-all.drawio", pages)
