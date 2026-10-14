"""Export the measured diagrams as editable draw.io shapes (no embedded images)."""

import base64
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


def preparation_overview():
    """One-page flow including unmeasured work before Q/K/V quantization."""
    page = Page("04 含量化前准备的整体流程", 1920, 1200)
    ms = bycase["dense4096"]["ms"]
    prep_drop = 100 * (1 - ms["prepare_new"] / ms["prepare_old"])
    attn_drop = 100 * (1 - ms["dev3_attention"] / ms["baseline_attention"])
    total_drop = 100 * (1 - ms["dev3_pipeline"] / ms["baseline_pipeline"])
    page.text(60, 32, 1800, 60, "FMHA-v2：从量化前准备到 Attention 输出", 40, True)
    page.text(
        60,
        103,
        1800,
        36,
        "SM120 / RTX PRO 5000 · GLM 第 23 层真实激活 · Q = KV = 4096 · 初版 → dev-3",
        23,
        color=MUTED,
    )
    page.box(
        60,
        162,
        1800,
        90,
        f"02 + 03 连跑实测：{ms['baseline_pipeline']:.4f} ms → "
        f"{ms['dev3_pipeline']:.4f} ms     耗时下降 {total_drop:.2f}%",
        "#218574",
        32,
    )
    columns = [(220, 400), (690, 500), (1260, 600)]
    for (x, w), title in zip(
        columns,
        (
            "01  量化前准备",
            "02  Q/K/V 量化与布局准备",
            "03  Attention → BF16 输出",
        ),
        strict=True,
    ):
        page.text(x, 278, w, 40, title, 26, True)

    def card(x, y, w, title, detail, timing, color):
        parent = page.rect(
            x,
            y,
            w,
            240,
            fill="#ffffff",
            stroke=color,
            extra="rounded=1;arcSize=9;strokeWidth=2;",
        )
        page.rect(0, 0, w, 60, fill=color, parent=parent, extra="rounded=1;arcSize=9;")
        page.text(20, 10, w - 40, 40, title, 25, True, parent=parent, color="#ffffff")
        page.text(20, 77, w - 40, 100, detail, 22, parent=parent)
        page.text(20, 185, w - 40, 40, timing, 28, True, parent=parent, color=color)
        return parent

    for i, y in enumerate((338, 674)):
        old = i == 0
        color = "#68788d" if old else "#218574"
        page.text(
            60, y + 65, 145, 50, "优化前" if old else "优化后", 31, True, color=color
        )
        page.text(60, y + 121, 145, 40, "初版" if old else "dev-3", 25, color=MUTED)
        a = card(
            220,
            y,
            400,
            "上游处理（合并为一个 stage）",
            "RoPE 与 cache 写入\n有前缀时读取压缩 cache\nkv_b_proj 展开各 head 的 K/V",
            "—  未单独测量",
            "#68788d",
        )
        b = card(
            690,
            y,
            500,
            "整张量动态 scale" if old else "每 head / 64 token 动态 scale",
            "独立 abs / amax + scale 运算\npack：拼接 K、量化 Q/K/V\n写 BSHD；前缀 / 变长时补齐"
            if old
            else "两个融合 kernel：Q 与 K/V\n求 scale + 拼接 + 量化\n写 head-major；保留真实长度",
            f"{ms['prepare_old' if old else 'prepare_new']:.4f} ms  ·  准备单测",
            "#287fa8",
        )
        c = card(
            1260,
            y,
            600,
            "初版 FMHA-v2" if old else "FMHA-v2 dev-3",
            "cp.async 分片装载 Q/K/V\nQK → online softmax → P 转 FP8 → PV\nO 经共享内存中转，再写回"
            if old
            else "完整 tile + TMA 预取；Q 寄存器复用\nQK → online softmax → P 转 FP8 → PV\n读取分块 scale；O 从寄存器直接写回",
            f"{ms['baseline_attention' if old else 'dev3_attention']:.4f} ms  ·  Attention 单测",
            "#7958a1",
        )
        page.line(620, y + 120, 690, y + 120, MUTED, a, b)
        page.line(1190, y + 120, 1260, y + 120, MUTED, b, c)
        page.text(220, y + 254, 400, 35, "01 不包含在右侧连跑计时中", 21, color=MUTED)
        page.rect(
            690,
            y + 253,
            1170,
            47,
            f"02 + 03 连跑实测：{ms['baseline_pipeline' if old else 'dev3_pipeline']:.4f} ms",
            fill="#f2f6f8",
            size=25,
            extra="fontStyle=1;",
        )
    page.text(220, 984, 400, 38, "流程方框不按耗时比例绘制", 21, color=MUTED)
    page.text(
        690, 980, 500, 44, f"准备耗时下降 {prep_drop:.2f}%", 28, True, color="#287fa8"
    )
    page.text(
        1260,
        980,
        600,
        44,
        f"Attention 耗时下降 {attn_drop:.2f}%",
        28,
        True,
        color="#7958a1",
    )
    page.line(60, 1050, 1860, 1050)
    page.text(
        60,
        1070,
        1800,
        40,
        "执行关系：01 → 02 → 03；量化准备与 Attention 顺序执行。TMA 搬运 / 计算重叠发生在 03 内部。",
        23,
    )
    page.text(
        60,
        1120,
        1800,
        50,
        "计时口径：组件单测不能相加替代连跑实测。数据来自 latency.json，无打点、5 轮中位数；不是服务 E2E / QPS。",
        22,
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


def cta_pipeline(traces):
    page = Page("02 CTA 阶段与单轮放大", 1800, 1520)
    page.title(
        "一个 CTA 内部：搬运、两次 MMA 与 online softmax",
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
    for y, name, nodes in [(145, "初版", old), (310, "dev-3", new)]:
        page.text(36, y + 43, 160, 40, name, 23, True)
        previous = None
        for i, (label, kind) in enumerate(nodes):
            x = 210 + i * 313
            node = page.box(x, y, 277, 126, label, COLORS[kind], 19)
            if previous:
                page.line(x - 36, y + 63, x, y + 63, MUTED, previous, node)
            previous = node
    page.text(
        36,
        452,
        1728,
        48,
        "上方为阶段示意，非时间比例；QK → softmax → PV 对每个 KV tile 重复。初版也有 cp.async 预取。\n两版均为 FP8 × FP8 MMA、FP32 累加与 softmax。dev-3 在循环中额外消费分块 scale。",
        16,
        color=MUTED,
    )
    page.text(36, 523, 1728, 36, "clock64 实测放大：第 33 个 KV tile，warp 0", 23, True)
    maximum = max(
        (max(d[0, 33, 44], d[0, 34, 22]) if version != "baseline" else d[0, 33, 44])
        - d[0, 33, 0]
        for version, (d, _, _) in traces.items()
    )
    xmax = math.ceil(maximum / 1000) * 1000
    x0, width = 235, 1480
    scale = width / xmax
    for i, version in enumerate(VERSIONS):
        d, partition, _ = traces[version]
        origin = d[0, 33, 0]
        y = 605 + i * 172
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
        1120,
        width,
        30,
        "距本轮开始的 SM clock cycles（×1000）；三行采用相同的周期比例。",
        17,
    )
    page.legend(36, 1168, list(COLORS), columns=4)
    page.text(
        36,
        1255,
        1728,
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
        page.box(36 + i * 587, 1322, 552, 112, label, COLORS[kind], 18)
    page.text(
        36,
        1453,
        1728,
        40,
        "编辑提示：色块是独立对象；悬停查看周期，右键“编辑数据”查看阶段属性。缩放时间条不会自动重算测量数值。",
        16,
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


def save_drawio(path, pages):
    document = ET.Element("mxfile", host="app.diagrams.net", type="device")
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
        preparation_overview(),
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
        save_drawio(ROOT / f"{name}.drawio", [page])
    save_drawio(ROOT / "fmha-v2-pipeline-all.drawio", pages)
