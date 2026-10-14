# SM120 GLM FMHA-v2：逐项修改、原实现原因与代码位置

本页解释初版 FMHA-v2 `481a4e4a` 到 dev-3 内核 `1c73959e` 的完整变化。输入仍支持不同 batch、Q/KV 长度和前缀；固定的是这条特化的 head 维度 256/256、64-token tile、4 个 warp 和输出布局。4096 是一个测试长度，不是支持范围的唯一长度。

“之前为什么没有做”分为两类：**代码或提交能确认的事实**，以及**该设计的一般取舍**。后者不是对原作者决策过程的推断。没有单独测量的项目明确写“未单独测量”，不会从整组收益分摊出百分比。

[返回图表与完整测量报告](./README.md)。源码链接固定到实际提交和行号，后续分支更新不会改变这里引用的代码。

**实现提交顺序**

| 提交 | 实现范围 |
|---|---|
| [`481a4e4a`](https://github.com/sjtulizecheng/flashinfer/commit/481a4e4a11e25e742f25162b2fa85d85729dafdc) | 最初 GLM FP8 256/256 接口与 kernel 支持 |
| [`67820b0b`](https://github.com/sjtulizecheng/flashinfer/commit/67820b0b70fc18e29ad96421a46cf66e7724df26) | 完整 QK/PV tile 与单缓冲区复用同步 |
| [`4b8524af`](https://github.com/sjtulizecheng/flashinfer/commit/4b8524af2aea0cadcb29617baec01a013ed338ec) | 寄存器直接写 O，去除 shared-memory 输出中转 |
| [`5f5a0361`](https://github.com/sjtulizecheng/flashinfer/commit/5f5a036187e3838d390aee77b564ffae88721332) | TMA 搬运、Q 寄存器复用、K/V 预取、V 字节转置加载与输出适配 |
| [`28fa8310`](https://github.com/sjtulizecheng/flashinfer/commit/28fa831011312d69a5c64373e1ef7c7a41bde499) | 跨 head、长 Q tile 优先的 causal CTA 调度 |
| [`1c73959e`](https://github.com/sjtulizecheng/flashinfer/commit/1c73959e6dd7597186d0aba06bbe61015382c047) | packed 变长 Q/KV、分块 scale、对应输出重缩放 |
| [`0aed5c41`](https://github.com/sjtulizecheng/sglang/commit/0aed5c41035b09cee1be973c831b36ce8ed3bbc0)（SGLang） | 复用已有融合准备，接入 packed API，去掉 Q 补齐和输出 unpack |

这些是不同阶段的修改。dev-3 分支包含此前的优化，并不表示每一项都在最后一个提交中新增。

**1. 扩大 QK 和 PV 的内部 tile**

之前：每个 CTA 仍处理 64 个 Q token 和一块 64-token KV，但 QK 沿 head 维度每次只处理 64 维；完整 256 维要做四片。PV 沿 KV 归约维度每次处理 32 个 token，要做两片。

修改：为 SM120、E4M3、Q/K 与 V 维度 256/256、BF16/FP16 输出的特化选择完整 QK 256 维和 PV 64-token tile。外层 Q/KV tile 仍为 64 token；底层 `m16n8k32` MMA 没有变成一条更大的指令。

作用：减少分片循环、地址推进、读写缓冲区切换和同步；完整 Q 也可跨 KV 轮次复用。点积数学不变：四个 64 维部分结果仍对应同一个 256 维点积的求和，新版底层多条 MMA 同样使用 FP32 累加器。

之前为什么没有做：初版沿用 `USE_GRANULAR_TILING` 的通用规则。初版提交还记录了当时在 macOS 上只进行了生成/CPU 检查，没有执行 SM120 GPU 测试；它不是经过该 GPU 调优的最优 tile。较小 tile 通常降低单次数据驻留需求，但是否更快要测试。本形状的完整 FP8 Q/K/V 各 16 KiB、共 48 KiB，能放进旧版因 O 中转已经申请的 64 KiB 预算，所以扩大 tile 没有额外推高当时的峰值共享内存。

代码：[初版的 64 维 QK / 32-token PV 规则](https://github.com/sjtulizecheng/flashinfer/blob/481a4e4a11e25e742f25162b2fa85d85729dafdc/csrc/fmha_v2/fmha/traits.h#L42)；[选择完整 GLM tile 的条件](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/flashinfer/jit/attention/fmha_v2/generator_utils.py#L2100)；[完整 tile 的尺寸定义](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fmha/traits.h#L30)。真正改变的是生成器对既有完整 tile 路径的选择，不能把已有尺寸模板也算作新写的代码。实现提交：[`67820b0b`](https://github.com/sjtulizecheng/flashinfer/commit/67820b0b70fc18e29ad96421a46cf66e7724df26)。

收益：扩大 tile 单项未独立测量。历史 5093-token 测试中，“完整 tile＋直接输出”的 attention 合计 1.217822 → 0.887180 ms，耗时下降 27.15%；不能把这 27.15% 全算给 tile。

**2. Q 在 KV 循环外装载，并留在寄存器复用**

之前：初版已经使用寄存器执行 MMA，但 `frag_q` 只保存当前深度分片。四片依次覆盖同一组 fragment 寄存器；`CTA_P_TILE_K != D` 使 `RELOAD_Q` 为真，下个 KV 块再次加载 Q。

修改：初始 TMA Q/K 完成后，在 KV 循环外把完整 Q 加载到 `frag_q`。循环内部的 Q fragment 加载在 TMA 路径被编译期条件跳过；每轮只更新 K/V。

作用：同一个 Q tile 遍历多个 KV 块时避免重复的 Q 全局/共享内存读取。4096 token 最后一个 Q tile 遍历 64 个 KV 块，完整 Q 可以贯穿这些轮次。全局重复读取可能命中硬件缓存，不能把逻辑读次数减少直接换成等比例 DRAM 流量或速度提升。

之前为什么没有做：初版的数据生命周期只覆盖当前 64 维分片，没有为四片 Q 同时保留寄存器空间。这是已实现路径的选择，不是做不到。保留完整 Q 会增加活跃寄存器数量，需要测量是否降低 CTA 驻留或导致 spill。仍按四片计算、但把四片都常驻寄存器也可实现；Q 复用既不要求 TMA，也不逻辑上要求扩大 tile。

代码：[初版 fragment 大小](https://github.com/sjtulizecheng/flashinfer/blob/481a4e4a11e25e742f25162b2fa85d85729dafdc/csrc/fmha_v2/fused_multihead_flash_attention_kernel_noloop_tiled.h#L189)；[RELOAD_Q 条件](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fmha/kernel_traits.h#L180)；[循环前加载完整 Q](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fused_multihead_flash_attention_kernel_noloop_tiled.h#L299)；[TMA 路径跳过循环内 Q 加载](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fused_multihead_flash_attention_kernel_noloop_tiled.h#L358)。实现提交：[`5f5a0361`](https://github.com/sjtulizecheng/flashinfer/commit/5f5a036187e3838d390aee77b564ffae88721332)，在完整 tile 支持之上实现。

收益：没有“只开启 Q 寄存器复用”的成对计时，独立收益未知。初版与最终版都未测到 local-memory spill 读写，不能说这项优化消除了初版寄存器溢出。

**3. 使用 TMA 搬运完整 Q/K/V tile，并适配 shared-memory 布局**

之前：使用 cp.async/LDGSTS，线程按既有分片加载器计算地址并搬运；初版并非同步 memcpy，也已经具有异步预取。

修改：增加 `Sm120_tma_params`、官方 128-byte tensor map、128-byte swizzle 和两组 mbarrier。由被选中的线程发起 `cp.async.bulk.tensor.5d`，每个 Q/K/V 的 64×256 FP8 tile 一次描述；适配 Q/K 的 shared-memory 读取地址。完整 K tile 的收益同时依赖第 1 项，不能重复计算。

作用：改变全局内存到 shared memory 的搬运与地址生成方式，配合下面的发起时机形成流水线。Q/K 初始加载期望 32768 字节，随后单 K/V tile 各 16384 字节。

之前为什么没有做：初版没有这一 SM120 GLM 的 tensor-map/mbarrier 传输实现，沿用了 cp.async 加载器。改成 TMA 需要同时适配布局、barrier phase、字节计数、输入 stride 与输出映射，不能只替换一条加载指令；也没有证据表明 TMA 对所有形状都必然更快。

代码：[tensor map 构造与布局检查](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fmha/sm120_tma.h#L42)；[Q/K shared-memory 读取适配](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fmha/sm120_tma.h#L93)；[TMA load / expect / wait](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fmha/sm120_tma.h#L155)；[初始完整 Q/K 搬运](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fused_multihead_flash_attention_kernel_noloop_tiled.h#L202)。实现提交：[`5f5a0361`](https://github.com/sjtulizecheng/flashinfer/commit/5f5a036187e3838d390aee77b564ffae88721332)。

收益：历史 5093-token 同轮对照，“TMA＋Q/V 加载适配＋预取＋调度”整组 0.889218 → 0.778541 ms，耗时下降 12.45%。该组基线已经包含完整 tile 和直接输出；这不是 TMA 指令单项收益。

**4. 提前发起当前 V 的搬运**

之前：初版在 BMM1 tail 中已经发起 V，但之前的三个 64 维 QK 分片已经算完，只剩最后一片 QK 以及后续 softmax 可与之重叠。

修改：完整 tile 路径没有前三个 BMM1 分片循环，在整段 QK MMA 前就发起当前 V 的 TMA。到 PV 前才等待 V 就绪。

作用：扩大 V 搬运可以与 QK、scale/mask、online softmax 重叠的窗口。等待不一定全部消失，是否隐藏取决于实际搬运和计算耗时。

之前为什么没有更早：原有发起位置绑定在通用分片流水线的 BMM1 tail。更早搬 V 还需要确认缓冲区是否可安全使用；本次完整 tile 和分离的 K/V 缓冲使该时机更合适。不能把它描述为“初版完全没有 V 预取”。

代码：[初版在 QK tail 发起 V](https://github.com/sjtulizecheng/flashinfer/blob/481a4e4a11e25e742f25162b2fa85d85729dafdc/csrc/fmha_v2/fused_multihead_flash_attention_kernel_noloop_tiled.h#L329)；[新版发起 V 并等待 K](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fused_multihead_flash_attention_kernel_noloop_tiled.h#L380)；[到 PV 前等待 V](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fused_multihead_flash_attention_kernel_noloop_tiled.h#L569)。实现提交：[`5f5a0361`](https://github.com/sjtulizecheng/flashinfer/commit/5f5a036187e3838d390aee77b564ffae88721332)。

收益：未单独测量，包含在第 3 项的 TMA 整组对照中。图上 issue→wait 的纹理条是重叠窗口，不是 DMA 实际耗时。

**5. 读完当前 K 后就预取下一块 K**

之前：下一块 Q/K 的发起在 BMM2 tail，能与最后一段 PV 计算重叠。初版已有下一块 K 预取，只是时机较晚。

修改：当前 QK 完成，并确认所有 warp 都结束读取单 K 缓冲区后，立刻发起下一块 K；随后执行当前块的 softmax 和 PV，到下一轮 QK 前等待。

作用：把下一块 K 的搬运向前移动，与当前 softmax/PV 重叠，而非只利用 PV 尾部。

之前为什么没有更早：旧分片实现把指针回绕、下一块 Q/K 预取和 V 等待放在 BMM2 tail。提前复用 K 缓冲必须证明所有 warp 已结束读取；仅移动 load、省掉同步可能覆盖仍在使用的 K。

代码：[初版 BMM2 tail 的下一块预取](https://github.com/sjtulizecheng/flashinfer/blob/481a4e4a11e25e742f25162b2fa85d85729dafdc/csrc/fmha_v2/fused_multihead_flash_attention_kernel_noloop_tiled.h#L484)；[完整 tile 的 K 释放同步及下一块预取](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fused_multihead_flash_attention_kernel_noloop_tiled.h#L409)；[下一块 K TMA](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fused_multihead_flash_attention_kernel_noloop_tiled.h#L418)。实现提交：[`5f5a0361`](https://github.com/sjtulizecheng/flashinfer/commit/5f5a036187e3838d390aee77b564ffae88721332)，复用同步始于 [`67820b0b`](https://github.com/sjtulizecheng/flashinfer/commit/67820b0b70fc18e29ad96421a46cf66e7724df26)。

收益：未单独测量；不能与第 3、4 项的整组收益相加。

**6. V 逐 fragment 读寄存器并执行 PV；使用 SM120 字节转置加载**

之前：非 TMA 路径用 `smem_v.load(frag_v[ki], ki)` 读出当前 ki 所需的一组 V fragment，再由 `gemm` 消费。它不是把整条序列 V 全放寄存器，但当前组 V 的活跃寄存器更多。

修改：适配 TMA 的 V shared-memory 排列，使用 `ldmatrix.sync.aligned.m16n16.x2.trans.shared::cta.b8`；在 ni 循环中只取当前 fragment，马上执行对应的 PV MMA，再继续下一片。

作用：控制同时活跃的 V fragment 数量，使 Q 常驻寄存器和完整 tile 仍能在寄存器预算内运行；同时完成 FP8 字节布局到 MMA 输入的转置适配。“消费整块 V”是多次 fragment 操作合起来完成，并非一条 MMA 处理整块。

之前为什么没有做：原实现使用已有的通用 `load + gemm` 组合和旧 shared-memory 布局。逐 fragment 消费和字节转置加载需要与特定 MMA/P 布局匹配。更小的活跃寄存器集合是否更快要实测；不能把它当作所有 V 布局的通用替代。

代码：[V 字节转置及寄存器顺序](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fmha/sm120_tma.h#L115)；[PV tail 的逐 fragment 消费](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fused_multihead_flash_attention_kernel_noloop_tiled.h#L612)。实现提交：[`5f5a0361`](https://github.com/sjtulizecheng/flashinfer/commit/5f5a036187e3838d390aee77b564ffae88721332)。

收益：未单独测量。最终版约 252 registers/thread，未测到 local load/store；共享内存 bank conflict 仍然存在，不能声称这项修改消除了 bank conflict。

**7. O 从寄存器直接转换并写回 BF16/FP16**

之前：通用 epilogue 把 MMA 累加器写到 shared memory，跨线程重排并读回，再转换、写全局内存。当前 64×256 输出 tile 的 FP32 暂存占 64 KiB。

修改：增加 `store_accumulators`，按 lane/warp 推导寄存器对应的输出行列，在寄存器中完成打包并写显存。原 V 布局将四个相邻输出打包为 8 字节；TMA 的 canonical 布局按相邻两个输出写 4 字节，循环覆盖对应列。保留归一化、输出缩放、dtype 转换和尾行谓词。

作用：省掉 shared-memory 中转，尤其是取消 O 的 64 KiB 资源预算。总预算取 Q/K/V 与 O 工作区的较大值，并非简单相加；去掉 O 后保留 Q/K/V 的 48 KiB，TMA 再加 16 字节 barrier，允许每 SM 两个 CTA。

之前为什么没有做：之前同一个 kernel 的 CTA 布局也固定，且 O 本来就在寄存器里；shared memory 不是该形状的数学必需。初版复用了通用输出路径，没有这条专门映射。新路径必须证明每个输出元素的寄存器归属、无需额外合并 O、地址正确及尾块不越界；不同布局仍保留通用实现。不能解释成“后来输入长度固定了才可以取消中转”。

代码：[初版 shared-memory 输出路径](https://github.com/sjtulizecheng/flashinfer/blob/481a4e4a11e25e742f25162b2fa85d85729dafdc/csrc/fmha_v2/fused_multihead_flash_attention_kernel_noloop_tiled.h#L551)；[直接写回映射和尾行检查](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fmha/gmem_tile_o_packed.h#L1134)；[DIRECT_16BIT_OUTPUT 适用范围](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fmha/kernel_traits.h#L263)；[O 预算置零及总共享内存计算](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fmha/kernel_traits.h#L338)；[写回分派](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fused_multihead_flash_attention_kernel_noloop_tiled.h#L652)。主体提交：[`4b8524af`](https://github.com/sjtulizecheng/flashinfer/commit/4b8524af2aea0cadcb29617baec01a013ed338ec)；TMA 输出排列适配：[`5f5a0361`](https://github.com/sjtulizecheng/flashinfer/commit/5f5a036187e3838d390aee77b564ffae88721332)。

收益：单独 epilogue 的干净 CUDA-event 时延未隔离。与完整 tile 合测为第 1 项的 27.15%。本次 NCU 初版→dev-3：动态共享内存 65536 → 49168 bytes，shared-memory CTA 驻留上限 1 → 2，active-warp 占用率 8.33% → 16.56%。占用率翻倍不等于速度翻倍，直接写显存这一小段也不保证更快。

**8. 跨 head 调整 causal CTA 的执行顺序**

之前：逻辑 Q tile 由 `blockIdx.x` 正序映射，head 来自 `blockIdx.y`。因果 attention 的后部 Q 需要访问更多 KV，CTA 工作量不均。

修改：把 CTA 网格展平，长 Q tile 优先，head 变化最快；同时把新的逻辑 head 传给 TMA，保证实际读到同一请求/head 的数据。

作用：减少执行末尾只剩少数长 CTA 的情况，提高 SM 整体利用率。它改变 CTA 间的分配顺序，不改变每个 Q 的因果语义或 KV 累加顺序；单 CTA 阶段图无法体现全部收益。

之前为什么没有做：旧映射是简单的二维网格映射，没有根据 causal tile 工作量安排长任务。代码没有提供原作者为何未选逆序的决策记录；只能确认本轮通过对照实验发现该映射更合适。该优化仅对匹配的 TMA causal 分支启用。

代码：[linear_tile / q_loop / bidh 映射](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fused_multihead_flash_attention_kernel_noloop_tiled.h#L120)；[TMA 保存逻辑 head](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fmha/sm120_tma.h#L158)。实现提交：[`28fa8310`](https://github.com/sjtulizecheng/flashinfer/commit/28fa831011312d69a5c64373e1ef7c7a41bde499)。

收益：历史单独调度对照 v6→v17，5093 token 0.878224 → 0.790172 ms，下降 10.03%。源码快照对比只有网格/head 映射和相应 TMA head 参数调整。数据见 [historical-cta-scheduling.json](./historical-cta-scheduling.json)；这项已包含在 TMA＋调度的整组收益中，不能重复相加。

**9. 复用两个融合准备 kernel，替代逐张量准备流程**

之前：旧 FMHA 适配器分别执行整张量 abs/amax、scale 运算，再用 pack kernel 拼接 K、量化 Q/K/V、写入 BSHD，并为前缀/变长输入补齐。

修改：SGLang 的新 `prepare` 直接调用既有 `materialized_fp8_quantize.prepare`：一个 kernel 处理 Q，另一个融合 K_nope/K_rope 拼接与 K/V 量化，按每 head、64 token 求 scale，并写 packed head-major 数据。

作用：减少独立 kernel、全张量中间结果和重复读写。head-major 使同一个 head 的 token 连续存放，与新的 TMA 描述符配合。布局本身没有独立收益对照，不能把整组收益全部归给 head-major。

之前为什么没有做：旧 FMHA 的 scalar-scale API 要求不同布局和 scale 粒度，不能直接消费原融合 FP8 准备的输出。本次同时改了消费端和 SGLang 适配，才复用了该流程。两个量化 kernel 在原融合 FP8 分支中已经存在；不是在 dev-3 重新发明或新增的 kernel。

代码：SGLang [prepare 调用已有融合准备](https://github.com/sjtulizecheng/sglang/blob/0aed5c41035b09cee1be973c831b36ce8ed3bbc0/benchmark/scheduler/experimental_sm120/fmha_v2_glm.py#L81)；复用的旧分支实现 [_quant_q](https://github.com/sjtulizecheng/flashinfer/blob/a724e3b5352e7840b99a7218a152746e696753bc/flashinfer/experimental/mla_fp8_sm120/materialized_fp8_quantize.py#L9)、[_concat_quant_kv](https://github.com/sjtulizecheng/flashinfer/blob/a724e3b5352e7840b99a7218a152746e696753bc/flashinfer/experimental/mla_fp8_sm120/materialized_fp8_quantize.py#L39)、[head-major 分配与两个 kernel 启动](https://github.com/sjtulizecheng/flashinfer/blob/a724e3b5352e7840b99a7218a152746e696753bc/flashinfer/experimental/mla_fp8_sm120/materialized_fp8_quantize.py#L94)。SGLang 实现提交：[`0aed5c41`](https://github.com/sjtulizecheng/sglang/commit/0aed5c41035b09cee1be973c831b36ce8ed3bbc0)。注意：这个复用文件在带原实验目录的 FlashInfer 分支中，不在精简的 dev-3 目录树里。

收益：本次 4096-token 准备阶段 0.302220 → 0.139794 ms，下降 53.74%；相应整条中间版→dev-3 流水线 0.773923 → 0.619992 ms，下降 19.89%。这是融合准备/布局/scale 接口整体的收益，不是每个子操作分别降低 53.74%。

**10. 独立 Q/KV 长度，取消 Q 补齐、输出 unpack，并复用输出缓冲**

之前：scalar FMHA 接口按统一 BSHD 序列长度处理 Q/K/V。旧适配器把 Q 补到 KV 长度、不同请求补到统一长度，运行 attention 后用 unpack 取回真实 Q 行。无前缀且单请求等长时，unpack 只是视图，不能声称那里省掉了一次 GPU 拷贝。

修改：增加 packed GLM API、独立 `cu_q` / `cu_kv`、真实请求长度和 head-major tensor map。只为真实 Q 行计算，前缀使用右下对齐 causal 位置；输出直接写真实 Q，适配器优先传入 SGLang 已分配的输出 buffer。

作用：减少补齐行的准备、attention 计算和输出裁剪。Q=3072、KV=4096 时，可从按 4096 个 Q 行计算改为 3072 个真实行，但时间不会按行数简单等比例缩短，因为 causal 每行工作量不同。

之前为什么没有做：旧 API 和相应布局约定不支持这组 packed、Q/KV 独立长度的输入，适配器采用 padding 兼容。要去掉 padding，必须一起改绑定、长度/stride 参数、TMA 地址、causal 位置及输出地址，不能只删除一段补零代码。Q 总长度改变，寄存器对应 tile 内行列的规则仍不变；尾块用实际长度屏蔽越界行。

代码：[新的 Python API](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/flashinfer/prefill.py#L7371)；[packed 绑定、stride/shape 检查及 cu_q/cu_kv](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/trtllm_fmha_v2_binding.cu#L482)；[packed 与 padded 的 TMA stride](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fmha/sm120_tma.h#L45)；[因果位置加上 KV-Q 前缀偏移](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fused_multihead_flash_attention_kernel_noloop_tiled.h#L141)；SGLang [传入真实长度](https://github.com/sjtulizecheng/sglang/blob/0aed5c41035b09cee1be973c831b36ce8ed3bbc0/benchmark/scheduler/experimental_sm120/fmha_v2_glm.py#L168)、[复用输出 buffer 并直接返回](https://github.com/sjtulizecheng/sglang/blob/0aed5c41035b09cee1be973c831b36ce8ed3bbc0/benchmark/scheduler/experimental_sm120/fmha_v2_glm.py#L195)。实现提交：[`1c73959e`](https://github.com/sjtulizecheng/flashinfer/commit/1c73959e6dd7597186d0aba06bbe61015382c047) 和 [`0aed5c41`](https://github.com/sjtulizecheng/sglang/commit/0aed5c41035b09cee1be973c831b36ce8ed3bbc0)。

收益：未将“去掉 padding/unpack”与第 9 项单独拆开。本次 Q=3072/KV=4096，中间版→dev-3 attention 0.497319 → 0.471034 ms；整条流水线 0.789462 → 0.581807 ms，下降 26.30%。变长双请求样例的整条下降 58.69%，同样是整组接口/准备变化，不能当作单独去 padding 的收益。

**11. 支持每块 scale，并在 online softmax 中正确重缩放 O**

之前：旧 FMHA 使用统一 QK scale 和 V scale，不需要在相邻 KV 块之间切换 V 的量化单位。

修改：每个 Q tile 读取 Q scale，每个 KV tile 读取对应 K/V scale。QK 分数乘 Q scale×K scale×softmax scale；V scale 变化时，已有 O 累加器乘以前一个 V scale / 当前 V scale，最后乘回当前 V scale。softmax 分母不乘这个 V scale 比例。

作用：正确消费第 9 项既有融合准备输出的分块量化数据。这是必要的接口与数值支持，不应单独列为已证明让 attention 更快的优化。amax 求 scale 位于准备 kernel，不在 attention 中重新扫描原始 Q/K/V。

之前为什么没有做：旧接口只接收统一 scale，对应数据也按整张量量化，根本不需要这种跨块 V 单位转换。切换为分块 scale 后才需要添加。Q/KV cache 的 scale 与经过 kv_b_proj、拼接后的展开 attention 张量 scale 属于不同阶段，不能因为 cache 的 scale=1 就删掉这里的 scale 处理。

代码：[绑定每 head 每块的 scale](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/trtllm_fmha_v2_binding.cu#L558)；[Q/K/V scale 和 V 比例](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fused_multihead_flash_attention_kernel_noloop_tiled.h#L428)；[按块 QK 缩放](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fmha/softmax.h#L1235)；[O 重缩放与最终归一化](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fmha/fragment.h#L1224)。实现提交：[`1c73959e`](https://github.com/sjtulizecheng/flashinfer/commit/1c73959e6dd7597186d0aba06bbe61015382c047)。

收益：没有独立正收益证据。本次无前缀 4096-token attention 中间版→dev-3 为 0.503777 → 0.513957 ms，反而增加 2.02%；因同时改变布局/地址/scale，不能把这 2.02% 全归因于 scale。整条流水线仍下降 19.89%。两版量化粒度不同，不能宣称 FP8 输出逐 bit 相同；数值误差记录见主报告。

**正确性与分派配套：不能作为额外收益重复累计**

- 完整 tile 的 K/V 变为单缓冲区，分别在所有 warp 结束读取后同步再复用。[K 复用同步](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fused_multihead_flash_attention_kernel_noloop_tiled.h#L409)、[V 复用同步](https://github.com/sjtulizecheng/flashinfer/blob/1c73959e6dd7597186d0aba06bbe61015382c047/csrc/fmha_v2/fused_multihead_flash_attention_kernel_noloop_tiled.h#L629) 是防止竞态所需的修改，不是可任意删除的性能开关。TMA 的字节计数和 phase 管理同理。
- 当前 direct-output 静态条件限定 CTA/MMA 布局；packed 入口检查形状、stride、dtype、device 和 scale 形状。它支持动态序列长度，不代表支持任意 head dimension。
- SGLang [eligible 条件与原路径回退](https://github.com/sjtulizecheng/sglang/blob/0aed5c41035b09cee1be973c831b36ce8ed3bbc0/benchmark/scheduler/experimental_sm120/fmha_v2_glm.py#L129) 只在符合条件时使用此路径。源 checkout 按内容 hash 进行 JIT，避免旧缓存掩盖改动；这属于集成机制，不是算子速度贡献。
- O 直接写回不会新增整份 O 寄存器缓存：累加器原本就在寄存器里。当前测量没有寄存器 spill 读写，不代表未来扩大 tile 或改变形状也不会 spill。
- 打点补丁只用于诊断，未启用在生产路径。它有观测开销，不能算作优化项或从色块长度推导单项生产收益。

**性能数据如何对应**

| 数据 | 对照口径 | 可用于说明什么 |
|---|---|---|
| [latency.json](./latency.json) | 本次三个无打点版本，五种形状，每项五轮交替顺序取中位数 | 初版→dev-3 4096-token attention -34.29%，整条 -43.98%；组件独立计时不能直接相加 |
| [完整 tile＋直接输出历史数据](../../../benchmarks/results/fmha_v2_sm120_glm_20261014.json) | 旧测试：初版 vs 完整 tile＋直接输出，纯 attention | 5093-token 1.217822→0.887180 ms，-27.15% |
| [historical-tma-group.json](./historical-tma-group.json) | 旧测试：已有完整 tile＋直接输出 vs TMA＋调度整组 | 5093-token 0.889218→0.778541 ms，-12.45% |
| [historical-cta-scheduling.json](./historical-cta-scheduling.json) | 旧测试：v6 vs v17，只调整 causal 网格/head 映射及传参 | 5093-token 0.878224→0.790172 ms，-10.03%；该收益包含在上行整组内 |
| [ncu-summary.json](./ncu-summary.json) | 本次 clean/trace 各版本的资源计数 | 共享内存、CTA 驻留上限、占用率、local 读写、bank conflict；不能当作 CUDA-event 时延 |

旧测试与本次测试不是同一轮，不能拼成严格的逐项百分比分解；耗时下降百分比不相加。本页只是整理已有数据，未新跑服务或 GPU 消融。纯算子收益也不能直接替代完整 SGLang 请求的 QPS 收益。
