"""Add diagnostic-only, per-warp clock64 stamps to isolated FMHA worktrees.

No barriers are added. Only lane 0 of each warp in (batch=0, head=0,
last logical Q tile) records timestamps. These are observer timings, not
Tensor Core busy counters or asynchronous-copy completion timestamps.
"""

from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parent
VARIANTS = ("baseline", "tma_padded", "dev3")


def library(root):
    from flashinfer.jit import env

    csrc = root / "csrc"
    codegen = root / "flashinfer/jit/attention/fmha_v2"
    paths = sorted((csrc / "fmha_v2").rglob("*")) + [
        csrc / "trtllm_fmha_v2_binding.cu",
        csrc / "tvm_ffi_utils.h",
        codegen / "generate_kernels.py",
        codegen / "generator_utils.py",
    ]
    digest = hashlib.sha256()
    for p in paths:
        if p.is_file():
            digest.update(str(p.relative_to(root)).encode())
            digest.update(p.read_bytes())
    name = "glm_sm120_fmha_v2_" + digest.hexdigest()[:16]
    return str(env.FLASHINFER_JIT_DIR / name / (name + ".so"))


def once(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:100])
    return text.replace(old, new)


def section(text, start, end, transform):
    a = text.index(start)
    b = text.index(end, a)
    return text[:a] + transform(text[a:b]) + text[b:]


def stamp(slot, event):
    return f"    FI_CTA_STAMP({slot}, {event});\n"


HEADER = r"""// Diagnostic-only clock64 recording; not part of production dispatch.
#pragma once
#define FI_CTA_STAMP(SLOT, EVENT)                                                   \
  do {                                                                            \
    if ((threadIdx.x & 31) == 0 && params.cta_trace != nullptr && bidb == 0 &&        \
        bidh == 0 && q_loop == (params.s - 1) / 64) {                               \
      uint64_t trace_clock;                                                       \
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(trace_clock) : : "memory");       \
      params.cta_trace[((threadIdx.x >> 5) * 128 + (SLOT)) * 48 + (EVENT)] =          \
          trace_clock;                                                            \
    }                                                                             \
  } while (0)
"""


def instrument(root):
    header = root / "csrc/fmha_v2/fmha/cta_trace.h"
    header.write_text(HEADER)
    params = root / "csrc/fmha_v2/fused_multihead_attention.h"
    text = params.read_text()
    text = once(
        text,
        "  int dv = 0;",
        "  int dv = 0;\n  uint64_t* cta_trace = nullptr;  // Diagnostic build only.",
    )
    params.write_text(text)

    binding = root / "csrc/trtllm_fmha_v2_binding.cu"
    text = binding.read_text()
    text = once(
        text,
        "bool is_bf16_output) {",
        "bool is_bf16_output, Optional<TensorView> maybe_trace) {",
    )
    set_trace = """
  if (maybe_trace.has_value()) {
    auto trace = maybe_trace.value();
    CHECK_INPUT_AND_TYPE(trace, dl_int64);
    CHECK_DEVICE(q, trace);
    TVM_FFI_ICHECK_EQ(trace.numel(), 32 * 128 * 48);
    TVM_FFI_ICHECK_LE(params.s, 8000);
    params.cta_trace = static_cast<uint64_t*>(trace.data_ptr());
  }
"""
    anchor = "  params.is_uniform_q = batch_size > 0;"
    text = once(text, anchor, anchor + set_trace)
    if "void TRTLLMFMHAv2RunGLM" in text:
        text = once(
            text,
            "double softmax_scale, bool causal) {",
            "double softmax_scale, bool causal, Optional<TensorView> maybe_trace) {",
        )
        anchor = "  params.sage.v = {64, nk, static_cast<float*>(sv.data_ptr())};"
        text = once(text, anchor, anchor + set_trace)
    binding.write_text(text)

    p = root / "csrc/fmha_v2/fused_multihead_flash_attention_kernel_noloop_tiled.h"
    text = p.read_text()
    text = once(
        text,
        "#include <fmha/gemm.h>",
        "#include <fmha/gemm.h>\n#include <fmha/cta_trace.h>",
    )
    text = once(
        text,
        "  // Create the object to control the masks.",
        stamp(0, 0) + "  // Create the object to control the masks.",
    )
    text = once(
        text,
        "  // Declare the fragments for Q/K/V.",
        stamp(0, 1) + "  // Declare the fragments for Q/K/V.",
    )
    loop = "  for (int kv_loop = kv_loop_start; kv_loop < kv_loop_end; kv_loop += Cta_tile_p::N) {"
    text = once(text, loop, stamp(0, 2) + loop + "\n" + stamp("kv_loop / 64 + 1", 0))
    slot = "kv_loop / 64 + 1"

    def qk_main(s):
        s = once(
            s,
            "      // Trigger the load for the next Q/K values",
            stamp(slot, "2 + bmm1_k / Mma_tile_p::MMAS_K * 5")
            + "      // Trigger the load for the next Q/K values",
        )
        s = once(
            s,
            "      // Double SMEM buffer:",
            stamp(slot, "3 + bmm1_k / Mma_tile_p::MMAS_K * 5")
            + "      // Double SMEM buffer:",
        )
        s = once(
            s,
            "// Do this part of P^T",
            stamp(slot, "4 + bmm1_k / Mma_tile_p::MMAS_K * 5")
            + "// Do this part of P^T",
        )
        s = once(
            s,
            "      // Make sure we are done reading the data (smem).",
            stamp(slot, "5 + bmm1_k / Mma_tile_p::MMAS_K * 5")
            + "      // Make sure we are done reading the data (smem).",
        )
        s = once(
            s,
            "      smem_q.move_to_next_read_buffer();",
            stamp(slot, "6 + bmm1_k / Mma_tile_p::MMAS_K * 5")
            + "      smem_q.move_to_next_read_buffer();",
        )
        return s

    text = section(text, "    // BMM1 main loop", "    // BMM1 tail loop", qk_main)

    def qk_tail(s):
        s = once(
            s,
            "      // Trigger the load for next V values",
            stamp(slot, 20) + "      // Trigger the load for next V values",
        )
        if "tma.wait(0," in s:
            s = once(s, "        tma.wait(0,", stamp(slot, 21) + "        tma.wait(0,")
        else:
            s = once(
                s,
                "      // Wait for Q/K to finish loading",
                stamp(slot, 21) + "      // Wait for Q/K to finish loading",
            )
        s = once(
            s, "// Do this part of P^T", stamp(slot, 22) + "// Do this part of P^T"
        )
        s = once(
            s,
            "      smem_q.move_to_next_read_buffer();",
            stamp(slot, 23) + "      smem_q.move_to_next_read_buffer();",
        )
        return s

    text = section(text, "    // BMM1 tail loop", "    // Store the P matrix.", qk_tail)
    text = once(
        text,
        "    // Store the P matrix.\n#if defined(STORE_P)",
        stamp(slot, 25) + "    // Store the P matrix.\n#if defined(STORE_P)",
    )
    text = once(
        text,
        "    // First step of the flash attention",
        stamp(slot, 26) + "    // First step of the flash attention",
    )
    text = once(
        text,
        "    // Repack for the next BMM.",
        stamp(slot, 27) + "    // Repack for the next BMM.",
    )
    text = once(
        text,
        "    softmax.pack(frag_p);",
        "    softmax.pack(frag_p);\n" + stamp(slot, 28),
    )

    def pv_main(s):
        s = once(
            s,
            "      // Trigger the load for next V values",
            stamp(slot, 30) + "      // Trigger the load for next V values",
        )
        s = once(
            s,
            "      // Leave 1 outstanding batch loadings",
            stamp(slot, 31) + "      // Leave 1 outstanding batch loadings",
        )
        s = s.replace("#pragma unroll", stamp(slot, 32) + "#pragma unroll", 1)
        s = once(
            s,
            "      // Make sure we are done reading the data (smem).",
            stamp(slot, 33) + "      // Make sure we are done reading the data (smem).",
        )
        s = once(
            s,
            "      smem_v.move_to_next_read_buffer();",
            stamp(slot, 34) + "      smem_v.move_to_next_read_buffer();",
        )
        return s

    text = section(text, "    // BMM2 main loop", "    // BMM2 tail loop", pv_main)

    def pv_tail(s):
        s = once(s, "    {", "    {\n" + stamp(slot, 40))
        s = once(
            s,
            "// Do this part of O = P^T * V^T.",
            stamp(slot, 42) + "// Do this part of O = P^T * V^T.",
        )
        s = once(
            s,
            "      smem_v.move_to_next_read_buffer();",
            stamp(slot, 43) + "      smem_v.move_to_next_read_buffer();",
        )
        return s

    text = section(text, "    // BMM2 tail loop", "  }  // Inner loop over", pv_tail)
    text = once(
        text, "  }  // Inner loop over", stamp(slot, 44) + "  }  // Inner loop over"
    )
    text = once(
        text,
        "  // Update the sum if attention sinks are used.",
        stamp(127, 0) + "  // Update the sum if attention sinks are used.",
    )
    text = once(
        text,
        "  // If kv_loop breaks prematurely",
        stamp(127, 1) + "  // If kv_loop breaks prematurely",
    )
    text = once(
        text,
        "  // Wait for last round of LDS K/V to finish\n  __syncthreads();",
        "  // Wait for last round of LDS K/V to finish\n  __syncthreads();\n"
        + stamp(127, 2),
    )
    text = once(
        text,
        "  if (params.softmax_stats_ptr != nullptr)",
        stamp(127, 3) + "  if (params.softmax_stats_ptr != nullptr)",
    )
    text = once(
        text,
        "}  // device_flash_attention_1xN",
        stamp(127, 4) + "}  // device_flash_attention_1xN",
    )
    p.write_text(text)


if __name__ == "__main__":
    records = {}
    for variant in VARIANTS:
        root = ROOT / variant
        clean = library(root)
        assert Path(clean).is_file(), clean
        instrument(root)
        records[variant] = {"clean_library": clean, "trace_library": library(root)}
        patch = subprocess.check_output(["git", "-C", str(root), "diff", "--binary"])
        rel = "csrc/fmha_v2/fmha/cta_trace.h"
        lines = (root / rel).read_text().splitlines()
        added = (
            f"diff --git a/{rel} b/{rel}\nnew file mode 100644\n"
            f"--- /dev/null\n+++ b/{rel}\n@@ -0,0 +1,{len(lines)} @@\n"
            + "".join("+" + line + "\n" for line in lines)
        )
        (ROOT / f"trace-{variant}.patch").write_bytes(patch + added.encode())
    (ROOT / "libraries.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))
