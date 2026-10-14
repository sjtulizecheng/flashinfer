"""Measure prepare and a selected CTA on one GPU-global nanosecond clock.

Uses the existing diagnostic worktrees with clock64 replaced by globaltimer.
The two prepare boundaries include marker/dispatch overhead. Production
speedups remain based on the original uninstrumented latency.json, not this
observer trace. No production source or running service is modified.
"""

import argparse
import gc
import gzip
import json
from pathlib import Path
import statistics

import measure
import torch
import triton
import triton.language as tl
import triton.testing
import tvm_ffi


@triton.jit
def stamp(pointer, slot: tl.constexpr):
    ns = tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;",
        constraints="=l",
        args=[],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )
    tl.store(pointer + slot, ns)


def measure_variant(variant, traced_module, clean_module):
    q, kn, kr, v, cq, ck, qlens, klens = measure.inputs("dense4096")

    def prepare():
        if variant == "dev3":
            return measure.new.prepare(
                q, kn, kr, v, cq, ck, max(qlens), max(klens), 1 / 16
            )
        return measure.old.prepare(q, kn, kr, v, cq, ck, max(klens), 1 / 16)

    adapter = measure.new if variant == "dev3" else measure.old
    initial = prepare()
    out = torch.empty(initial[0].shape, dtype=torch.bfloat16, device="cuda")
    trace = torch.zeros((32, 128, 48), dtype=torch.int64, device="cuda")
    boundaries = torch.zeros(3, dtype=torch.int64, device="cuda")
    recording = measure.Traced(traced_module, trace)
    expected = adapter.run_packed(clean_module, initial, out=out).clone()
    adapter.run_packed(recording, initial, out=out)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def pipeline(module, markers=False):
        if markers:
            stamp[(1,)](boundaries, 0, num_warps=1)
        packed = prepare()
        if markers:
            stamp[(1,)](boundaries, 1, num_warps=1)
        adapter.run_packed(module, packed, out=out)
        if markers:
            stamp[(1,)](boundaries, 2, num_warps=1)

    for _ in range(5):
        pipeline(recording, True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pipeline(recording, True)
    for _ in range(50):
        graph.replay()
    torch.cuda.synchronize()

    samples = []
    for _ in range(7):
        trace.zero_()
        graph.replay()
        torch.cuda.synchronize()
        times = boundaries.cpu().tolist()
        records = measure.trace_records(trace)
        for record in records:
            record["ns"] = record.pop("cycles")
        assert times[0] < times[1] < times[2]
        assert all(times[1] <= r["ns"] <= times[2] for r in records)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        samples.append({"boundaries_ns": times, "records": records})

    spans = [s["boundaries_ns"][2] - s["boundaries_ns"][0] for s in samples]
    selected = sorted(range(len(samples)), key=lambda i: spans[i])[len(samples) // 2]
    chosen = samples[selected]
    t0, t1, t2 = chosen["boundaries_ns"]
    d = {(r["warp"], r["step"], r["event"]): r["ns"] for r in chosen["records"]}
    clean_ms = []
    instrumented_ms = []
    for _ in range(3):
        clean_ms.append(
            triton.testing.do_bench_cudagraph(lambda: pipeline(clean_module), rep=30)
        )
        instrumented_ms.append(
            triton.testing.do_bench_cudagraph(lambda: pipeline(recording, True), rep=30)
        )
    result = {
        "variant": variant,
        "case": "dense4096",
        "clock": "GPU %globaltimer",
        "unit": "nanoseconds",
        "sample": selected,
        "samples": samples,
        "outputs_bitwise_equal": True,
        "summary_us": {
            "prepare_interval": (t1 - t0) / 1000,
            "attention_interval": (t2 - t1) / 1000,
            "pipeline_interval": (t2 - t0) / 1000,
            "selected_cta_start": (min(d[w, 0, 0] for w in range(4)) - t0) / 1000,
            "selected_cta_end": (max(d[w, 127, 4] for w in range(4)) - t0) / 1000,
            "tile33_start_warp0": (d[0, 33, 0] - t0) / 1000,
            "tile33_end_warp0": (d[0, 33, 44] - t0) / 1000,
        },
        "observer_ms": {
            "clean_pipeline": statistics.median(clean_ms),
            "instrumented_pipeline": statistics.median(instrumented_ms),
        },
        "notes": [
            "Markers and selected-CTA stamps share GPU globaltimer; no SM-frequency conversion.",
            "Prepare is a whole multi-kernel interval, not a single CTA duration.",
            "Boundary intervals include marker and graph dispatch overhead.",
            "Instrumentation changes execution; original latency.json remains the speedup source.",
        ],
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--variant", choices=["baseline", "tma_padded", "dev3"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    libraries = json.loads(
        (Path(__file__).resolve().parent / "libraries.json").read_text()
    )
    variants = [args.variant] if args.variant else list(libraries)
    summaries = {}
    for variant in variants:
        header = (
            args.source_root / variant / "csrc/fmha_v2/fmha/cta_trace.h"
        ).read_text()
        if "%%globaltimer" not in header or "%%clock64" in header:
            raise ValueError(
                "Apply joint-globaltimer.patch to an isolated diagnostic source first"
            )
        print("BUILD_GLOBALTIMER", variant, flush=True)
        module = measure.new.load_kernel(str(args.source_root / variant))
        clean = tvm_ffi.load_module(libraries[variant]["clean_library"])
        result = measure_variant(variant, module, clean)
        filename = args.output_dir / f"joint-timeline-{variant}.json.gz"
        filename.write_bytes(gzip.compress(json.dumps(result).encode(), mtime=0))
        summaries[variant] = {k: v for k, v in result.items() if k != "samples"}
        print(json.dumps(summaries[variant]), flush=True)
        del result, module, clean
        gc.collect()
        torch.cuda.empty_cache()
    (args.output_dir / "joint-timeline-summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
