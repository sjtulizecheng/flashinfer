"""Operator-only latency and selected-CTA trace measurements."""

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [
    "/sgl-workspace/sglang/benchmark/scheduler/experimental_sm120",
    "/sgl-workspace/flashinfer-fp8-mla-sm120/flashinfer/experimental/mla_fp8_sm120",
]
import torch
import triton.testing
import tvm_ffi
import fmha_v2_glm as new

spec = importlib.util.spec_from_file_location(
    "old_adapter",
    "/sgl-workspace/optimization-delivery/fmha-v2-packed-20261014/fmha_v2_glm_before.py",
)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

CASES = {
    "dense1024": ([1024], [1024]),
    "dense4096": ([4096], [4096]),
    "dense5093": ([5093], [5093]),
    "prefix": ([3072], [4096]),
    "ragged": ([1024, 512], [1536, 768]),
}


class Traced:
    def __init__(self, module, trace):
        self.module, self.trace = module, trace

    def run(self, *args):
        return self.module.run(*args, self.trace)

    def run_glm(self, *args):
        return self.module.run_glm(*args, self.trace)


def inputs(case):
    qlens, klens = CASES[case]
    data = torch.load(
        "/tmp/chat-opt-20261009/real-prefill/23.pt",
        map_location="cuda",
        weights_only=True,
    )
    total = sum(klens)
    qsrc, k, v = [data[key][:total] for key in ("q", "k", "v")]
    qparts = []
    offset = 0
    for nq, nk in zip(qlens, klens, strict=True):
        qparts.append(qsrc[offset + nk - nq : offset + nk])
        offset += nk
    q = torch.cat(qparts)
    cu = lambda ns: torch.tensor(
        [0] + list(torch.tensor(ns).cumsum(0)), device="cuda", dtype=torch.int32
    )
    cq, ck = cu(qlens), cu(klens)
    return q, k[..., :192], k[:, :1, 192:], v, cq, ck, qlens, klens


def setup(case, modules):
    q, kn, kr, v, cq, ck, qlens, klens = inputs(case)
    prepare_old = lambda: old.prepare(q, kn, kr, v, cq, ck, max(klens), 1 / 16)
    prepare_new = lambda: new.prepare(
        q, kn, kr, v, cq, ck, max(qlens), max(klens), 1 / 16
    )
    po, pn = prepare_old(), prepare_new()
    out_old = torch.empty(po[0].shape, device=q.device, dtype=torch.bfloat16)
    out_new = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
    funcs = {"prepare_old": prepare_old, "prepare_new": prepare_new}
    for name, module in modules.items():
        if name == "dev3":
            funcs[name + "_attention"] = lambda m=module: new.run_packed(
                m, pn, out=out_new
            )
            funcs[name + "_pipeline"] = lambda m=module: new.run_packed(
                m, prepare_new(), out=out_new
            )
        else:
            funcs[name + "_attention"] = lambda m=module: old.run_packed(
                m, po, out=out_old
            )
            funcs[name + "_pipeline"] = lambda m=module: old.unpack(
                old.run_packed(m, prepare_old(), out=out_old),
                cq,
                ck,
                q.shape[0],
                max(qlens),
            )
    funcs["output_unpack"] = lambda: old.unpack(out_old, cq, ck, q.shape[0], max(qlens))
    return funcs, (po, pn, out_old, out_new), (qlens, klens)


def trace_records(tensor):
    a = tensor.cpu().tolist()
    rows = []
    for warp, steps in enumerate(a):
        for step, events in enumerate(steps):
            for event, cycles in enumerate(events):
                if cycles:
                    rows.append(
                        {"warp": warp, "step": step, "event": event, "cycles": cycles}
                    )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", choices=["latency", "trace", "profile"], default="latency"
    )
    ap.add_argument("--variant", default="dev3")
    ap.add_argument("--instrumented", action="store_true")
    ap.add_argument("--case", default="dense4096")
    args = ap.parse_args()
    libraries = json.loads((ROOT / "libraries.json").read_text())
    if args.mode == "latency":
        modules = {
            name: tvm_ffi.load_module(paths["clean_library"])
            for name, paths in libraries.items()
        }
        results = []
        for case in CASES:
            funcs, buffers, (qlens, klens) = setup(case, modules)
            # Initial and TMA scalar implementations must agree on identical
            # prepared operands; the packed API uses a different quantization.
            first = funcs["baseline_attention"]().clone()
            middle = funcs["tma_padded_attention"]().clone()
            torch.testing.assert_close(first, middle, rtol=0, atol=0)
            for f in funcs.values():
                for _ in range(3):
                    f()
            samples = {key: [] for key in funcs}
            for repeat in range(5):
                order = list(funcs)[:: 1 if repeat % 2 == 0 else -1]
                for key in order:
                    samples[key].append(
                        triton.testing.do_bench_cudagraph(funcs[key], rep=50)
                    )
            row = {
                "case": case,
                "qlens": qlens,
                "klens": klens,
                "ms": {k: statistics.median(v) for k, v in samples.items()},
                "samples_ms": samples,
            }
            results.append(row)
            (ROOT / "latency.json").write_text(json.dumps(results, indent=2))
            print(
                json.dumps({k: v for k, v in row.items() if k != "samples_ms"}),
                flush=True,
            )
        return

    paths = libraries[args.variant]
    clean = tvm_ffi.load_module(paths["clean_library"])
    instrumented = tvm_ffi.load_module(paths["trace_library"])
    trace = torch.zeros((32, 128, 48), device="cuda", dtype=torch.int64)
    recording = Traced(instrumented, trace)
    funcs, buffers, _ = setup(args.case, {args.variant: clean})
    po, pn, out_old, out_new = buffers
    run = (
        lambda m: new.run_packed(m, pn, out=out_new)
        if args.variant == "dev3"
        else old.run_packed(m, po, out=out_old)
    )
    for _ in range(3):
        run(clean)
    expected = run(clean).clone()
    run(recording)
    torch.testing.assert_close(run(recording), expected, rtol=0, atol=0)
    if args.mode == "profile":
        m = recording if args.instrumented else clean
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        run(m)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return

    traces = []
    for _ in range(7):
        trace.zero_()
        run(recording)
        torch.cuda.synchronize()
        traces.append(trace_records(trace))
    # Measure observer overhead separately. Published speedups use only clean
    # kernels from the latency mode, never an instrumented build.
    variants = {
        "clean": clean,
        "compiled_trace_disabled": Traced(instrumented, None),
        "trace_enabled": recording,
    }
    samples = {name: [] for name in variants}
    for repeat in range(5):
        for name in list(variants)[:: 1 if repeat % 2 == 0 else -1]:
            samples[name].append(
                triton.testing.do_bench_cudagraph(
                    lambda m=variants[name]: run(m), rep=30
                )
            )
    result = {
        "variant": args.variant,
        "case": args.case,
        "traces": traces,
        "observer_ms": {k: statistics.median(v) for k, v in samples.items()},
        "observer_samples_ms": samples,
        "outputs_bitwise_equal": True,
    }
    (ROOT / f"trace-{args.variant}-{args.case}.json").write_text(json.dumps(result))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("traces", "observer_samples_ms")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
