# SM120 GLM FMHA-v2 backport to 0.6.15.post1

This development branch adds packed, block-scaled GLM FP8 prefill to the
official `0.6.15.post1` base `de358e829443d2cab2261b4c2b403444f6925ab6`.
The CUDA implementation comes from `1c73959e6dd7597186d0aba06bbe61015382c047`.
The new TMA path and the original-operator regressions pass on the tested
SM120 environment described below.

## Implementation and compatibility

- `flashinfer.prefill.fmha_v2_prefill_glm` accepts packed head-major E4M3
  Q/K/V with head dimensions 256/256, cumulative Q/KV lengths, and per-head,
  per-64-token dequantization scales. Output is BF16 or FP16. Quantization
  runs upstream; causal masks align to the bottom right for cached prefixes.
- `flashinfer/jit/attention/fmha_v2_glm_sm120.py` uses an independent module
  URI. Only this module includes the eight private headers under
  `csrc/fmha_v2_glm_sm120` ahead of the original FMHA headers.
- The launcher instantiates only the GLM causal and padding variants for
  the two output types. It retains the full QK256/PV64 tiles, TMA transport,
  Q-register reuse, K/V prefetch, block-scaled online accumulation, and direct
  output stores from the source implementation.
- The released scalar FMHA generator and its cache URI are unchanged.
  Original decode, GEMM, MoE, quantization and utility kernels are unchanged.
  The package audit found 1127 original runtime source files unchanged and
  no missing original runtime sources.
- The public package version stays `0.6.15.post1`, preserving the existing
  cubin and JIT-cache version checks. Local wheels use build tag
  `1sm120g1c73959e`. An existing installation of the same version requires
  explicit reinstallation to pick up this wheel.
- `FLASHINFER_BUILD_NO_PIP=1` skips the build backend's automatic cuTile
  compiler dependency installation. It does not remove runtime functionality
  or change package requirements. The tested wheel used `BUILD_NVEP=0`,
  matching the single-GPU environment; multi-GPU EP was not validated.

SGLang's separate direct-integration patch imports this public API. This
backport does not contain SGLang code or its Triton FP8 decode implementation.
The new CUDA module is JIT compiled on first use and needs CUDA 12.8 or newer
and an SM120 GPU.

## Recorded validation, 2026-10-15

Environment: RTX PRO 5000 72GB Blackwell, Python 3.12.3, CUDA 12.9,
PyTorch 2.11.0+cu129, CUTLASS DSL 4.6.0, and the existing
`flashinfer-cubin` / `flashinfer-jit-cache` 0.6.15.post1 packages.
The wheel was installed into a separate virtual environment. The system
installation was not replaced. The model service was stopped by the user
before the complete GPU regression run.

| Check | Result |
| --- | --- |
| Wheel build, installation and package-file audit | Passed; 4801 installed package files matched the wheel |
| Dependencies and old cubin/JIT-cache selection | Unchanged requirements; no new dependency conflicts |
| CPU trace consistency, initialization and GLM output metadata | 734 passed, 150 skipped |
| Original GPU operators | 11 passed |
| SGLang quantization preparation and Triton FP8 decode | 4 passed, plus 3 subtests |
| SGLang direct prefill/decode dispatch and installed import | 4 passed |
| New GLM CUDA module | JIT compilation passed |
| New TMA GLM numerical, prefix/ragged, graph and input-contract tests | 25 passed |
| Combined GPU regression suite | 44 passed, plus 3 subtests |

The 11 original-operator regressions are retained in
`tests/attention/test_fmha_v2_sm120_backport_regression.py`. They cover FA2
prefill, native decode, the original BF16/FP8 DeepSeek FMHA API, RMSNorm,
SiLU, sampling, BF16 GEMM, NVFP4 quantization/GEMM, and CUTLASS NVFP4 MoE
with both BF16 and quantized inputs.

The initial TMA probes reported CUDA out-of-memory errors while the existing
service occupied approximately 71.5 GiB and only about 1.3 GiB remained free.
After the user stopped that service, the unchanged CUDA implementation passed
the complete 44-test suite. These results cover operator correctness and the
direct SGLang attention paths. No full-model serving, end-to-end performance
or multi-GPU validation is claimed.
