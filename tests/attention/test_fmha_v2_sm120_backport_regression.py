"""Representative released operators; errors are failures, never silent skips."""

import math

import flashinfer as fi
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 backport regression",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_prefill_decode(dtype):
    torch.manual_seed(11)
    q = torch.randn(17, 4, 128, device="cuda", dtype=dtype)
    k = torch.randn(65, 2, 128, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    logits = torch.einsum(
        "qhd,khd->hqk", q.float(), k.float().repeat_interleave(2, 1)
    ) / math.sqrt(128)
    mask = (
        torch.arange(65, device="cuda")[None, :]
        > torch.arange(17, device="cuda")[:, None] + 48
    )
    logits.masked_fill_(mask, -float("inf"))
    ref = (
        logits.softmax(-1) @ v.float().repeat_interleave(2, 1).transpose(0, 1)
    ).transpose(0, 1)
    actual = fi.single_prefill_with_kv_cache(q, k, v, causal=True, backend="fa2")
    torch.testing.assert_close(actual.float(), ref, rtol=0.02, atol=0.003)
    actual = fi.single_decode_with_kv_cache(q[-1], k, v)
    torch.testing.assert_close(actual.float(), ref[-1], rtol=0.02, atol=0.003)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_original_fmha_deepseek(dtype):
    from flashinfer.prefill import fmha_v2_prefill_deepseek

    torch.manual_seed(13)
    q = (torch.randn(1, 64, 4, 192, device="cuda") * 0.25).to(dtype)
    k = (torch.randn(1, 64, 4, 192, device="cuda") * 0.25).to(dtype)
    v = torch.randn(1, 64, 4, 128, device="cuda").to(dtype)
    out = torch.empty(1, 64, 4, 128, device="cuda", dtype=torch.bfloat16)
    scale = 1 / math.sqrt(192)
    # Original API is causal and folds QK normalization into scale_bmm1 for FP8.
    fmha_v2_prefill_deepseek(
        q, k, v, out, 4, 192, 64, 1.0, scale_bmm1=scale, scale_bmm2=1.0
    )
    logits = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * scale
    logits.masked_fill_(
        torch.ones(64, 64, device="cuda", dtype=torch.bool).triu(1), -float("inf")
    )
    ref = torch.einsum("bhqk,bkhd->bqhd", logits.softmax(-1), v.float())
    assert (out.float() - ref).norm() / ref.norm() < 0.035


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_norm_activation(dtype):
    torch.manual_seed(17)
    x = torch.randn(16, 256, device="cuda", dtype=dtype)
    weight = torch.randn(256, device="cuda", dtype=dtype)
    ref = (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        * weight.float()
    )
    torch.testing.assert_close(fi.rmsnorm(x, weight).float(), ref, rtol=0.02, atol=0.02)
    actual = fi.silu_and_mul(x)
    ref = F.silu(x[:, :128].float()) * x[:, 128:].float()
    torch.testing.assert_close(actual.float(), ref, rtol=0.02, atol=0.02)


def test_sampling():
    # One-hot distributions have an exact expected result regardless of RNG.
    ids = torch.arange(16, device="cuda") * 7
    probs = torch.zeros(16, 256, device="cuda")
    probs.scatter_(1, ids[:, None], 1.0)
    torch.testing.assert_close(fi.sampling_from_probs(probs).long(), ids)
    torch.testing.assert_close(fi.top_k_sampling_from_probs(probs, 1).long(), ids)


def test_bf16_gemm():
    torch.manual_seed(19)
    a = torch.randn(32, 256, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda").T
    out = fi.mm_bf16(a, b, backend="cublaslt")
    torch.testing.assert_close(out, a @ b, atol=0.25, rtol=0.02)


def test_nvfp4_quantize_gemm():
    torch.manual_seed(23)
    a = torch.randn(32, 256, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    sa = (448 * 6) / a.float().abs().max()
    sb = (448 * 6) / b.float().abs().max()
    aq, asc = fi.fp4_quantize(a, sa)
    bq, bsc = fi.fp4_quantize(b, sb)
    out = fi.mm_fp4(aq, bq.T, asc, bsc.T, 1 / (sa * sb), backend="cutlass")
    ref = a @ b.T
    assert (
        F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0) > 0.97
    )


@pytest.mark.parametrize("quantized_input", [False, True])
def test_cutlass_nvfp4_moe(quantized_input):
    # Reuse upstream's independent block dequantization and expert reference.
    from flashinfer.fused_moe.core import ActivationType
    from tests.moe.test_trtllm_cutlass_fused_moe import test_moe_nvfp4

    test_moe_nvfp4(
        batch_size=8,
        hidden_size=256,
        num_experts=4,
        top_k=2,
        intermediate_size=128,
        otype=torch.bfloat16,
        wtype=torch.uint8,
        quantized_input=quantized_input,
        activation_type=ActivationType.Swiglu,
        use_4over6=False,
    )
