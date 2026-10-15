"""The optional output defaults to BF16; an explicit FP16 output stays FP16."""

import json

import pytest
import torch

from flashinfer import fmha_v2_prefill_glm


@pytest.mark.parametrize("dtype", [None, torch.bfloat16, torch.float16])
def test_glm_trace_output_dtype(dtype, tmp_path):
    q = torch.empty(20, 64, 256, dtype=torch.float8_e4m3fn).transpose(0, 1)
    cu = torch.tensor([0, 64], dtype=torch.int32)
    scale = torch.ones(1, 20, 1)
    result = fmha_v2_prefill_glm.fi_trace(
        query=q,
        key=q,
        value=q,
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        q_scale=scale,
        k_scale=scale,
        v_scale=scale,
        max_seqlen_q=64,
        max_seqlen_kv=64,
        out=None if dtype is None else torch.empty(q.shape, dtype=dtype),
        save_dir=str(tmp_path),
    )
    expected = "float16" if dtype == torch.float16 else "bfloat16"
    assert result["outputs"]["out"]["dtype"] == expected
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["outputs"]["out"]["dtype"] == expected
