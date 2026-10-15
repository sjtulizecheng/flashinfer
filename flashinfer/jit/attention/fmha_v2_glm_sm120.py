# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Isolated GLM backport; the 0.6.15 FMHA generator and cache stay unchanged."""

import jinja2

from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec
from ..utils import write_if_different
from ...compilation_context import CompilationContext


def gen_fmha_v2_glm_sm120_module() -> JitSpec:
    # Never reuse trtllm_fmha_v2: released AOT wheels contain the old interface.
    uri = "fmha_v2_glm_sm120_0615_v1"
    gen_dir = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    gen_dir.mkdir(parents=True, exist_ok=True)
    csrc = jit_env.FLASHINFER_CSRC_DIR
    template = jinja2.Template((csrc / "fmha_v2_glm_sm120_kernel.jinja").read_text())
    sources = []
    for output_dtype in ("bf16", "fp16"):
        path = gen_dir / f"glm_{output_dtype}.cu"
        write_if_different(path, template.render(output_dtype=output_dtype))
        sources.append(path)
    binding = gen_dir / "binding.cu"
    write_if_different(binding, (csrc / "fmha_v2_glm_sm120_binding.cu").read_text())
    sources.append(binding)
    flags = CompilationContext().get_nvcc_flags_list(supported_major_versions=[12])
    flags += [
        f"-I{csrc / 'fmha_v2_glm_sm120'}",
        f"-I{csrc / 'fmha_v2'}",
        "-Wno-deprecated-gpu-targets",
    ]
    return gen_jit_spec(uri, sources, extra_cuda_cflags=flags)
