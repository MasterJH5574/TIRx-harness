#!/usr/bin/env python3
"""FlashAttention-4 (``flash_attn.cute``) baseline for Wan hybrid attention.

Calls FA4 the way flashinfer PR #4543's ``bench_wan_hybrid_attention.py``
"production SGLang FA4" arm does: the private ``_flash_attn_fwd`` entry point
with a caller-owned output tensor and ``pack_gqa=False``.
"""

from __future__ import annotations

import torch
from flash_attn.cute.interface import _flash_attn_fwd


@torch.no_grad()
def run(q, k, v, sm_scale):
    """Dense non-causal bf16 attention, PR #4543's production FA4 arm.

    ``_flash_attn_fwd(q, k, v, out=..., pack_gqa=False)`` with the contract's
    ``sm_scale`` passed explicitly (the PR relies on the default, which equals
    ``1/sqrt(128)`` for this shape). The tuple it returns is ignored; ``out``
    is written in place.
    """

    out = torch.empty_like(q)
    _flash_attn_fwd(q, k, v, softmax_scale=float(sm_scale), out=out, pack_gqa=False)
    return out
