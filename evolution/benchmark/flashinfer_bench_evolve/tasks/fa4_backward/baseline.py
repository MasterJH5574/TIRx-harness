#!/usr/bin/env python3
"""Direct FlashAttention-4 backward baseline for prepared forward tensors."""

from __future__ import annotations


def run(q, k, v, out, grad_out, lse, causal, sm_scale):
    """Call the low-level FA4 backward operator directly."""

    from flash_attn.cute.interface import _flash_attn_bwd

    return _flash_attn_bwd(
        q=q,
        k=k,
        v=v,
        out=out,
        dout=grad_out,
        lse=lse,
        softmax_scale=float(sm_scale),
        causal=bool(causal),
    )
