#!/usr/bin/env python3
"""flash-attn baseline for fp16 FA4 forward."""

from __future__ import annotations

import torch
from flash_attn import flash_attn_func


@torch.no_grad()
def run(q, k, v, causal, sm_scale):
    return flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=float(sm_scale),
        causal=bool(causal),
    )
