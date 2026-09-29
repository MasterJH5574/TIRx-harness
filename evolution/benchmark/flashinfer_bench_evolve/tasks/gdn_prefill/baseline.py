#!/usr/bin/env python3
"""FlashInfer ``chunk_gated_delta_rule`` baseline for GDN prefill."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from flashinfer.gdn_prefill import chunk_gated_delta_rule


@torch.no_grad()
def prepare(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """Hoist the per-call gate activation out of the timed region.

    The published API takes the forget gate in linear space (alpha in (0, 1]);
    exp() folds the contract's log-space gate -exp(A_log) * softplus(a + dt_bias).
    Candidates may precompute gates in their own prepare step outside timing —
    the baseline must be timed on the same scope."""

    alpha = torch.exp(-torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias.float()))
    beta = torch.sigmoid(b.float())
    return q, k, v, state, alpha, beta, cu_seqlens, float(scale)


@torch.no_grad()
def run_prepared(q, k, v, state, alpha, beta, cu_seqlens, scale):
    """Kernel-only dispatch on arguments produced by :func:`prepare`."""

    return chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=alpha,
        beta=beta,
        scale=scale,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )


@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """Call FlashInfer's GDN prefill kernel directly."""

    return run_prepared(*prepare(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale))
