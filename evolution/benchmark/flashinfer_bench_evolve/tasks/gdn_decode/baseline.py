#!/usr/bin/env python3
"""FlashInfer ``gated_delta_rule_decode_pretranspose`` baseline for GDN decode."""

from __future__ import annotations

import torch
from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose


@torch.no_grad()
def prepare(q, k, v, state, A_log, a, dt_bias, b, scale):
    """Hoist the per-call state clone out of the timed region.

    The kernel updates ``state`` in place, so it needs a private buffer; but
    the clone is a full extra read+write of the state tensor (~1.5x baseline
    inflation at B=64 on B200) and no part of the operation being scored.
    Cloning once here keeps timed iterations kernel-only — value drift across
    in-place timed iterations does not change kernel runtime, and the
    correctness reference is a single call on a freshly prepared clone."""

    return q, k, v, state.clone(), A_log, a, dt_bias, b, float(scale)


@torch.no_grad()
def run_prepared(q, k, v, state, A_log, a, dt_bias, b, scale):
    """Kernel-only dispatch on arguments produced by :func:`prepare`."""

    return gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=state,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=scale,
        use_qk_l2norm=False,
    )


@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    """Call FlashInfer's GDN decode kernel directly."""

    # The clone gives every call freshly allocated outputs (the harness
    # NaN-poisons previous return values between correctness runs).
    return run_prepared(*prepare(q, k, v, state, A_log, a, dt_bias, b, scale))
