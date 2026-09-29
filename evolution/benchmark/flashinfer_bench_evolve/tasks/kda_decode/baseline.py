#!/usr/bin/env python3
"""FlashInfer ``recurrent_kda`` (CuTe-DSL route) baseline for KDA decode."""

from __future__ import annotations

import torch
from flashinfer.kda_decode import recurrent_kda


@torch.no_grad()
def prepare(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    scale,
    initial_state,
    cu_seqlens,
    ssm_state_indices,
    num_spec_tokens,
    num_accepted_tokens,
    lower_bound,
):
    """Hoist the state-pool copy and the output allocation out of timing.

    The kernel updates its state pool in place; the contract returns a new
    pool and leaves ``initial_state`` untouched, so the copy is made here
    once and the timed call is the kernel alone (gdn_decode precedent).
    Like flashinfer PR #4279's upstream arm, the output buffer is
    caller-owned and allocated here too."""

    return (
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        float(scale),
        initial_state.clone(),
        cu_seqlens,
        ssm_state_indices,
        num_spec_tokens,
        num_accepted_tokens,
        lower_bound,
        torch.empty_like(v),
    )


@torch.no_grad()
def run_prepared(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    scale,
    state,
    cu_seqlens,
    ssm_state_indices,
    num_spec_tokens,
    num_accepted_tokens,
    lower_bound,
    output,
):
    """Kernel-only dispatch on arguments produced by :func:`prepare`.

    This is the call form PR #4279's benchmark times for ``recurrent_kda``:
    a caller-owned ``output`` and an in-place update of ``state``."""

    output, _ = recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=state,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=A_log is not None,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=num_spec_tokens,
        num_accepted_tokens=num_accepted_tokens,
        output=output,
        backend="cute-dsl",
    )
    return output, state


@torch.no_grad()
def run(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    scale,
    initial_state,
    cu_seqlens,
    ssm_state_indices,
    num_spec_tokens,
    num_accepted_tokens,
    lower_bound,
):
    """Call FlashInfer's recurrent KDA decode kernel on a fresh state copy."""

    return run_prepared(
        *prepare(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            scale,
            initial_state,
            cu_seqlens,
            ssm_state_indices,
            num_spec_tokens,
            num_accepted_tokens,
            lower_bound,
        )
    )
