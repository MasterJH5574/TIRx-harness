#!/usr/bin/env python3
"""FlashInfer trtllm-gen DeepSeek-V4 sparse MLA decode baseline."""

from __future__ import annotations

import torch
from flashinfer.decode import trtllm_batch_decode_sparse_mla_dsv4

_WORKSPACE = {}


def _workspace(device):
    key = str(device)
    if key not in _WORKSPACE:
        _WORKSPACE[key] = torch.zeros(128 * 1024 * 1024, dtype=torch.int8, device=device)
    return _WORKSPACE[key]


def _kernel_kwargs(
    query,
    swa_kv_cache,
    compressed_kv_cache,
    sparse_indices,
    sparse_topk_lens,
    seq_lens,
    cum_seq_lens_q,
    max_q_len,
    sinks,
    bmm1_scale,
    bmm2_scale,
    kv_layout="HND",
):
    """Public-wrapper arguments; an absent compressed pool is stood in for by
    the SWA pool, and on fp8 rows the scales travel as fp32 device tensors
    (upstream's own test does both)."""

    if query.dtype == torch.float8_e4m3fn:
        bmm1_scale = torch.tensor([float(bmm1_scale)], dtype=torch.float32, device=query.device)
        bmm2_scale = torch.tensor([float(bmm2_scale)], dtype=torch.float32, device=query.device)
    else:
        bmm1_scale, bmm2_scale = float(bmm1_scale), float(bmm2_scale)
    return dict(
        query=query,
        swa_kv_cache=swa_kv_cache,
        workspace_buffer=_workspace(query.device),
        sparse_indices=sparse_indices,
        compressed_kv_cache=swa_kv_cache if compressed_kv_cache is None else compressed_kv_cache,
        sparse_topk_lens=sparse_topk_lens,
        seq_lens=seq_lens,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        sinks=sinks,
        kv_layout=str(kv_layout),
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=None if max_q_len is None else int(max_q_len),
        # PR #4573 reports GPU-active sums, which count each kernel's full
        # duration; PDL overlap between the wrapper's launches would undercount.
        enable_pdl=False,
    )


def _launch(kwargs):
    return trtllm_batch_decode_sparse_mla_dsv4(**kwargs)


@torch.no_grad()
def prepare(*contract_args):
    """Capture the public wrapper's call in a CUDA graph outside timing.

    The wrapper issues three launches per call (q_lens difference, counter
    memset, attention); eager launching leaves host-dependent gaps between
    them inside the harness's CUPTI span. Replaying a graph times the kernels
    alone, the quantity the PR's GPU-active column reports, with the same
    three-kernel composition. The output is bitwise equal to the eager call.
    """

    kwargs = _kernel_kwargs(*contract_args)
    _launch(kwargs)  # JIT load, cubin fetch, workspace
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _launch(kwargs)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = _launch(kwargs)
    # kwargs travels with the graph: it owns the tensors the graph reads.
    return graph, output, kwargs


@torch.no_grad()
def run_prepared(graph, output, kwargs):
    """Kernel-only dispatch: replay the graph captured by :func:`prepare`."""

    graph.replay()
    return output


@torch.no_grad()
def run(
    query,
    swa_kv_cache,
    compressed_kv_cache,
    sparse_indices,
    sparse_topk_lens,
    seq_lens,
    cum_seq_lens_q,
    max_q_len,
    sinks,
    bmm1_scale,
    bmm2_scale,
    kv_layout="HND",
):
    """Call FlashInfer's trtllm-gen DSv4 sparse MLA decode kernel (eager)."""

    return _launch(
        _kernel_kwargs(
            query,
            swa_kv_cache,
            compressed_kv_cache,
            sparse_indices,
            sparse_topk_lens,
            seq_lens,
            cum_seq_lens_q,
            max_q_len,
            sinks,
            bmm1_scale,
            bmm2_scale,
            kv_layout,
        )
    )
